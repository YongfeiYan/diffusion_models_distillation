import os
import time
import glob
import torch 
import shutil
import logging 
import numpy as np
from torch.utils.data import Dataset


logger = logging.getLogger()


class DebugDataset(Dataset):
    def __init__(self, dataset, n_examples, seed):
        rnd = random.Random(seed)
        self._indexes = list(range(len(dataset)))
        rnd.shuffle(self._indexes)
        self._n_examples = min(n_examples, len(self._indexes))
        self.dataset = dataset
        logger.info('DebugDataset uses {}/{} examples with seed {}'.format(self._n_examples, len(dataset), seed))

    def __len__(self):
        return self._n_examples

    def __getitem__(self, index):
        item = self.dataset[index]
        origin_index = 'origin_index'
        assert origin_index not in item
        item[origin_index] = index
        return item


class FolderReplayDatasetReader(Dataset):
    def __init__(self, writer):
        self.data_dir = writer.data_dir 
        self.files = sorted(glob.glob('{}/*.pt'.format(self.data_dir)))
        logger.info('len files: {}'.format(len(self.files)))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        return torch.load(path)

    def clear(self):
        pass


class FolderReplayDatasetWriter:
    def __init__(self, data_dir, worker_id, n_workers):
        assert 'replay' in data_dir, 'Naming restriction: {}'.format(data_dir)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._keys = None
        self._key_shapes = {}
        self._key_dtypes = {}
        self._total_count = 0
        self._batch_count = 0
        self._all_saved = False
        self._parquet_writer = None
        self._shard_idx = worker_id
        self._n_workers = n_workers
        self._schema = None
        self._cost_time = 0

    @property
    def dtypes_shapes_to_read_data(self):
        return None, None
    
    @property 
    def batch_num(self):
        return self._batch_count 
    
    def __len__(self):
        return self._total_count

    def add_batch(self, batch):
        begin_time = time.time()
        assert isinstance(batch, dict), 'Batch is supposed to be a dict, but found {}'.format(type(batch))
        batch_keys = set(list(batch.keys()))
        if self._keys is None:
            self._keys = batch_keys
        assert self._keys == batch_keys, (self._keys.symmetric_difference(batch_keys))
        
        batch = {k: v.cpu() for k, v in batch.items()}
        batch_size = None
        for k, v in batch.items():
            batch_size = v.shape[0]
            break
        self._total_count += batch_size
        self._batch_count += 1
        
        for i in range(batch_size):
            sample = {k: batch[k][i].clone() for k in batch}  # use clone, or the whole batch will be saved multiple times
            file = '{}/{:010d}.pt'.format(self.data_dir, self._shard_idx)
            torch.save(sample, file)
            self._shard_idx += self._n_workers
        
        self._cost_time += time.time() - begin_time

    def prepare_reading(self):
        logger.info('data_dir {}, dtype {}, batches {}, total count {}, shards_last_idx {}, time adding batches {}s'.format(self.data_dir, self._key_dtypes, self._batch_count, self._total_count, self._shard_idx, self._cost_time))
    
    def clear(self):
        assert 'replay' in self.data_dir, self.data_dir
        shutil.rmtree(self.data_dir, ignore_errors=True)


if __name__ == '__main__':
    import random
    n = 5
    data = FolderReplayDatasetWriter('/tmp/test/replay/data', worker_id=0, n_workers=2)
    for _ in range(n):
        bsz = random.randint(2, 5)
        a = torch.arange(bsz, device='cpu', dtype=torch.float32) + len(data)
        b = a * 2
        batch = {'a': a, 'b': b}
        data.add_batch(batch)
        print(batch)
    data2 = FolderReplayDatasetWriter('/tmp/test/replay/data', worker_id=1, n_workers=2)
    for _ in range(n):
        bsz = random.randint(2, 5)
        a = torch.arange(bsz, device='cpu', dtype=torch.float32) + len(data2) + len(data)
        b = a * 2
        batch = {'a': a, 'b': b}
        data2.add_batch(batch)
        print(batch)
    data.prepare_reading()
    data2.prepare_reading()
    dtypes, shapes = data.dtypes_shapes_to_read_data

    reader = FolderReplayDatasetReader(data)

    for index in range(len(reader)):
        print(reader[index], reader[index]['a'].device)
    input('check files')
    data.clear()
    reader.clear()
    data2.clear()
    input('check memory')
