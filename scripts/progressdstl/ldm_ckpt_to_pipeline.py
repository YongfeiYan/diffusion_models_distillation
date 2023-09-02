import os
import torch
import argparse
from diffdstl import LDMCodeWrapperPipeline
from diffdstl.diffusers_pipeline import CACHE_MODELS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
print('args', args)

device = args.device
save_dir = args.save_dir
pipeline = LDMCodeWrapperPipeline.build(args.config_path, args.ckpt_path).to(device)

assert len(CACHE_MODELS) == 1
ldm_model = list(CACHE_MODELS.values())[0]

print('use ema weights')
ldm_model.ema_scope().__enter__()  # use ema weights

cond = torch.LongTensor([25, 187, 187, 187, 448, 992]).to(device=device)
uncond = torch.ones_like(cond) * 1000

print('save pipeline')
pipeline_dir = save_dir
pipeline.unet.register_to_config(ldm_ckpt_path=None)
pipeline.save_pretrained(pipeline_dir)

print('load pipeline')
pipeline = LDMCodeWrapperPipeline.from_pretrained(pipeline_dir).to(device)

print('sample')
output = pipeline(batch_size=cond.shape[0], cond=cond, uncond=uncond, guidance_scale=3)
os.makedirs(save_dir + '/images', exist_ok=True)
for i, image in enumerate(output.images):
    image.save('{}/{:05d}.png'.format(save_dir, i))

print('finished, save pipeline to', pipeline_dir, '\nSave sampled images to', save_dir + '/images')

"""
Run:
PYTHONPATH=.:third_party/stable-diffusion/ python scripts/progressdstl/ldm_ckpt_to_pipeline.py configs/imagenet/cin256-v2.yaml data/ldm/cin256-v2/model.ckpt data/test-pipeline
"""
