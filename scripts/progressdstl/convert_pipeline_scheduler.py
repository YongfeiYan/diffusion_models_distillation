
import argparse

from diffdstl import DistillDDIMScheduler, LDMCodeWrapperPipeline


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model', type=str)
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    print('args', args)
    
    pipeline = LDMCodeWrapperPipeline.from_pretrained(args.pretrained_model)
    config = pipeline.scheduler.config
    print('scheduler.config:', config)
    
    scheduler = DistillDDIMScheduler(**config)
    pipeline.register_modules(scheduler=scheduler)
    pipeline.register_to_config(scheduler=(scheduler.__module__, scheduler.__class__.__name__))
    print('pipeline', pipeline)
    pipeline.save_pretrained(args.save_dir)
    
    print('finished')
