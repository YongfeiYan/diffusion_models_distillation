import os
import sys 
import torch
import argparse

from diffusers import DiffusionPipeline
from diffdstl import LDMCodeWrapperPipeline
from diffdstl.train.make_image_grid import make_image_grid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model', type=str, )
    parser.add_argument('prompt_path', type=str, )
    parser.add_argument('save_dir', type=str, )
    parser.add_argument('--is_text2image', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--uncond', type=str, default=None)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--prediction_type', type=str, )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--img_cols', type=int, default=4)
    args = parser.parse_args()
    print('args', args)

    images = []
    os.makedirs(args.save_dir, exist_ok=True)
    if args.is_text2image:
        pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model).to(args.device)
    else:
        pipeline = LDMCodeWrapperPipeline.from_pretrained(args.pretrained_model).to(args.device)
    if args.prediction_type:
        pipeline.scheduler.register_to_config(prediction_type=args.prediction_type)
        print('Set prediction_type:')
        print(pipeline.scheduler)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    with open(args.prompt_path) as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            prompt = line.rstrip('\n')
            print('prompt {}/{}:'.format(i + 1, len(lines)), prompt)
        
            with torch.no_grad():
                if args.is_text2image:
                    image = pipeline(prompt, num_inference_steps=args.num_inference_steps, generator=generator).images[0]
                else:
                    cond = torch.full((1,), fill_value=int(prompt), dtype=torch.long, device=args.device)
                    uncond = torch.full_like(cond, fill_value=int(args.uncond))
                    image = pipeline(cond.shape[0], cond=cond, uncond=uncond, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(args.save_dir + '/{:03d}.png'.format(i))
            images.append(image)
        
        grid = make_image_grid(images, len(images) // args.img_cols, args.img_cols)
        grid.save(os.path.abspath(args.save_dir) + '/grid.png')
        print('save_dir:', args.save_dir)

"""
PYTHONPATH=.:third_party/stable-diffusion/ python -u scripts/progressdstl/sample_pipeline.py data/log/imagenet/stage_one/stage_one_convert_scheduler/ configs/imagenet/prompts_imagenet.txt data/log/imagenet/stage_one/images/step4 --uncond 1000 --num_inference_steps 4 --guidance_scale 1.0
"""
