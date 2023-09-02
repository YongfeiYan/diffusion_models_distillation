#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import json 
import argparse
import logging
import math
import os
import gc
import random
import shutil
from pathlib import Path
from PIL import Image 
import io
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from diffdstl import LDMCodeWrapperPipeline, UNetWrapper, EncoderDecoderWrapper, DistillDDIMScheduler
from diffdstl.data.replay import FolderReplayDatasetReader, FolderReplayDatasetWriter


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class RunMode:
    FINETUNE = 'finetune_v_prediction'
    STAGE_TWO = 'stage_two'  # less sampling steps
    STAGE_ONE = 'stage_one'  # classifier-free guidance removal


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def run_validation(args, accelerator, vae, unet, ema_unet, th_unet, optimizer, lr_scheduler, th_scheduler, st_scheduler, dataloader, progress_bar, global_step, epoch, best_metrics, weight_dtype=torch.float32, loss_key='eval_st_loss'):
    logger.info("Generting image samples ... ")
    torch.cuda.empty_cache()

    if args.use_ema:
        logger.info('Switch ema_unet parameters')
        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())

    images = []
    if accelerator.is_main_process:
        # check parameter dtype 
        ema_dtype = ema_unet.shadow_params[0].dtype
        logger.info('EMA weight dtype: {}'.format(ema_dtype))
        assert ema_dtype == torch.float32, ema_dtype
        unet_dtype = next(unet.parameters()).dtype
        logger.info('Unet weight dtype: {}'.format(unet_dtype))

        pipeline = LDMCodeWrapperPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            torch_type=weight_dtype,
        )
        del pipeline.scheduler
        pipeline.scheduler = type(st_scheduler)(**st_scheduler.config)
        
        for k, v in st_scheduler.config.items():
            assert pipeline.scheduler.config[k] == v, (k, v, pipeline.scheduler.config[k])
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        image_dir = os.path.join(args.output_dir, 'images/epoch{}-step{}/'.format(epoch, global_step))
        os.makedirs(image_dir, exist_ok=True)
        for i in range(len(args.validation_prompts)):
            with torch.autocast("cuda"), torch.no_grad():
                logger.info('prompt {}: {}'.format(i, args.validation_prompts[i]))
                guidance_scale = args.guidance_scale if args.run_mode == RunMode.FINETUNE else 1.0
                cond = torch.full((1,), fill_value=int(args.validation_prompts[i]), dtype=torch.long, device=accelerator.device)
                uncond = torch.full_like(cond, fill_value=int(args.uncond))
                image = pipeline(cond.shape[0], cond=cond, uncond=uncond, num_inference_steps=args.student_steps, guidance_scale=guidance_scale, generator=generator).images[0]
                image.save(image_dir + '/{:03d}.png'.format(i))
            images.append(image)
        grid = make_image_grid(images, len(images) // 4, 4)
        grid.save(os.path.abspath(image_dir) + '.grid.png')

        pipeline.save_pretrained(os.path.join(args.output_dir, args.pipeline_save_name))
        del pipeline
        torch.cuda.empty_cache()
    
    # Evaluate loss
    logger.info('Evaluating on eval_dataloader ... ')
    with torch.no_grad():
        _, n_samples, metrics = run_epoch(args, vae=vae, accelerator=accelerator,
            unet=unet, ema_unet=ema_unet, th_unet=th_unet, optimizer=optimizer, lr_scheduler=lr_scheduler, 
            th_scheduler=th_scheduler, st_scheduler=st_scheduler, dataloader=dataloader, 
            progress_bar=progress_bar, global_step=global_step, is_train=False, weight_dtype=weight_dtype)
        if accelerator.is_local_main_process:
            logger.info('Eval epoch {}, n_samples {}, metrics {}'.format(epoch, n_samples, metrics))
    
    if best_metrics is None or best_metrics[loss_key] >= metrics[loss_key]:
        if accelerator.is_local_main_process:
            logger.info('Found better {}: {}, and save bestcheckpoint'.format(loss_key, metrics[loss_key]))
            save_path = os.path.join(args.output_dir, f"best-ckpt")
            accelerator.save_state(save_path)
        best_metrics = metrics
    
    if args.use_ema:
        logger.info('Switch unet parameters back from ema_unet')
        ema_unet.restore(unet.parameters())

    logger.info('Evaluation finished.')
    
    return images, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",        type=str,        default=None,        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        '--pipeline_save_name', type=str, required=True,
        help='Pipeline saved to output_dir'
    )
    parser.add_argument(
        "--unet_config_path",         type=str,         required=False, default='',
        help='Student unet config for distillation.'
    )
    parser.add_argument(
        "--puncond",         type=float,         default=0.1,
        help='Classifier-free guidance of unconditional sample rate'
    )
    parser.add_argument(
        '--uncond', type=str, default='',
        help='uncond for classifier-free guidance'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=5.0,
        help=' ',
    )
    parser.add_argument(
        '--student_steps', type=int, required=True, 
        help='Students'
    )
    parser.add_argument(
        '--run_mode', type=str, choices=[RunMode.FINETUNE, RunMode.STAGE_ONE, RunMode.STAGE_ONE], default=RunMode.FINETUNE,
    )
    parser.add_argument(
        '--origin_loss_weight', type=float, default=1.0,
    )
    parser.add_argument(
        "--distill_loss_weight",        type=float,        default=1.0,
    )
    parser.add_argument(
        "--revision",        type=str,        default=None,        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default='imagenet', choices=['imagenet', 'mnist', 'laion'],
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
            "example: imagenet/mnist/laion"
        ),
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None,
    )
    parser.add_argument(
        '--replay_data_type', type=str, default='folder',
    )
    parser.add_argument(
        '--replay-batches', type=int, default=1000,
    )
    parser.add_argument(
        '--replay-times', type=int, default=4, 
    )
    parser.add_argument(
        '--replay-num-workers', type=int, default=0,
    )
    parser.add_argument(
        "--loss_key", 
        type=str, default='eval_st_loss',
    )
    parser.add_argument(
        '--use_debug_examples', type=int, default=None,
        help='select N examples of dataset for training'
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompt_path", type=str, default=None, 
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir", type=str, default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution", type=int, default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", default=False, action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--encoder_batch_size", type=int, default=12, help='Split the teacher"s batch to predict.'
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",        type=float,        default=1e-5,        
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",        action="store_true",        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",        type=str,        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        '--snr_weight_mode', type=str, default='', choices=['', 'min', 'max', 'truncated', 'plus1'],
        help=''
    )
    parser.add_argument(
        "--snr_gamma",        type=float,        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
    )
    parser.add_argument(
        "--non_ema_revision", type=str,         default=None,        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",         type=int,        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tracker-log",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    
    if args.unet_config_path:
        with open(args.unet_config_path) as f:
            args.unet_config = json.load(f)
    else:
        args.unet_config = ''

    if args.validation_prompt_path:
        with open(args.validation_prompt_path) as f:
            args.validation_prompts = [line.rstrip('\n') for line in f]
    else:
        args.validation_prompts = None

    return args


def calculate_loss(model_pred, target, snr_weights, agg='mean'):
    assert agg in ('mean', 'sum_batch_wise')
    if snr_weights is None:
        if agg == 'mean':
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none').sum(dim=0).mean()
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        # snr = compute_snr(timesteps)
        # mse_loss_weights = (
        #     torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        # )
        # We first calculate the original loss. Then we mean over the non-batch dimensions and
        # rebalance the sample-wise losses with their respective loss weights.
        # Finally, we take the mean of the rebalanced loss.
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weights
        if agg == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
    return loss


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class WeightedStatistic:
    """
    """
    def __init__(self, name, init, postprocessing=None, preprocessing=None):
        self.name = name
        self.init = init
        self.v = init
        self.w = 0
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def add(self, v, w):
        if self.preprocessing is not None:
            v, w = self.preprocessing(v, w)
        self.v = self.v + v * w
        self.w = self.w + w

    def get_weighted_value(self):
        if self.w == 0: # no accumulated value
            return 0
        v = self.v / self.w
        if self.postprocessing is not None:
            v = self.postprocessing(v)
        return v

    def clear(self):
        self.v = self.init
        self.w = 0

    def __repr__(self):
        return '{} {:.5f}'.format(self.name, self.get())


def get_distillation_data(args, accelerator, vae, th_unet, unet, th_scheduler, st_scheduler, weight_dtype, batch):
    with torch.no_grad(), accelerator.autocast():
        bsz = list(batch.values())[0].shape[0]
        mask = (torch.rand(bsz, dtype=torch.float32, device=accelerator.device) > args.puncond).to(dtype=weight_dtype)  # .reshape(bsz, 1, 1)

        latents = vae.encode(batch['image'].permute(0, 3, 1, 2).contiguous())
        cond = batch['class_label']
        uncond = torch.full_like(cond, fill_value=int(args.uncond))
        if args.run_mode == RunMode.FINETUNE:
            cond = (mask * cond + (1 - mask) * uncond).to(dtype=cond.dtype)
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=weight_dtype,
            )

        # Sample a random timestep for each image
        if args.run_mode == RunMode.STAGE_TWO:
            th_timesteps = th_scheduler.timesteps  # 2N
            st_timesteps = st_scheduler.timesteps  # N
            assert len(th_timesteps) == 2 * len(st_timesteps), (len(th_timesteps), len(st_timesteps))
            idx = torch.randint(0, len(st_timesteps), (bsz,), dtype=torch.long, device=st_timesteps.device)
            timesteps = st_timesteps[idx]
            th_timesteps, th_timesteps_1 = th_timesteps[2 * idx], th_timesteps[2 * idx + 1]
        else:
            timesteps = torch.randint(0, th_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = st_scheduler.add_noise(latents, noise, timesteps)

        # Convert to float to calculate loss
        if args.run_mode == RunMode.FINETUNE:
            th_pred = th_scheduler.get_velocity(latents, noise, timesteps)
        elif args.run_mode == RunMode.STAGE_ONE:
            cond_out = th_unet(noisy_latents, timesteps, cond).sample.float()
            uncond_out = th_unet(noisy_latents, timesteps, uncond).sample.float()
            th_pred = uncond_out + args.guidance_scale * (cond_out - uncond_out)
        elif args.run_mode == RunMode.STAGE_TWO:
            """
            paper        -    code of DDIMScheduler
            alpha_t **2  -    alpha_prod_t, self.alphas_cumprod[timestep]
            """
            out1 = th_unet(noisy_latents, th_timesteps, cond).sample.float()
            z1 = th_scheduler.step(out1, th_timesteps, noisy_latents).prev_sample.float()
            out2 = th_unet(z1, th_timesteps_1, cond).sample.float()
            z2 = th_scheduler.step(out2, th_timesteps_1, z1).prev_sample.float()
            alpha_prod_t, alpha_prod_t_prev = st_scheduler.get_alpha_cumprods(timesteps, out1.shape)
            alpha_t, alpha_t2 = alpha_prod_t ** .5, alpha_prod_t_prev ** .5
            sigma_t, sigma_t2 = (1 - alpha_prod_t) ** .5, (1 - alpha_prod_t_prev) ** .5
            sigma_div = sigma_t2 / sigma_t
            th_pred = (z2 - sigma_div * noisy_latents) / (alpha_t2 - sigma_div * alpha_t)

        batch = {
            'latents': latents,
            'mask': mask,
            'noisy_latents': noisy_latents,
            'noise': noise,
            'timesteps': timesteps,
            'th_pred': th_pred,
            'cond': cond,
        }

    return batch


def run_epoch(args, accelerator, unet, ema_unet, vae, th_unet, optimizer, lr_scheduler, th_scheduler, 
    st_scheduler, dataloader, progress_bar, global_step, is_train, weight_dtype):
    """
    Args:
        is_train: training or evaluation

    for batch in dataloader:
        if not_replay:
            get data 
            gather and save batch 
        else:
            create dataloader, prepare accelerator
            for batch in ds:
                run_batch()
    "if replaydata > 0: run_replay"
    """

    if is_train:
        unet.train()
    else:
        unet.eval()
    accumulate_steps = 1 if not is_train else args.gradient_accumulation_steps

    # Metrics
    prefix = 'train_' if is_train else 'eval_'
    m_loss = WeightedStatistic(prefix + 'loss', 0)
    m_ds_loss = WeightedStatistic(prefix + 'ds_loss', 0)
    m_st_loss = WeightedStatistic(prefix + 'st_loss', 0)
    m_th_loss = WeightedStatistic(prefix + 'th_loss', 0)
    m_n_cond = WeightedStatistic(prefix + 'n_cond', 0)
    m_cond_loss_sum = WeightedStatistic(prefix + 'cond_loss_sum', 0)
    train_loss = train_ds_loss = train_st_loss = \
        train_th_loss = train_n_cond = train_cond_loss_sum = train_n_samples = 0.0


    def run_replay(dataset):
        """Train student model using distillation data."""
        torch.cuda.empty_cache()
        nonlocal train_loss, train_ds_loss, train_st_loss, train_th_loss, train_n_cond, train_cond_loss_sum, train_n_samples, global_step
        
        replay_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=args.train_batch_size,
            num_workers=args.replay_num_workers,  # args.dataloader_num_workers,
            pin_memory=True,
        )
        replay_dataloader = accelerator.prepare_data_loader(replay_dataloader)
    
        for step, batch in enumerate(replay_dataloader):
            with accelerator.accumulate(unet):
                latents, noise, timesteps, noisy_latents, mask, th_pred, cond = \
                    batch['latents'], batch['noise'], batch['timesteps'], batch['noisy_latents'], \
                        batch['mask'], batch['th_pred'], batch['cond']
                n_cond = mask.sum()
                bsz = latents.shape[0]
                mask4 = mask.reshape(-1, 1, 1, 1).contiguous()  # B x 1 x 1 x 1

                if st_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif st_scheduler.config.prediction_type == "v_prediction":
                    target = st_scheduler.get_velocity(latents, noise, timesteps)

                # Predict the noise residual and compute loss
                # print(noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape)
                # convert to float32 to calucate loss, otherwise the loss may be too small when in fp16
                model_pred = unet(noisy_latents, timesteps, cond).sample.float()
                if args.run_mode == RunMode.STAGE_TWO:
                    alpha_prod_t, _ = st_scheduler.get_alpha_cumprods(timesteps, model_pred.shape)
                    x_pred = (alpha_prod_t ** 0.5) * noisy_latents - ((1 - alpha_prod_t) ** 0.5) * model_pred
                target = target.float()

                if not args.snr_weight_mode:
                    snr_weights = None
                elif args.snr_weight_mode == 'max':
                    snr = compute_snr(st_scheduler, timesteps)
                    snr_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                elif args.snr_weight_mode == 'truncated':
                    snr = compute_snr(st_scheduler, timesteps)
                    snr_weights = torch.stack([snr, torch.ones_like(timesteps)], dim=1).max(dim=1)[0]

                loss = calculate_loss(model_pred, target, snr_weights, agg='mean') * args.origin_loss_weight
                cond_loss_sum = calculate_loss(model_pred * mask4, target * mask4, snr_weights, agg='sum_batch_wise') * args.origin_loss_weight
                st_loss = loss 

                # Distillation loss
                if args.distill_loss_weight > 0:
                    if args.run_mode == RunMode.STAGE_TWO:
                        ds_loss = calculate_loss(x_pred, th_pred, snr_weights, agg='mean')
                        th_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
                    else:
                        th_loss = calculate_loss(th_pred, target, snr_weights, agg='mean')
                        ds_loss = F.mse_loss(model_pred, th_pred, reduce='mean')
                    loss = ds_loss * args.distill_loss_weight + loss
                else:
                    ds_loss = th_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)

                # Gather the losses across all processes for logging (if we use distributed training).
                def gather_scalar(v, bsz, agg='mean'):
                    avg = accelerator.gather(v.repeat(1))  # some process may have smaller batches
                    if agg == 'mean':
                        avg = avg.mean()
                    elif agg == 'sum':
                        avg = avg.sum()
                    else:
                        raise RuntimeError('Unknown agg: {}'.format(agg))
                    return avg.item() / accumulate_steps if agg == 'mean' else avg.item()

                train_loss += gather_scalar(loss, bsz)
                train_ds_loss += gather_scalar(ds_loss, bsz)
                train_st_loss += gather_scalar(st_loss, bsz)
                train_th_loss += gather_scalar(th_loss, bsz)
                train_n_cond += gather_scalar(n_cond, bsz, agg='sum')
                train_cond_loss_sum += gather_scalar(cond_loss_sum, bsz, agg='sum')
                assert len(timesteps.shape) == 1, timesteps.shape
                train_n_samples += gather_scalar(torch.ones_like(timesteps), bsz, 'sum')
                # accumulate metrics 
                if not is_train or (is_train and accelerator.sync_gradients):
                    for m, v in zip([m_loss, m_ds_loss, m_st_loss, m_th_loss, m_n_cond, m_cond_loss_sum],
                                    [train_loss, train_ds_loss, train_st_loss, train_th_loss, train_n_cond, train_cond_loss_sum]):
                        m.add(v, train_n_samples)
                
                if is_train:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    lr_scheduler.step()  # decrease each single step
                    optimizer.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if is_train and accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({
                            "train_loss": train_loss, "train_ds_loss": train_ds_loss, 
                            "train_st_loss": train_st_loss, "train_th_loss": train_th_loss, 
                            "train_n_cond": train_n_cond, "train_cond_loss_sum": train_cond_loss_sum,
                            "train_n_samples": train_n_samples,
                        }, 
                        step=global_step
                    )

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            if not is_train or (is_train and accelerator.sync_gradients):
                logs = {
                    "loss": train_loss, 
                    "st_loss": train_st_loss,
                    "ds_loss": train_ds_loss,
                    "th_loss": train_th_loss,
                    "n_cond": train_n_cond,
                    "cond_loss_sum": train_cond_loss_sum,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if is_train:
                    progress_bar.update(1)
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix(**logs)
                train_loss = train_ds_loss = train_st_loss = train_th_loss = \
                    train_n_cond = train_cond_loss_sum = train_n_samples = 0.0
    
    # Init replay writer
    replay_times = 1 if not is_train else args.replay_times
    replay_data_dir = '{}/replay/data/'.format(args.output_dir)
    if accelerator.is_local_main_process:
        logger.info('local_main_process: makedirs(replay)')
        os.makedirs(args.output_dir + '/replay/data/', exist_ok=True)
    accelerator.wait_for_everyone()
    
    if args.replay_data_type == 'folder':
        logger.info('using folder replay')
        Writer, Reader = FolderReplayDatasetWriter, FolderReplayDatasetReader
    else:
        raise RuntimeError('Unknown replay writer')
    replay_writer = Writer(replay_data_dir, worker_id=accelerator.local_process_index, n_workers=accelerator.num_processes)

    def run_multi_replay():
        """Train student model for multiple times."""
        nonlocal replay_writer
        replay_writer.prepare_reading()
        accelerator.wait_for_everyone()
        replay_reader = Reader(replay_writer)
        accelerator.wait_for_everyone()
        for t in range(replay_times):
            if accelerator.is_local_main_process:
                logger.info('Relay round {}: total count {}'.format(t, len(replay_writer)))
            run_replay(replay_reader)
            accelerator.wait_for_everyone()
        
        replay_reader.clear()
        replay_writer.clear()
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        replay_writer = Writer(replay_data_dir, worker_id=accelerator.local_process_index, n_workers=accelerator.num_processes)

    for batch in dataloader:
        if replay_writer.batch_num < args.replay_batches:
            if replay_writer.batch_num == 0:
                logger.info('Generating replay dataset ...')
            batch = get_distillation_data(args, accelerator=accelerator, vae=vae, 
                th_unet=th_unet, unet=unet, th_scheduler=th_scheduler, st_scheduler=st_scheduler, weight_dtype=weight_dtype, batch=batch,)
            replay_writer.add_batch(batch)
            if replay_writer.batch_num % 100 == 0:
                accelerator.wait_for_everyone()
        else:
            run_multi_replay()

    # There may be left batches
    if len(replay_writer) > 0:
        run_multi_replay()

    metrics = {
        m.name: m.get_weighted_value() for m in [m_loss, m_st_loss, m_ds_loss, m_th_loss, m_n_cond, m_cond_loss_sum]
    }
    n_samples = m_loss.w
    if not is_train and accelerator.is_main_process:
        metrics['eval_n_samples'] = n_samples
        accelerator.log(metrics, step=global_step)
        metrics.pop('eval_n_samples')

    return global_step, n_samples, metrics


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info('Args: {}'.format(args))
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        logger.info('set seed to {}'.format(args.seed))
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        logger.info("Using fp16")
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        accelerator.info("Using bf16")
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    model_cls = UNetWrapper
    assert os.path.exists(args.pretrained_model_name_or_path)
    th_unet = UNetWrapper.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet')
    unet = UNetWrapper.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet')
    assert unet.model is not th_unet.model, 'teacher and student use the same weights.'
    vae = EncoderDecoderWrapper.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    scheduler_cls = DDIMScheduler if args.run_mode != RunMode.STAGE_TWO else DistillDDIMScheduler
    th_scheduler = scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    st_scheduler = scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    st_scheduler.register_to_config(prediction_type='v_prediction')  #
    if args.run_mode == RunMode.STAGE_TWO:
        th_scheduler.set_timesteps(args.student_steps * 2, device=accelerator.device)
    st_scheduler.set_timesteps(args.student_steps, device=accelerator.device)
    assert st_scheduler.timesteps[0] == th_scheduler.timesteps[0]
    logger.info('th_scheduler {}'.format(th_scheduler))
    logger.info('th_scheduler.timesteps: {}'.format(th_scheduler.timesteps))
    logger.info('st_scheduler {}'.format(st_scheduler))
    logger.info('st_scheduler.timesteps: {}'.format(st_scheduler.timesteps))

    # Freeze vae, th_unet
    vae.eval()
    vae.requires_grad_(False)
    vae.train = disabled_train
    th_unet.eval()
    th_unet.requires_grad_(False)
    th_unet.train = disabled_train
    logger.info('vae {}'.format(type(vae)))
    logger.info('scheduler_cls {}'.format(scheduler_cls))

    # Create EMA for the unet.
    if args.use_ema:
        logger.info('create ema model.')
        ema_unet = EMAModel(unet.parameters(), model_cls=model_cls, model_config=args.unet_config, decay=args.ema_decay)
    else:
        ema_unet = None
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                # ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                ema_save_path = os.path.join(output_dir, "unet_ema.pt")
                logger.info('Save EMA to {}'.format(ema_save_path))
                torch.save(ema_unet.state_dict(), ema_save_path)

            for i, model in enumerate(models):
                if not isinstance(model, (AutoencoderKL, CLIPTextModel, EncoderDecoderWrapper)):
                    save_dir = os.path.join(output_dir, "unet")
                    model.save_pretrained(save_dir)
                    logger.info('save unet to {}'.format(save_dir))
                else:
                    logger.info('skip save {}'.format(model.__class__))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                ema_save_path = os.path.join(input_dir, 'unet_ema.pt')
                logger.info('Load EMA from {}'.format(ema_save_path))
                with open(ema_save_path, 'rb') as f:
                    state_dict = torch.load(f, map_location=accelerator.device)
                ema_unet.load_state_dict(state_dict)
                ema_unet.to(accelerator.device)

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, (AutoencoderKL, CLIPTextModel, EncoderDecoderWrapper)):
                    logger.info('skip load {}'.format(model.__class__))
                    continue
                # load diffusers style into model
                load_model = model_cls.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # # TODO: check gradient checkpoint
    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        logger.info('Enable tf32')
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    assert not args.use_8bit_adam
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_name == 'imagenet':
        from ldm.data.imagenet import ImageNetTrain, ImageNetValidation
        dataset = {
            'train': ImageNetTrain(),
            'test': ImageNetValidation()
        }
    else:
        raise RuntimeError('Unknown dataset_name {}'.format(args.dataset_name))

    logger.info('train examples: {}'.format(len(dataset['train'])))
    logger.info('eval exmpales: {}'.format(len(dataset['test'])))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = list(dataset["train"][0].keys())
    logger.info('column_names: {}'.format(column_names))

    with accelerator.main_process_first():
        assert not args.max_train_samples
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train)
        # train_dataset = dataset['train']
    with accelerator.main_process_first():
        from diffdstl.data.replay import DebugDataset
        if args.use_debug_examples is not None:
            for mode in ['train', 'test']:
                dataset[mode] = DebugDataset(dataset[mode], args.use_debug_examples, args.seed)

    # DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'],
        shuffle=True,
        collate_fn=None,
        batch_size=args.encoder_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset['test'], 
        shuffle=False, 
        collate_fn=None,
        batch_size=args.encoder_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # Donot prepare(th_unet) or it will be saved in hook
    named_params_before = dict(unet.named_parameters())
    vae, unet, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        vae, unet, optimizer, train_dataloader, eval_dataloader, lr_scheduler,
    )
    th_unet.to(accelerator.device)
    assert next(th_unet.parameters()).dtype == torch.float32, next(th_unet.parameters()).dtype

    named_params_after = dict(unet.named_parameters())
    params_diff = set(named_params_after.keys()) - set(named_params_before.keys())
    logger.info('before prepare, num_params {}, after parare, num params {}'.format(len(named_params_before), len(named_params_after)))
    assert len(named_params_after) == len(named_params_before), 'EMA would be wrong if the number of parameters are not the same after accelerator.prepare'
    logger.info('accelerator.prepare: unet.device={}, th_unet.device={}'.format(unet.device, th_unet.device))
    logger.info('type(unet)={}, type(th_unet)={}'.format(type(unet), type(th_unet)))

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop('unet_config', None)
        # logger.info('tracker_config: {}'.format(tracker_config))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            path = os.path.join(args.output_dir, path)
            accelerator.print(f"Resuming from: {path}")
            accelerator.load_state(path,)
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    best_metrics = None

    for epoch in range(first_epoch, args.num_train_epochs):
        # train for one epoch
        global_step, n_samples, metrics = run_epoch(args, vae=vae, 
            accelerator=accelerator, unet=unet, ema_unet=ema_unet, th_unet=th_unet, optimizer=optimizer, 
            lr_scheduler=lr_scheduler, th_scheduler=th_scheduler, st_scheduler=st_scheduler, 
            dataloader=train_dataloader, progress_bar=progress_bar, global_step=global_step, 
            is_train=True, weight_dtype=weight_dtype)
        logger.info('Train Epoch {}, n_samples: {}, metrics: {}'.format(epoch, n_samples, metrics))

        if epoch % args.validation_epochs == 0:
            images, best_metrics = run_validation(
                args, vae=vae, accelerator=accelerator, unet=unet, 
                ema_unet=ema_unet, th_unet=th_unet, optimizer=optimizer, lr_scheduler=lr_scheduler,
                th_scheduler=th_scheduler, st_scheduler=st_scheduler, dataloader=eval_dataloader, 
                progress_bar=progress_bar, global_step=global_step, epoch=epoch, best_metrics=best_metrics, 
                loss_key=args.loss_key, weight_dtype=weight_dtype,
            )

        if global_step >= args.max_train_steps:
            logger.info('global_step {} >= max_train_steps {}'.format(global_step, args.max_train_steps))
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    if accelerator.is_local_main_process:
        logger.info('End traning')


if __name__ == "__main__":
    main()

