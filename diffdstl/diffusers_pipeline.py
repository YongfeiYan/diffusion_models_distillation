import os
import yaml
import torch
import inspect
import argparse
import numpy as np 
from torch import nn 
from tqdm import tqdm
from typing import Optional, Union, List
from omegaconf import OmegaConf
from diffusers import LDMPipeline, DiffusionPipeline, DDIMScheduler

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler as LDMDDIMSampler
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL


class DDIMSchedulerOutput:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


class UNetWrapperOutput:
    def __init__(self, sample):
        self.sample = sample


def load_model_from_config(config, state_dict):
    print('instantiate:', config['target'])
    model = instantiate_from_config(config)
    if not state_dict:
        return model
    print('load state dict')
    model.load_state_dict(state_dict, strict=False)
    return model


CACHE_MODELS = {}  # cache loaded_model from ckpt_path to prevent consuming too much memory 


def load_model(config_path, ckpt_path=None):
    """
    Cache load_model when ckpt_path is not None, so that load_model(config_path) can create muliple models.
    """
    key = (config_path, ckpt_path)
    if key in CACHE_MODELS:
        print('load_model found cache:', key)
        return CACHE_MODELS[key]
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = config['model']
    print('model_name:', model['target'])
    if ckpt_path:
        print(f"Loading model from {ckpt_path}")
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    model = load_model_from_config(model,
                                   pl_sd["state_dict"])
    print('global_step', global_step)
    
    if ckpt_path:
        CACHE_MODELS[key] = model
    
    return model


class EncoderDecoderWrapper(ModelMixin, ConfigMixin):
    """Wrap first_stage_model in checkpoints of third_party/stable-diffusion/ldm."""
    @register_to_config
    def __init__(self, ldm_config_path='', ldm_ckpt_path=None):
        super().__init__()
        model = load_model(ldm_config_path, ldm_ckpt_path)
        self.first_stage_model = model.first_stage_model
        self.scale_factor = model.scale_factor
        assert self.scale_factor == 1

    @torch.no_grad()
    def encode(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode(self, z):
        z = 1. / self.scale_factor * z
        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=False)
        else:
            return self.first_stage_model.decode(z)


class UNetWrapper(ModelMixin, ConfigMixin):
    """Wrap unet in checkpoints of third_party/stable-diffusion/ldm."""    
    @register_to_config
    def __init__(self, in_channels, sample_size, ldm_config_path, ldm_ckpt_path=None):
        super().__init__()
        model = load_model(ldm_config_path, ldm_ckpt_path)
        self.model = model.model  # DiffusionWrapper
        self.cond_stage_model = model.cond_stage_model
        self.cond_stage_key = model.cond_stage_key
        self.cond_stage_forward = model.cond_stage_forward

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def forward(self, x, timestep, cond=None):
        cond = {self.cond_stage_key: cond}
        cond = self.get_learned_conditioning(cond)
        if not isinstance(cond, list):
            cond = [cond]
        key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
        cond = {key: cond}
        sample = self.model(x, timestep, **cond)
        return UNetWrapperOutput(sample)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """Copied from diffusers.
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class LDMCodeWrapperPipeline(DiffusionPipeline):
    """Warp a model in third_party/stable-diffusion to diffusers pipeline."""
    def __init__(self, unet, vae, scheduler):
        super().__init__()
        self.register_modules(unet=unet, vae=vae, scheduler=scheduler)
    
    @property
    def device(self):
        return next(self.unet.parameters()).device
    
    @staticmethod
    def build(config_path, model_or_ckpt_path=None):
        """Create an instance from a saved checkpint of third_party/stable-diffusion."""
        with open(config_path) as f:
            print('load config from', config_path)
            config = yaml.safe_load(f)
        if model_or_ckpt_path is not None and isinstance(model_or_ckpt_path, str):
            model = load_model(config_path, model_or_ckpt_path)
        else:
            model = load_model(config_path)
        params = config['model']['params']
        print('params', params)

        # build scheduler
        params['beta_schedule'] = params.get('beta_schedule', 'linear')
        assert params['beta_schedule'] == 'linear', 'Only linear is considerated now, but found {}'.format(params['beta_schedule'])
        params['beta_schedule'] = 'scaled_linear'  # linear in LDM in equivalent to scaled_linear in diffusers
        scheduler = DDIMScheduler(num_train_timesteps=params['timesteps'], beta_start=params['linear_start'], beta_end=params['linear_end'], beta_schedule=params['beta_schedule'], set_alpha_to_one=False, steps_offset=1, clip_sample=False)
        assert (model.betas - scheduler.betas).abs().sum().item() < 1e-5, (model.betas[::20], scheduler.betas[::20], (model.betas - scheduler.betas).abs().sum().item())
        
        # build uent and vae 
        unet = UNetWrapper(model.channels, model.image_size, config_path, model_or_ckpt_path)
        vae = EncoderDecoderWrapper(config_path, model_or_ckpt_path)

        return LDMCodeWrapperPipeline(unet, vae, scheduler)

    @torch.no_grad()
    def __call__(self,
        batch_size: int = 1,
        cond = None,
        uncond = None, 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        guidance_scale = 7.5,  # set to 1, mean no guidance
        guidance_rescale = 0.0,  # same as StableDiffusionPipeline
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil", 
    ):
        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator, device=self.device,
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # 
        do_classifier_free_guidance = cond is not None and guidance_scale > 1.0
        if do_classifier_free_guidance:
            assert uncond is not None, 'uncond should not be None when guidance_scale is {}'.format(guidance_rescale)
            assert cond.shape == uncond.shape, 'cond.shape {} != uncond.shape {}'.format(cond.shape, uncond.shape)
            assert cond.shape[0] == batch_size, 'conda.shape[0] != batch_size'
            cond = torch.cat([uncond, cond], dim=0).to(self.device)
        
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            # 
            timestep = torch.full((batch_size,), t, device=latents.device, dtype=torch.int64)
            latent_model_input = latents
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents, latents], dim=0)
                timestep = torch.cat([timestep, timestep], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            model_pred = self.unet(latent_model_input, timestep, cond=cond).sample
            
            if do_classifier_free_guidance:
                pred_uncond, pred_cond = model_pred.chunk(2)
                model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                model_pred = rescale_noise_cfg(model_pred, pred_cond, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(model_pred, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VAE
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return ImagePipelineOutput(images=image)


if __name__ == '__main__':
    pass
