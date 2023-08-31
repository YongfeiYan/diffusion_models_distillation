
import numpy as np
from typing import List, Optional, Union, Tuple
from diffusers import DDIMScheduler
from diffusers.configuration_utils import register_to_config, ConfigMixin
import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class DistillDDIMScheduler(DDIMScheduler, SchedulerMixin, ConfigMixin):
    """Making sampling occurs only on a subsequence of DDIMSchedler.timesteps.
    DistillDDIMScheduler.timesteps is a subsequence of super().set_timesteps(teacher_max_train_steps).
    """
    @register_to_config
    def __init__(
        self,
        teacher_max_train_steps: int = 64,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas, clip_sample, set_alpha_to_one, steps_offset, prediction_type, thresholding, dynamic_thresholding_ratio, clip_sample_range, sample_max_value, timestep_spacing, rescale_betas_zero_snr)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Set timesteps as a subsequence of teacher scheduler timesteps."""
        assert self.config.teacher_max_train_steps % num_inference_steps == 0, 'num_inference_steps {} is not a factor of teacher_max_train_steps {}'.format(num_inference_steps, self.config.teacher_max_train_steps)
        super().set_timesteps(num_inference_steps=self.config.teacher_max_train_steps)
        raw_timesteps = self.timesteps.cpu().numpy()
        factor = self.config.teacher_max_train_steps // num_inference_steps
        timesteps = np.array([raw_timesteps[i * factor] for i in range(num_inference_steps)], dtype=np.int64)  # from large to small
        self.num_inference_steps = len(timesteps)

        self.timesteps = torch.from_numpy(timesteps).to(device)
        if device is not None:
            self.alphas_cumprod = self.alphas_cumprod.to(device)

    def get_alpha_cumprods(self, timestep: torch.Tensor, broadcast_to_shape=None):
        """
            alphas_cumprod[timestep], alphas_cumprod[prev_timesptes]
        """
        assert len(timestep.shape) == 1, '1-dimensional timestep is required, but found {}'.format(timestep.shape)
        if self.timesteps.device != timestep.device:
            self.timesteps = self.timesteps.to(timestep.device)  # timesteps are in the order from large to small !
            self.alphas_cumprod = self.alphas_cumprod.to(timestep.device)
                
        bsz = timestep.shape[0]
        mask = ((self.timesteps.repeat(bsz, 1) - timestep.unsqueeze(1)) == 0).to(torch.int64)  # bsz x T
        assert torch.all(mask.sum(dim=1)), 'Found unseen timestep {}, {}'.format(timestep, self.timesteps)
        index = (torch.arange(len(self.timesteps), device=timestep.device).repeat(bsz, 1) * mask).sum(dim=1)
        
        prev_index = index + 1
        prev_cond = prev_index < len(self.timesteps)
        prev_index = torch.where(prev_cond, prev_index, 0)
        prev_timestep = torch.where(prev_cond, self.timesteps[prev_index], 0)

        alpha_prod_t = self.alphas_cumprod[timestep]  # bsz
        alpha_prod_t_prev = torch.where(prev_cond, self.alphas_cumprod[prev_timestep], self.final_alpha_cumprod)
        
        if broadcast_to_shape is not None:
            shape = [bsz] + [1] * len(broadcast_to_shape[1:])
            alpha_prod_t = alpha_prod_t.reshape(*shape)
            alpha_prod_t_prev = alpha_prod_t_prev.reshape(*shape)

        return alpha_prod_t, alpha_prod_t_prev
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        assert eta == 0.0, eta
        assert self.config.prediction_type == 'v_prediction', self.config.prediction_type
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        bsz = model_output.shape[0]
        if isinstance(timestep, int) or len(timestep.shape) == 0:
            timestep = torch.full((bsz,), fill_value=timestep, device=model_output.device, dtype=torch.int64)

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        # 2. compute alphas, betas
        # alpha_prod_t = self.alphas_cumprod[timestep]  # bsz
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t, alpha_prod_t_prev = self.get_alpha_cumprods(timestep, model_output.shape)

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

        # 4. Clip or threshold "predicted x_0"
        assert not self.config.thresholding
        assert not self.config.clip_sample

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self._get_variance(timestep, prev_timestep)
        # std_dev_t = eta * variance ** (0.5)
        std_dev_t = 0.0

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
