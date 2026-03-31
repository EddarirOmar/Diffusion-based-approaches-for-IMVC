"""
Noise schedules for diffusion models.

Implements linear and cosine variance schedules for the forward diffusion process.
"""

import torch
import numpy as np
from typing import Literal


class NoiseScheduler:
    """
    Manages the noise schedule for diffusion models.
    
    This implements the forward process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    
    Args:
        num_timesteps: Number of diffusion steps (typically 1000)
        schedule_type: 'linear' or 'cosine'
        beta_start: Starting noise level (default 0.0001 for linear)
        beta_end: Ending noise level (default 0.02 for linear)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: Literal['linear', 'cosine'] = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        # Compute schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
        # Precompute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        
        # Posterior variance
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.clamp(
            torch.log(posterior_variance), min=-20.0
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_schedule(self, num_timesteps: int) -> torch.Tensor:
        """
        Cosine schedule as per Improved DDPM paper.
        """
        s = 0.008
        steps = torch.arange(num_timesteps + 1)
        alphas_cumprod = torch.cos(
            ((steps / num_timesteps) + s) / (1 + s) * np.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def to(self, device: torch.device):
        """Move schedule to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, 
                  noise: torch.Tensor) -> torch.Tensor:
        """
        Add noise to data at timestep t.
        
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        
        Args:
            x_0: Original data, shape (batch_size, ...)
            t: Timestep indices, shape (batch_size,)
            noise: Gaussian noise, same shape as x_0
        
        Returns:
            Noisy data x_t
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
    
    def denoise_step(self, x_t: torch.Tensor, predicted_noise: torch.Tensor, 
                     t: torch.Tensor) -> torch.Tensor:
        """
        One step of reverse diffusion (Langevin dynamics).
        
        Args:
            x_t: Noisy data at timestep t
            predicted_noise: Model's noise prediction
            t: Timestep indices
        
        Returns:
            x_{t-1}
        """
        # Posterior mean and variance
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
        
        mean = posterior_mean_coef1_t * predicted_noise + posterior_mean_coef2_t * x_t
        
        # Add noise if not at the end
        noise = torch.randn_like(x_t)
        x_t_minus_1 = mean + torch.sqrt(posterior_var_t) * noise
        
        return x_t_minus_1
    
    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps and reshape to match x_shape.
        
        Args:
            a: 1D tensor of shape (num_timesteps,)
            t: Tensor of timestep indices
            x_shape: Target shape for broadcasting
        
        Returns:
            Extracted and reshaped tensor
        """
        batch_size = x_shape[0]
        out = a.gather(0, t)
        
        # Reshape to match x_shape: (batch_size, 1, 1, ..., 1)
        out = out.reshape(batch_size, *([1] * (len(x_shape) - 1)))
        return out
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
