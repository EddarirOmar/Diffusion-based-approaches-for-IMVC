"""IMVCDC base models: Stage 1 (AE), Stage 2 (Diffusion), Stage 3 (Clustering)."""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# STAGE 1: AUTOENCODERS
# ============================================================================

class ViewAutoEncoder(nn.Module):
    """Simple MLP autoencoder for one view."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class MultiViewAutoEncoder(nn.Module):
    """Container for per-view autoencoders."""

    def __init__(
        self,
        view_dims: list[int],
        latent_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.view_dims = view_dims
        self.latent_dim = latent_dim

        self.view_aes = nn.ModuleList(
            [
                ViewAutoEncoder(
                    input_dim=d,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for d in view_dims
            ]
        )

    def forward_view(self, x: torch.Tensor, view_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.view_aes[view_idx](x)

    def encode_view(self, x: torch.Tensor, view_idx: int) -> torch.Tensor:
        return self.view_aes[view_idx].encode(x)

    def decode_view(self, z: torch.Tensor, view_idx: int) -> torch.Tensor:
        return self.view_aes[view_idx].decode(z)


# ============================================================================
# STAGE 2: DIFFUSION
# ============================================================================

class NoiseScheduler:
    """Manages noise schedule for diffusion models."""

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
        """Cosine schedule as per Improved DDPM paper."""
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

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add noise to data at timestep t."""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract coefficients at specified timesteps."""
        batch_size = t.shape[0]
        t_index = t.to(device=a.device, dtype=torch.long)
        out = a.gather(0, t_index)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1))).to(a.device)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class LatentConditionalDenoiser(nn.Module):
    """Latent-space diffusion denoiser with cross-attention conditioning."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.cond_proj = nn.Linear(latent_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = x_t.shape[0]

        x_token = self.x_proj(x_t).unsqueeze(1)
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(1)
        x_token = x_token + t_emb

        if cond_latents is not None and cond_latents.numel() > 0:
            cond_tokens = self.cond_proj(cond_latents)
            attn_out, _ = self.cross_attn(x_token, cond_tokens, cond_tokens, need_weights=False)
            h = x_token + attn_out
        else:
            h = x_token

        h = h + self.ffn(h)
        eps = self.out(h).squeeze(1)
        return eps


# ============================================================================
# STAGE 3: CONTRASTIVE CLUSTERING
# ============================================================================

class ContrastiveClusteringHead(nn.Module):
    """View-specific MLPs + view-shared projection + clustering classifier."""

    def __init__(
        self,
        n_views: int,
        latent_dim: int,
        hidden_dim: int,
        proj_dim: int,
        n_clusters: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_clusters = n_clusters

        self.view_specific = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_views)
            ]
        )

        self.view_shared = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.classifier = nn.Linear(proj_dim, n_clusters)

    def forward_view(self, z_v: torch.Tensor, view_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.view_specific[view_idx](z_v)
        h = self.view_shared(s)
        logits = self.classifier(h)
        return h, logits

    def forward_all(self, z: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hs = []
        ys = []
        for v in range(self.n_views):
            h, y = self.forward_view(z[:, v, :], v)
            hs.append(h)
            ys.append(y)
        return hs, ys


__all__ = [
    # Stage 1
    "ViewAutoEncoder",
    "MultiViewAutoEncoder",
    # Stage 2
    "NoiseScheduler",
    "SinusoidalTimeEmbedding",
    "LatentConditionalDenoiser",
    # Stage 3
    "ContrastiveClusteringHead",
]
