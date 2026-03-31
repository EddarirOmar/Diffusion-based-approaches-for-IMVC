"""Latent-space diffusion denoiser with attention conditioning (Stage 2)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
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
    """Predict epsilon in latent space using cross-attention on condition latents.

    Inputs:
    - x_t: [B, D]
    - t: [B]
    - cond_latents: [B, C, D] where C = number of conditioning views
    """

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

        x_token = self.x_proj(x_t).unsqueeze(1)  # [B,1,H]
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(1)  # [B,1,H]
        x_token = x_token + t_emb

        if cond_latents is not None and cond_latents.numel() > 0:
            cond_tokens = self.cond_proj(cond_latents)  # [B,C,H]
            attn_out, _ = self.cross_attn(x_token, cond_tokens, cond_tokens, need_weights=False)
            h = x_token + attn_out
        else:
            h = x_token

        h = h + self.ffn(h)
        eps = self.out(h).squeeze(1)
        return eps
