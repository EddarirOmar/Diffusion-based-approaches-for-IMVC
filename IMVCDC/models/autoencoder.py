"""Per-view autoencoder modules for Stage 1 reconstruction (paper-aligned)."""

from __future__ import annotations

import torch
import torch.nn as nn


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
