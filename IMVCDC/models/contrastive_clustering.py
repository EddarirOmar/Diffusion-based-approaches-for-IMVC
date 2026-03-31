"""Stage 3 contrastive clustering heads (paper-aligned approximation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from loss import loss_lc as _loss_lc
from loss import loss_lh as _loss_lh


class ContrastiveClusteringHead(nn.Module):
    """View-specific MLPs + view-shared MLP + clustering classifier."""

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
        probs = torch.softmax(logits, dim=1)
        return h, probs

    def forward_all(self, z: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hs = []
        ys = []
        for v in range(self.n_views):
            h, y = self.forward_view(z[:, v, :], v)
            hs.append(h)
            ys.append(y)
        return hs, ys


def loss_lh(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper for unified loss module."""
    return _loss_lh(h1, h2)


def loss_lc(y1: torch.Tensor, y2: torch.Tensor, entropy_weight: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Compatibility wrapper for unified loss module."""
    return _loss_lc(y1, y2, entropy_weight=entropy_weight, eps=eps)
