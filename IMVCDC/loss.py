"""Unified loss functions for IMVCDC stages.

This module centralizes:
- Stage 1: reconstruction losses
- Stage 2: diffusion denoising loss
- Stage 3: contrastive clustering losses
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked reconstruction loss used in Stage 1.

    Args:
        x: Ground-truth tensor [B, D].
        x_hat: Reconstructed tensor [B, D].
        mask: Availability mask [B, 1] where 1 means observed.
    """
    sq_err = ((x - x_hat) ** 2) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return sq_err.sum() / denom


def batch_reconstruction_loss(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    view_dims: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Stage 1 total reconstruction loss across all views.

    Returns:
        total_loss and per-view diagnostics.
    """
    n_views = views.shape[1]
    total = torch.tensor(0.0, device=views.device)
    per_view: dict[str, float] = {}

    for v in range(n_views):
        dim_v = view_dims[v]
        x_v = views[:, v, :dim_v]
        z_v, x_hat_v = model.forward_view(x_v, v)

        m_v = view_mask[:, v : v + 1]
        loss_v = masked_reconstruction_loss(x_v, x_hat_v, m_v)

        total = total + loss_v
        per_view[f"view_{v}_rec"] = float(loss_v.detach().cpu())
        per_view[f"view_{v}_z_norm"] = float(z_v.detach().norm(dim=1).mean().cpu())

    return total, per_view


def diffusion_noise_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """Stage 2 denoising objective L_dm."""
    return F.mse_loss(eps_pred, eps_true)


def loss_lh(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Spectral contrastive loss L_H (two-view approximation)."""
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    n = h1.shape[0]

    pos = -2.0 / n * torch.sum(torch.sum(h1 * h2, dim=1))

    sim = torch.matmul(h1, h2.t())
    eye = torch.eye(n, device=h1.device, dtype=torch.bool)
    neg = torch.sum(sim[~eye] ** 2) / max(n * (n - 1), 1)

    return pos + neg


def loss_lc(y1: torch.Tensor, y2: torch.Tensor, entropy_weight: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Category contrastive loss L_C (two-view approximation)."""
    c1 = y1.t()
    c2 = y2.t()
    k = c1.shape[0]

    def sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return torch.dot(a, b)

    terms = []
    for i in range(k):
        num = torch.exp(sim(c1[i], c2[i]))

        den1_terms = [torch.exp(sim(c1[i], c1[j])) for j in range(k) if j != i]
        den2_terms = [torch.exp(sim(c2[i], c2[j])) for j in range(k) if j != i]

        den1 = torch.sum(torch.stack(den1_terms)) + eps if den1_terms else torch.tensor(1.0, device=y1.device)
        den2 = torch.sum(torch.stack(den2_terms)) + eps if den2_terms else torch.tensor(1.0, device=y1.device)

        terms.append(-torch.log(num / den1 + eps))
        terms.append(-torch.log(num / den2 + eps))

    contrast = torch.mean(torch.stack(terms)) if terms else torch.tensor(0.0, device=y1.device)

    p1 = torch.clamp(y1.mean(dim=0), min=eps)
    p2 = torch.clamp(y2.mean(dim=0), min=eps)
    entropy_reg = torch.sum(p1 * torch.log(p1)) + torch.sum(p2 * torch.log(p2))

    return contrast + entropy_weight * entropy_reg


def clustering_loss(
    y1: torch.Tensor,
    y2: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
    entropy_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 3 objective L_clu = L_H + L_C.

    Returns:
        total, lh, lc
    """
    lh = loss_lh(h1, h2)
    lc = loss_lc(y1, y2, entropy_weight=entropy_weight)
    return lh + lc, lh, lc
