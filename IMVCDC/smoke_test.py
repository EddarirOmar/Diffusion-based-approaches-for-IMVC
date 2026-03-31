"""Smoke test for IMVCDC: quick validation of pipeline components."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from baseModels import MultiViewAutoEncoder, LatentConditionalDenoiser, ContrastiveClusteringHead, NoiseScheduler
from configure import get_default_config, dataset_slug
from loss import batch_reconstruction_loss, clustering_loss, diffusion_noise_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_test_stage1() -> None:
    """Test Stage 1: per-view autoencoders."""
    print("\n=== STAGE 1 SMOKE TEST ===")
    
    cfg = get_default_config("Synthetic3d")
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cpu")
    
    # Create minimal batch
    view_dims = [3, 3]
    batch_size = 4
    
    views = torch.randn(batch_size, 2, 3)
    view_mask = torch.ones(batch_size, 2)
    
    # Instantiate model
    model = MultiViewAutoEncoder(
        view_dims=view_dims,
        latent_dim=128,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)
    
    # Forward pass
    views = views.to(device)
    view_mask = view_mask.to(device)
    loss, stats = batch_reconstruction_loss(model, views, view_mask, view_dims)
    
    print(f"Stage 1 - Reconstruction loss: {loss.item():.6f}")
    view0_rec = stats.get("view_0_rec")
    if isinstance(view0_rec, (int, float)):
        print(f"  View 0 rec loss: {float(view0_rec):.6f}")
    else:
        print("  View 0 rec loss: N/A")
    print("  STAGE 1: PASS")


def smoke_test_stage2() -> None:
    """Test Stage 2: latent diffusion."""
    print("\n=== STAGE 2 SMOKE TEST ===")
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Create minimal batch of latents
    batch_size = 4
    latent_dim = 128
    n_views = 2
    
    z = torch.randn(batch_size, n_views, latent_dim)
    
    # Instantiate model
    model = LatentConditionalDenoiser(
        latent_dim=latent_dim,
        hidden_dim=256,
        time_dim=128,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    
    scheduler = NoiseScheduler(
        num_timesteps=100,
        schedule_type="linear",
    )
    scheduler.to(device)
    
    # Forward pass
    target_view = 0
    x_target = z[:, target_view, :]
    cond_parts = [z[:, v, :] for v in range(n_views) if v != target_view]
    cond = torch.stack(cond_parts, dim=1) if cond_parts else torch.empty(batch_size, 0, latent_dim, device=device)
    
    t = scheduler.sample_timesteps(batch_size, device)
    eps = torch.randn_like(x_target)
    x_t = scheduler.add_noise(x_target, t, eps)
    
    eps_pred = model(x_t, t, cond)
    loss = diffusion_noise_loss(eps_pred, eps)
    
    print(f"Stage 2 - Diffusion loss: {loss.item():.6f}")
    print("  STAGE 2: PASS")


def smoke_test_stage3() -> None:
    """Test Stage 3: contrastive clustering."""
    print("\n=== STAGE 3 SMOKE TEST ===")
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Create minimal batch of completed latents
    batch_size = 8
    n_views = 2
    latent_dim = 128
    n_clusters = 3
    
    z = torch.randn(batch_size, n_views, latent_dim)
    
    # Instantiate model
    model = ContrastiveClusteringHead(
        n_views=n_views,
        latent_dim=latent_dim,
        hidden_dim=256,
        proj_dim=128,
        n_clusters=n_clusters,
        dropout=0.1,
    ).to(device)
    
    # Forward pass
    z = z.to(device)
    hs, ys = model.forward_all(z)
    
    if len(hs) >= 2 and len(ys) >= 2:
        loss, ce_loss, ent_loss = clustering_loss(ys[0], ys[1], hs[0], hs[1], entropy_weight=0.5)
    else:
        loss = torch.tensor(0.0, device=device)
    
    print(f"Stage 3 - Clustering loss: {loss.item():.6f}")
    print("  STAGE 3: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="IMVCDC smoke test")
    parser.add_argument("--stage", default="all", choices=["all", "1", "2", "3"])
    args = parser.parse_args()
    
    print("=" * 50)
    print("IMVCDC SMOKE TEST")
    print("=" * 50)
    
    try:
        if args.stage in ["all", "1"]:
            smoke_test_stage1()
        if args.stage in ["all", "2"]:
            smoke_test_stage2()
        if args.stage in ["all", "3"]:
            smoke_test_stage3()
        
        print("\n" + "=" * 50)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
