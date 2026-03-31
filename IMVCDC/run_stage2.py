"""Stage 2: Latent space conditional diffusion for missing view completion."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baseModels import LatentConditionalDenoiser
from baseModels import NoiseScheduler
from configure import build_stage2_config, dataset_slug
from loss import diffusion_noise_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latent_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path)
    z = data["latents"]
    m = data["view_mask"]
    y = data["labels"] if "labels" in data.files else None
    return z, m, y


def split_target_cond(z: torch.Tensor, target_view: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split z into target view and conditional context."""
    x_target = z[:, target_view, :]
    cond_parts = []
    for v in range(z.shape[1]):
        if v == target_view:
            continue
        cond_parts.append(z[:, v, :])
    cond = torch.stack(cond_parts, dim=1) if cond_parts else torch.empty(z.shape[0], 0, z.shape[2], device=z.device)
    return x_target, cond


def select_stage2_training_samples(z: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select training samples (prefer complete samples per paper, fallback to all)."""
    keep = np.all(m > 0.5, axis=1)
    if int(np.sum(keep)) > 0:
        return z[keep], m[keep]
    return z, m


@torch.no_grad()
def eval_loss(
    model: LatentConditionalDenoiser,
    scheduler: NoiseScheduler,
    loader: DataLoader,
    device: torch.device,
    target_view: int,
) -> float:
    model.eval()
    vals = []
    for (z_batch,) in loader:
        z_batch = z_batch.to(device)
        x0, cond = split_target_cond(z_batch, target_view)
        t = scheduler.sample_timesteps(x0.shape[0], device)
        eps = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, t, eps)
        eps_pred = model(x_t, t, cond)
        loss = diffusion_noise_loss(eps_pred, eps)
        vals.append(float(loss.detach().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Latent diffusion for missing view completion")
    parser.add_argument("--data_name", required=True, help="Dataset name (e.g., NoisyMNIST, CUB)")
    parser.add_argument("--output_dir", default=None, help="Override default output directory")
    args = parser.parse_args()

    cfg = build_stage2_config(args.data_name)

    set_seed(int(cfg["seed"]))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("GPU: " + str(use_cuda))
    print("Data set: " + str(args.data_name))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"outputs/stage2_dm_{dataset_slug(args.data_name)}")
    
    ckpt_dir = out_dir / "checkpoints"
    latent_dir = out_dir / "latents"
    logs_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 1 latents...")
    z_train_np, m_train_np, _ = load_latent_npz(cfg["stage1"]["train_latents"])
    z_test_np, m_test_np, _ = load_latent_npz(cfg["stage1"]["test_latents"])

    z_train_np, m_train_np = select_stage2_training_samples(z_train_np, m_train_np)

    z_train = torch.from_numpy(z_train_np).float()
    z_test = torch.from_numpy(z_test_np).float()

    train_loader = DataLoader(
        TensorDataset(z_train),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(z_test),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    latent_dim = int(z_train.shape[-1])
    target_view = int(cfg["training"]["target_view"])

    model = LatentConditionalDenoiser(
        latent_dim=latent_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        time_dim=int(cfg["model"]["time_dim"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    scheduler = NoiseScheduler(
        num_timesteps=int(cfg["diffusion"]["timesteps"]),
        schedule_type=str(cfg["diffusion"]["schedule"]),
    )
    scheduler.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"]["save_every"])

    best_val = float("inf")
    history = []
    print("Start training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for (z_batch,) in train_loader:
            z_batch = z_batch.to(device)
            x0, cond = split_target_cond(z_batch, target_view)

            t = scheduler.sample_timesteps(x0.shape[0], device)
            eps = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, t, eps)

            eps_pred = model(x_t, t, cond)
            loss = diffusion_noise_loss(eps_pred, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            optimizer.step()

            loss_f = float(loss.detach().cpu())
            train_losses.append(loss_f)

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = eval_loss(model, scheduler, test_loader, device, target_view)

        history.append({"epoch": epoch, "train_dm_loss": mean_train, "val_dm_loss": val_loss})
        print("Epoch: {:.0f}/{:.0f} ==> loss = {:.4f} | train_dm={:.4f} val_dm={:.4f}".format(
            epoch,
            epochs,
            mean_train,
            mean_train,
            val_loss,
        ))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": latent_dim,
                    "target_view": target_view,
                    "best_val_dm_loss": best_val,
                },
                ckpt_dir / "best_stage2_dm.pt",
            )
        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": latent_dim,
                    "target_view": target_view,
                },
                ckpt_dir / f"stage2_dm_epoch_{epoch:04d}.pt",
            )

    with open(logs_dir / "stage2_dm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-------------------The training over data set = " + str(args.data_name) + "--------------------")
    print("[DONE] Stage 2 complete.")


if __name__ == "__main__":
    main()
