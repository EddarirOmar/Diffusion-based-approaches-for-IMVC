"""Stage 1: Per-view autoencoder reconstruction (DCG-style config, no YAML files)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from datasets import get_dataloader
from configure import build_stage1_config, dataset_slug
from loss import batch_reconstruction_loss
from baseModels import MultiViewAutoEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_rec(
    model: MultiViewAutoEncoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    view_dims: list[int],
) -> dict[str, float]:
    model.eval()
    losses = []

    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["view_mask"].to(device)
        loss, _ = batch_reconstruction_loss(model, views, mask, view_dims)
        losses.append(float(loss.detach().cpu()))

    return {"val_rec_loss": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def export_latents(
    model: MultiViewAutoEncoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    view_dims: list[int],
) -> dict[str, np.ndarray]:
    model.eval()
    z_all = []
    mask_all = []
    label_all = []

    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["view_mask"].cpu().numpy()
        n_views = views.shape[1]

        z_views = []
        for v in range(n_views):
            x_v = views[:, v, : view_dims[v]]
            z_v = model.encode_view(x_v, v)
            z_views.append(z_v.cpu().numpy())

        z_views = np.stack(z_views, axis=1)  # [B, V, latent_dim]
        z_all.append(z_views)
        mask_all.append(mask)

        if "label" in batch:
            label_all.append(batch["label"].cpu().numpy())

    out = {
        "latents": np.concatenate(z_all, axis=0),
        "view_mask": np.concatenate(mask_all, axis=0),
    }
    if label_all:
        out["labels"] = np.concatenate(label_all, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Per-view autoencoder reconstruction")
    parser.add_argument("--data_name", required=True, help="Dataset name (e.g., NoisyMNIST, CUB, Synthetic3d)")
    parser.add_argument("--data_path", default=None, help="Override default dataset path")
    parser.add_argument("--output_dir", default=None, help="Override default output directory")
    args = parser.parse_args()

    cfg = build_stage1_config(args.data_name, data_path=args.data_path)

    set_seed(int(cfg["seed"]))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("GPU: " + str(use_cuda))
    print("Data set: " + str(cfg["dataset"]["name"]))
    print("--------------------Missing rate = " + str(cfg["dataset"]["missing_rate"]) + "--------------------")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"outputs/stage1_rec_{dataset_slug(args.data_name)}")
    
    ckpt_dir = out_dir / "checkpoints"
    latent_dir = out_dir / "latents"
    logs_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_loader, test_loader = get_dataloader(
        dataset_name=cfg["dataset"]["name"],
        data_path=cfg["dataset"]["path"],
        batch_size=int(cfg["reconstruction"]["batch_size"]),
        missing_rate=float(cfg["dataset"]["missing_rate"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=int(cfg["seed"]),
    )

    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    view_dims = list(base_ds.view_dims)
    print("View dimensions: " + str(view_dims))

    model = MultiViewAutoEncoder(
        view_dims=view_dims,
        latent_dim=int(cfg["reconstruction"]["latent_dim"]),
        hidden_dim=int(cfg["reconstruction"]["hidden_dim"]),
        dropout=float(cfg["reconstruction"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["reconstruction"]["learning_rate"]),
        weight_decay=float(cfg["reconstruction"]["weight_decay"]),
    )

    epochs = int(cfg["reconstruction"]["epochs"])
    save_every = int(cfg["reconstruction"]["save_every"])

    best_val = float("inf")
    train_history = []

    print("Start training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            views = batch["views"].to(device)
            mask = batch["view_mask"].to(device)

            loss, stats = batch_reconstruction_loss(model, views, mask, view_dims)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_f = float(loss.detach().cpu())
            epoch_losses.append(loss_f)

        val = evaluate_rec(model, test_loader, device, view_dims)
        mean_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = val["val_rec_loss"]

        rec = {
            "epoch": epoch,
            "train_rec_loss": mean_train,
            "val_rec_loss": val_loss,
        }
        train_history.append(rec)
        print("Epoch: {:.0f}/{:.0f} ==> loss = {:.4f} | train_rec={:.4f} val_rec={:.4f}".format(
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
                    "view_dims": view_dims,
                    "latent_dim": int(cfg["reconstruction"]["latent_dim"]),
                    "best_val_rec_loss": best_val,
                },
                ckpt_dir / "best_stage1_rec.pt",
            )
        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "view_dims": view_dims,
                    "latent_dim": int(cfg["reconstruction"]["latent_dim"]),
                },
                ckpt_dir / f"stage1_rec_epoch_{epoch:04d}.pt",
            )

    with open(logs_dir / "stage1_rec_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)

    # Export latent codes for Stage 2.
    print("Exporting latent codes...")
    train_latents = export_latents(model, train_loader, device, view_dims)
    test_latents = export_latents(model, test_loader, device, view_dims)

    np.savez_compressed(latent_dir / "train_latents.npz", **train_latents)
    np.savez_compressed(latent_dir / "test_latents.npz", **test_latents)

    print("-------------------The training over Missing rate = " + str(cfg["dataset"]["missing_rate"]) + "--------------------")
    print("[DONE] Stage 1 complete. Latents saved in " + str(latent_dir))


if __name__ == "__main__":
    main()
