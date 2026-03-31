"""Stage 3: Contrastive clustering on completed latents."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baseModels import ContrastiveClusteringHead
from configure import build_stage3_config, dataset_slug
from loss import clustering_loss
from metrics import clustering_metrics_from_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_fusion(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """Fused prediction: argmax(y1 + y2)."""
    return torch.argmax(y1 + y2, dim=1)


@torch.no_grad()
def eval_epoch(
    model: ContrastiveClusteringHead,
    loader: DataLoader,
    device: torch.device,
    entropy_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    all_pred = []
    all_y = []

    for batch in loader:
        if len(batch) == 2:
            z, y = batch
            y = y.cpu().numpy()
        else:
            z = batch[0]
            y = None

        z = z.to(device)
        hs, ys = model.forward_all(z)

        if len(hs) >= 2 and len(ys) >= 2:
            loss, _, _ = clustering_loss(ys[0], ys[1], hs[0], hs[1], entropy_weight=entropy_weight)
        else:
            loss = torch.tensor(0.0, device=device)
        losses.append(float(loss.detach().cpu()))

        if y is not None and len(ys) >= 2:
            pred = predict_fusion(ys[0], ys[1]).cpu().numpy()
            all_pred.append(pred)
            all_y.append(y)

    out = {"val_clu_loss": float(np.mean(losses)) if losses else 0.0}

    if all_y:
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_pred)
        out.update(clustering_metrics_from_predictions(y_true, y_pred))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Contrastive clustering on latents")
    parser.add_argument("--data_name", required=True, help="Dataset name (e.g., NoisyMNIST, CUB)")
    parser.add_argument("--output_dir", default=None, help="Override default output directory")
    args = parser.parse_args()

    cfg = build_stage3_config(args.data_name)

    set_seed(int(cfg["seed"]))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("GPU: " + str(use_cuda))
    print("Data set: " + str(args.data_name))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"outputs/stage3_clu_{dataset_slug(args.data_name)}")
    
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    pred_dir = out_dir / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 2 completed latents...")
    data = np.load(cfg["stage2"]["completed_latents"])
    z = data["latents_completed"].astype(np.float32)
    labels = data["labels"].astype(np.int64) if "labels" in data.files else None

    n = z.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_val = int(float(cfg["training"]["val_split"]) * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    z_train = torch.from_numpy(z[train_idx])
    z_val = torch.from_numpy(z[val_idx])

    if labels is not None:
        y_train = torch.from_numpy(labels[train_idx])
        y_val = torch.from_numpy(labels[val_idx])
        train_ds = TensorDataset(z_train, y_train)
        val_ds = TensorDataset(z_val, y_val)
        n_clusters = int(len(np.unique(labels)))
    else:
        train_ds = TensorDataset(z_train)
        val_ds = TensorDataset(z_val)
        n_clusters = int(z.shape[1])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = ContrastiveClusteringHead(
        n_views=int(z.shape[1]),
        latent_dim=int(z.shape[2]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        proj_dim=int(cfg["model"]["proj_dim"]),
        n_clusters=n_clusters,
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"]["save_every"])
    entropy_weight = float(cfg["loss"]["entropy_weight"])

    best_loss = float("inf")
    history = []
    print("Start training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            z_b = batch[0].to(device)
            hs, ys = model.forward_all(z_b)

            if len(hs) >= 2 and len(ys) >= 2:
                loss, _, _ = clustering_loss(ys[0], ys[1], hs[0], hs[1], entropy_weight=entropy_weight)
            else:
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lf = float(loss.detach().cpu())
            train_losses.append(lf)

        val_stats = eval_epoch(model, val_loader, device, entropy_weight)
        train_mean = float(np.mean(train_losses)) if train_losses else 0.0

        rec = {"epoch": epoch, "train_clu_loss": train_mean}
        rec.update(val_stats)
        history.append(rec)

        msg = "Epoch: {:.0f}/{:.0f} ==> loss = {:.4f} | train_clu={:.4f} val_clu={:.4f}".format(
            epoch,
            epochs,
            train_mean,
            train_mean,
            val_stats["val_clu_loss"],
        )
        if "acc" in val_stats:
            msg += f" acc={val_stats['acc']:.4f} nmi={val_stats['nmi']:.4f} ari={val_stats['ari']:.4f}"
        print(msg)

        if val_stats["val_clu_loss"] < best_loss:
            best_loss = val_stats["val_clu_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "n_clusters": n_clusters,
                    "best_val_clu_loss": best_loss,
                },
                ckpt_dir / "best_stage3_clu.pt",
            )
        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "n_clusters": n_clusters,
                },
                ckpt_dir / f"stage3_clu_epoch_{epoch:04d}.pt",
            )

    with open(logs_dir / "stage3_clu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final fused prediction on all samples
    model.eval()
    z_all = torch.from_numpy(z).to(device)
    with torch.no_grad():
        _, ys = model.forward_all(z_all)
        y_pred = predict_fusion(ys[0], ys[1]).cpu().numpy()

    out = {"y_pred": y_pred}
    if labels is not None:
        out["y_true"] = labels
    np.savez_compressed(pred_dir / "stage3_predictions.npz", **out)

    print("-------------------The training over data set = " + str(args.data_name) + "--------------------")
    print("[DONE] Stage 3 complete. Predictions saved in " + str(pred_dir))


if __name__ == "__main__":
    main()
