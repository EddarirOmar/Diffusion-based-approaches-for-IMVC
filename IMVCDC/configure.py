"""Centralized dataset configuration for IMVCDC.

Provides a compact, dataset-first config style similar to DCG (NEW),
and adapters that generate stage-specific configs for Stage 1/2/3 scripts.
"""

from __future__ import annotations

import re
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOTS = [
    _THIS_DIR.parent / "datasets",
    _THIS_DIR / "data" / "raw",
]


def _resolve_dataset_path(filename: str) -> str:
    """Resolve dataset path, preferring ../datasets then IMVCDC/data/raw."""
    for root in _DEFAULT_DATA_ROOTS:
        candidate = root / filename
        if candidate.exists():
            return str(candidate)
    return str(_DEFAULT_DATA_ROOTS[0] / filename)


def _with_multiview_fields(cfg: dict) -> dict:
    ae_cfg = cfg["Autoencoder"]
    diff_cfg = cfg["diffusion"]

    if "archs" not in ae_cfg and "input_dims" in ae_cfg:
        hidden_dims = ae_cfg.get("hidden_dims", [1024, 1024, 1024])
        latent_dim = ae_cfg.get("latent_dim", diff_cfg.get("emb_size", 128))
        ae_cfg["archs"] = [
            [in_dim] + list(hidden_dims) + [latent_dim] for in_dim in ae_cfg["input_dims"]
        ]

    if "activations" not in ae_cfg and "activation" in ae_cfg:
        ae_cfg["activations"] = [ae_cfg["activation"]] * len(ae_cfg.get("archs", []))

    if "out_dims" not in diff_cfg:
        if "archs" in ae_cfg and len(ae_cfg["archs"]) > 0:
            diff_cfg["out_dims"] = [arch[-1] for arch in ae_cfg["archs"]]
        else:
            diff_cfg["out_dims"] = [diff_cfg.get("emb_size", 128)] * len(ae_cfg.get("input_dims", [0, 0]))

    return cfg


def _canonical_name(data_name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", data_name.lower())
    aliases = {
        "multifashion": "Multi-Fashion",
        "fashion": "Multi-Fashion",
        "multicoil20": "Multi-Coil20",
        "coil20": "Multi-Coil20",
        "noisymnist": "NoisyMNIST",
        "noisydigitproduct": "NoisyDigit-Product",
        "noisydigit": "NoisyDigit-Product",
        "cub": "CUB",
        "synthetic3d": "Synthetic3d",
        "handwritten": "HandWritten",
        "landuse21": "LandUse-21",
        "scene15": "Scene-15",
        "caltech1017": "Caltech101-7",
    }
    if key not in aliases:
        raise ValueError(f"Undefined data_name: {data_name}")
    return aliases[key]


def dataset_slug(data_name: str) -> str:
    return _canonical_name(data_name).lower().replace("-", "").replace(" ", "")


def get_default_config(data_name: str) -> dict:
    data_name = _canonical_name(data_name)

    if data_name == "Synthetic3d":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="Synthetic3d", path=_resolve_dataset_path("synthetic3d.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[3, 3], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=6, mask_seed=5, missing_rate=0.3, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=3),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "HandWritten":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="HandWritten", path=_resolve_dataset_path("Handwritten.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[76, 64], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=6, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=10),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "CUB":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="CUB", path=_resolve_dataset_path("cub_googlenet_doc2vec_c10.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[1024, 300], hidden_dims=[512, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=8, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=10),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "LandUse-21":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="LandUse-21", path=_resolve_dataset_path("LandUse-21.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[59, 40], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=3, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=21),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "Scene-15":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="Scene-15", path=_resolve_dataset_path("Scene-15.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[20, 59], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=3, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=15),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "Caltech101-7":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="Caltech101-7", path=_resolve_dataset_path("Caltech101-7.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[48, 40], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=3, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=7),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "Multi-Fashion":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="Multi-Fashion", path=_resolve_dataset_path("multi_fashion.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[784, 784], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=2, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=10),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=50, beta_schedule="linear"),
            )
        )

    if data_name == "Multi-Coil20":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="Multi-Coil20", path=_resolve_dataset_path("multi_coil20.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[1024, 1024], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=2, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=20),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "NoisyMNIST":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="NoisyMNIST", path=_resolve_dataset_path("NoisyMNIST.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[784, 784], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=2, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=10),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    if data_name == "NoisyDigit-Product":
        return _with_multiview_fields(
            dict(
                dataset=dict(name="NoisyDigit-Product", path=_resolve_dataset_path("noisydigit_product.mat"), test_split=0.2),
                Autoencoder=dict(input_dims=[784, 784], hidden_dims=[1024, 1024, 1024], latent_dim=128, activation="relu", batchnorm=True),
                training=dict(seed=2, mask_seed=5, missing_rate=0.5, batch_size=256, epoch=200, lr=1.0e-4, n_clusters=10),
                diffusion=dict(emb_size=128, time_type="sinusoidal"),
                noise_scheduler=dict(num_timesteps=100, beta_schedule="linear"),
            )
        )

    raise ValueError(f"Undefined data_name: {data_name}")


def build_stage1_config(data_name: str, data_path: str | None = None) -> dict:
    cfg = get_default_config(data_name)
    train_cfg = cfg["training"]
    ae_cfg = cfg["Autoencoder"]

    hidden_dims = list(ae_cfg.get("hidden_dims", [512]))
    hidden_dim = int(hidden_dims[0]) if hidden_dims else 512

    return {
        "seed": int(train_cfg.get("seed", 42)),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "path": str(data_path or cfg["dataset"]["path"]),
            "missing_rate": float(train_cfg.get("missing_rate", 0.5)),
            "test_split": float(cfg["dataset"].get("test_split", 0.2)),
        },
        "reconstruction": {
            "epochs": int(train_cfg.get("epoch", 80)),
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "learning_rate": float(train_cfg.get("lr", 1.0e-3)),
            "weight_decay": 1.0e-5,
            "latent_dim": int(ae_cfg.get("latent_dim", 128)),
            "hidden_dim": hidden_dim,
            "dropout": 0.1,
            "log_every": 20,
            "save_every": 10,
        },
    }


def build_stage2_config(data_name: str, output_root: str = "outputs") -> dict:
    cfg = get_default_config(data_name)
    train_cfg = cfg["training"]
    ns_cfg = cfg["noise_scheduler"]
    slug = dataset_slug(data_name)

    return {
        "seed": int(train_cfg.get("seed", 42)),
        "stage1": {
            "train_latents": f"{output_root}/stage1_rec_{slug}/latents/train_latents.npz",
            "test_latents": f"{output_root}/stage1_rec_{slug}/latents/test_latents.npz",
        },
        "diffusion": {
            "timesteps": int(ns_cfg.get("num_timesteps", 1000)),
            "schedule": str(ns_cfg.get("beta_schedule", "linear")),
        },
        "model": {
            "hidden_dim": 256,
            "time_dim": 128,
            "num_heads": 4,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 40,
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "target_view": 0,
            "log_every": 20,
            "save_every": 5,
        },
        "inference": {"num_steps": min(200, int(ns_cfg.get("num_timesteps", 1000)))},
    }


def build_stage3_config(data_name: str, output_root: str = "outputs") -> dict:
    cfg = get_default_config(data_name)
    train_cfg = cfg["training"]
    slug = dataset_slug(data_name)

    return {
        "seed": int(train_cfg.get("seed", 42)),
        "stage2": {
            "completed_latents": f"{output_root}/stage2_dm_{slug}/latents/test_latents_completed.npz",
        },
        "model": {
            "hidden_dim": 256,
            "proj_dim": 128,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 50,
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "val_split": 0.2,
            "log_every": 20,
            "save_every": 5,
        },
        "loss": {"entropy_weight": 1.0},
    }
