import itertools
import argparse
import os
import random

import numpy as np
import torch

from configure import get_default_config
from datasets import load_data
from ICDM import icdm
from utils import build_optimizer, prepare_inputs


def _run_smoke(config, x_list, y_list, device):
    x_train_list, mask = prepare_inputs(x_list, config["training"]["missing_rate"], device)

    model = icdm(config)
    model.to_device(device)
    optimizer = build_optimizer(model, config["training"]["lr"])

    acc, nmi, ari = model.train(config, x_train_list, y_list, mask, optimizer, device)
    print("SMOKE_RESULT", f"ACC={acc:.4f}", f"NMI={nmi:.4f}", f"ARI={ari:.4f}")


def main(dataset_name="synthetic3d"):
    dataset_name = dataset_name.lower()

    if dataset_name == "landuse":
        config = get_default_config("LandUse_21_3View")
        config["dataset"] = "LandUse_21_3View"
        config["training"]["epoch"] = 2
        config["training"]["batch_size"] = 32
        config["noise_scheduler"]["num_timesteps"] = 10
        config["training"]["missing_rate"] = 0.5
    else:
        config = get_default_config("Synthetic3d")
        config["dataset"] = "Synthetic3d"
        config["training"]["epoch"] = 2
        config["training"]["batch_size"] = 4
        config["noise_scheduler"]["num_timesteps"] = 10
        config["training"]["missing_rate"] = 0.3

    config["training"]["lambda_mmi"] = 0.01
    config["training"]["lambda_cluster"] = 0.1
    config["training"]["lambda_hc"] = 0.1
    config["print_num"] = 1

    seed = config["training"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    if dataset_name == "landuse":
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LandUse-21.mat")
        if os.path.exists(data_path):
            x_list, y_list = load_data(config)
            print("LANDUSE_SOURCE=real")
        else:
            # Fallback keeps the multi-view training path testable when LandUse files are not staged yet.
            n = 210
            dims = [20, 59, 40]
            x_list = [np.random.randn(n, d).astype("float32") for d in dims]
            y_list = [np.tile(np.arange(config["training"]["n_clusters"]), n // config["training"]["n_clusters"])[:n]]
            print("LANDUSE_SOURCE=fallback_synthetic")
    else:
        x_list, y_list = load_data(config)

    _run_smoke(config, x_list, y_list, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic3d", choices=["synthetic3d", "landuse"])
    args = parser.parse_args()
    main(dataset_name=args.dataset)
