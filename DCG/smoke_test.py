import itertools
import os
import random

import numpy as np
import torch

from configure import get_default_config
from datasets import load_data
from get_indicator_matrix_A import get_mask
from ICDM import icdm


def main():
    config = get_default_config("Synthetic3d")
    config["dataset"] = "Synthetic3d"
    config["training"]["epoch"] = 2
    config["training"]["batch_size"] = 4
    config["noise_scheduler"]["num_timesteps"] = 10
    config["training"]["lambda_mmi"] = 0.01
    config["training"]["lambda_cluster"] = 0.1
    config["training"]["lambda_hc"] = 0.1
    config["print_num"] = 1

    seed = config["training"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    x_list, y_list = load_data(config)
    x1_train_raw = x_list[0]
    x2_train_raw = x_list[1]

    missing_rate = 0.3
    mask = get_mask(2, x1_train_raw.shape[0], missing_rate)

    x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
    x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

    x1_train = torch.from_numpy(x1_train).float().to(device)
    x2_train = torch.from_numpy(x2_train).float().to(device)
    mask = torch.from_numpy(mask).long().to(device)

    model = icdm(config)
    model.to_device(device)

    optimizer = torch.optim.Adam(
        itertools.chain(
            model.autoencoder1.parameters(),
            model.autoencoder2.parameters(),
            model.df1.parameters(),
            model.df2.parameters(),
            model.clusterLayer.parameters(),
            model.AttentionLayer.parameters(),
        ),
        lr=config["training"]["lr"],
    )

    acc, nmi, ari = model.train(config, x1_train, x2_train, y_list, mask, optimizer, device)
    print("SMOKE_RESULT", f"ACC={acc:.4f}", f"NMI={nmi:.4f}", f"ARI={ari:.4f}")


if __name__ == "__main__":
    main()
