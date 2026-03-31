import argparse
import itertools
import os
import random

import numpy as np
import torch

from get_indicator_matrix_A import get_mask
from datasets import *
from configure import get_default_config
from ICDM import *


def _build_optimizer(model, lr):
    ae_params = itertools.chain.from_iterable(ae.parameters() for ae in model.autoencoders)
    df_params = itertools.chain.from_iterable(df.parameters() for df in model.dfs)
    return torch.optim.Adam(
        itertools.chain(ae_params, df_params, model.clusterLayer.parameters(), model.AttentionLayer.parameters()),
        lr=lr,
    )


def _prepare_inputs(x_list, missing_rate, device):
    n_views = len(x_list)
    n_samples = x_list[0].shape[0]
    mask = get_mask(n_views, n_samples, missing_rate)

    x_train_list = []
    for v in range(n_views):
        x_masked = x_list[v] * mask[:, v][:, np.newaxis]
        x_train_list.append(torch.from_numpy(x_masked).float().to(device))

    return x_train_list, torch.from_numpy(mask).long().to(device)


def main(MR=[0.3]):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    config = get_default_config(dataset)
    config['dataset'] = dataset
    if args.epoch is not None:
        config['training']['epoch'] = int(args.epoch)
    print("Data set: " + config['dataset'])
    config['print_num'] = 1
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    n_views = len(X_list)
    if n_views < 2:
        raise ValueError('At least two views are required for training.')

    for missingrate in MR:
        config['training']['missing_rate'] = missingrate
        print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
        for data_seed in range(1, args.test_time + 1):
            run_seed = seed + data_seed - 1
            np.random.seed(run_seed)
            random.seed(run_seed)
            os.environ['PYTHONHASHSEED'] = str(run_seed)
            torch.manual_seed(run_seed)
            if use_cuda:
                torch.cuda.manual_seed_all(run_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            x_train_list, mask = _prepare_inputs(X_list, config['training']['missing_rate'], device)
            # Build the model
            ICDM = icdm(config)
            optimizer = _build_optimizer(ICDM, config['training']['lr'])
            ICDM.to_device(device)
            # Training
            acc, nmi, ari = ICDM.train(config, x_train_list, Y_list, mask, optimizer, device)
            print('-------------------The ' + str(data_seed) + ' training over Missing rate = ' + str(missingrate) + '--------------------')
            print("ACC {:.2f}, NMI {:.2f}, ARI {:.2f}".format(acc, nmi, ari))

if __name__ == '__main__':
    dataset = {
               1: "LandUse_21",
               2: "CUB",
               3: "HandWritten",
               4: "Multi-Fashion",
               5: 'Synthetic3d',
               }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=5, help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=1, help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--epoch', type=int, default=None, help='override training epochs')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.3]

    main(MR=MisingRate)
