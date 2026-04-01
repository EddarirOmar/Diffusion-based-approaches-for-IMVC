import argparse
import csv
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from get_indicator_matrix_A import get_mask
from datasets import load_data
from configure import get_default_config
from ICDM import *


def _parse_missing_rates(raw_value):
    parts = [p.strip() for p in str(raw_value).split(',') if p.strip()]
    if not parts:
        raise ValueError('missing_rate must contain at least one float value.')
    rates = [float(p) for p in parts]
    for r in rates:
        if r < 0.0 or r >= 1.0:
            raise ValueError(f'Invalid missing rate: {r}. Expected 0 <= rate < 1.')
    return rates


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


def _apply_lambda_config(config, lambda_config_path):
    with open(lambda_config_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    params = payload.get('best_params', payload)
    if not isinstance(params, dict):
        raise ValueError('lambda_config must be a JSON object or contain a best_params object.')

    applied = {}
    for k, v in params.items():
        if k.startswith('lambda_') or k in ('mmi_temperature', 'mmi_internal_lambda'):
            config['training'][k] = float(v)
            applied[k] = float(v)

    if not applied:
        raise ValueError('No lambda_*, mmi_temperature, or mmi_internal_lambda keys found in lambda_config.')

    return applied


def _safe_rate_str(x):
    return str(x).replace('.', 'p')


def _build_checkpoint_path(root_dir, dataset_name, missing_rate, data_seed, tag):
    fname = f"{dataset_name}_mr{_safe_rate_str(missing_rate)}_seed{data_seed}_{tag}.pt"
    return os.path.join(root_dir, fname)


def _build_metrics_path(root_dir, dataset_name, missing_rate, data_seed, ext):
    fname = f"{dataset_name}_mr{_safe_rate_str(missing_rate)}_seed{data_seed}_metrics.{ext}"
    return os.path.join(root_dir, fname)


def _save_checkpoint(path, model, optimizer, config, run_seed, data_seed, missing_rate, metrics):
    payload = {
        'config': config,
        'run_seed': int(run_seed),
        'data_seed': int(data_seed),
        'missing_rate': float(missing_rate),
        'metrics': {
            'acc': float(metrics['acc']),
            'nmi': float(metrics['nmi']),
            'ari': float(metrics['ari']),
            'score': float(metrics['score']),
        },
        'model_state': model.checkpoint_state(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(payload, path)
    print('Checkpoint saved:', path)


def _save_metrics_history_json(path, history_payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history_payload, f, indent=2)
    print('Metrics JSON saved:', path)


def _save_metrics_history_csv(path, history_rows):
    if not history_rows:
        return
    fieldnames = [
        'epoch',
        'loss',
        'rec_loss',
        'df_loss',
        'ce_loss',
        'mmi_loss',
        'cluster_loss',
        'hc_loss',
        'accuracy',
        'NMI',
        'ARI',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print('Metrics CSV saved:', path)


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
    if args.lambda_config is not None:
        applied = _apply_lambda_config(config, args.lambda_config)
        print('Applied tuned params from', args.lambda_config)
        for k in sorted(applied.keys()):
            print(f'  {k}: {applied[k]}')
    print("Data set: " + config['dataset'])
    config['print_num'] = 1
    ckpt_root = args.checkpoint_dir
    if not os.path.isabs(ckpt_root):
        ckpt_root = str(Path(__file__).resolve().parent / ckpt_root)
    if not args.no_checkpoint:
        os.makedirs(ckpt_root, exist_ok=True)
        print('Checkpoint dir:', ckpt_root)

    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    n_views = len(X_list)
    if n_views < 2:
        raise ValueError('At least two views are required for training.')

    for missingrate in MR:
        config['training']['missing_rate'] = missingrate
        print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
        best_score = -1.0
        best_ckpt_path = None
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

            if args.resume_checkpoint and data_seed == 1:
                if os.path.exists(args.resume_checkpoint):
                    resume_payload = torch.load(args.resume_checkpoint, map_location=device)
                    ICDM.load_checkpoint_state(resume_payload['model_state'])
                    if 'optimizer_state' in resume_payload:
                        optimizer.load_state_dict(resume_payload['optimizer_state'])
                    print('Resumed from checkpoint:', args.resume_checkpoint)
                else:
                    print('Resume checkpoint not found, skipping:', args.resume_checkpoint)

            # Training
            acc, nmi, ari, history = ICDM.train(
                config,
                x_train_list,
                Y_list,
                mask,
                optimizer,
                device,
                return_history=True,
            )
            print('-------------------The ' + str(data_seed) + ' training over Missing rate = ' + str(missingrate) + '--------------------')
            print("ACC {:.2f}, NMI {:.2f}, ARI {:.2f}".format(acc, nmi, ari))

            if not args.no_metrics:
                metrics_json_path = _build_metrics_path(ckpt_root, dataset, missingrate, data_seed, 'json')
                metrics_csv_path = _build_metrics_path(ckpt_root, dataset, missingrate, data_seed, 'csv')
                history_payload = {
                    'dataset': dataset,
                    'missing_rate': float(missingrate),
                    'run_seed': int(run_seed),
                    'data_seed': int(data_seed),
                    'summary': {
                        'acc': float(acc),
                        'nmi': float(nmi),
                        'ari': float(ari),
                        'score': float((acc + nmi + ari) / 3.0),
                    },
                    'history': history,
                }
                _save_metrics_history_json(metrics_json_path, history_payload)
                _save_metrics_history_csv(metrics_csv_path, history)

            score = float((acc + nmi + ari) / 3.0)
            run_metrics = {
                'acc': acc,
                'nmi': nmi,
                'ari': ari,
                'score': score,
            }

            if not args.no_checkpoint:
                last_ckpt = _build_checkpoint_path(ckpt_root, dataset, missingrate, data_seed, 'last')
                _save_checkpoint(last_ckpt, ICDM, optimizer, config, run_seed, data_seed, missingrate, run_metrics)

                if score >= best_score:
                    best_score = score
                    best_ckpt_path = _build_checkpoint_path(ckpt_root, dataset, missingrate, data_seed, 'best')
                    _save_checkpoint(best_ckpt_path, ICDM, optimizer, config, run_seed, data_seed, missingrate, run_metrics)

        if not args.no_checkpoint and best_ckpt_path is not None:
            print('Best checkpoint for missing rate', missingrate, '->', best_ckpt_path)

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
    parser.add_argument('--lambda_config', type=str, default=None, help='path to best_lambda_params.json or JSON with lambda_* values')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to checkpoint for resume (applied to first run)')
    parser.add_argument('--no_checkpoint', action='store_true', help='disable checkpoint save')
    parser.add_argument('--no_metrics', action='store_true', help='disable per-epoch metrics save')
    parser.add_argument('--missing_rate', type=str, default='0.3', help='missing rate(s), e.g. 0.1 or 0.1,0.3')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = _parse_missing_rates(args.missing_rate)

    main(MR=MisingRate)
