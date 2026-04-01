import itertools
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from get_indicator_matrix_A import get_mask


def build_optimizer(model, lr):
    ae_params = itertools.chain.from_iterable(ae.parameters() for ae in model.autoencoders)
    df_params = itertools.chain.from_iterable(df.parameters() for df in model.dfs)
    return torch.optim.Adam(
        itertools.chain(
            ae_params,
            df_params,
            model.clusterLayer.parameters(),
            model.AttentionLayer.parameters(),
        ),
        lr=lr,
    )


def prepare_inputs(x_list, missing_rate, device):
    n_views = len(x_list)
    n_samples = x_list[0].shape[0]
    mask = get_mask(n_views, n_samples, missing_rate)

    x_train_list = []
    for v in range(n_views):
        x_masked = x_list[v] * mask[:, v][:, np.newaxis]
        x_train_list.append(torch.from_numpy(x_masked).float().to(device))

    return x_train_list, torch.from_numpy(mask).long().to(device)


def parse_missing_rates(raw_value):
    parts = [p.strip() for p in str(raw_value).split(',') if p.strip()]
    if not parts:
        raise ValueError('missing_rate must contain at least one float value.')
    rates = [float(p) for p in parts]
    for r in rates:
        if r < 0.0 or r >= 1.0:
            raise ValueError(f'Invalid missing rate: {r}. Expected 0 <= rate < 1.')
    return rates


def apply_lambda_config(config, lambda_config_path):
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


def safe_rate_str(x):
    return str(x).replace('.', 'p')


def build_checkpoint_path(root_dir, dataset_name, missing_rate, data_seed, tag):
    fname = f"{dataset_name}_mr{safe_rate_str(missing_rate)}_seed{data_seed}_{tag}.pt"
    return os.path.join(root_dir, fname)


def build_metrics_path(root_dir, dataset_name, missing_rate, data_seed, ext):
    fname = f"{dataset_name}_mr{safe_rate_str(missing_rate)}_seed{data_seed}_metrics.{ext}"
    return os.path.join(root_dir, fname)


def save_checkpoint(path, model, optimizer, config, run_seed, data_seed, missing_rate, metrics):
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


def save_metrics_history_json(path, history_payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history_payload, f, indent=2)
    print('Metrics JSON saved:', path)


def save_metrics_history_csv(path, history_rows):
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


def set_run_seed(run_seed, use_cuda):
    np.random.seed(run_seed)
    random.seed(run_seed)
    os.environ['PYTHONHASHSEED'] = str(run_seed)
    torch.manual_seed(run_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False