"""Unified metrics for IMVCDC.

This module centralizes reconstruction and clustering metrics used
across training and evaluation scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def global_ssim_like(y_true: np.ndarray, y_pred: np.ndarray, c1: float = 1e-4, c2: float = 9e-4) -> float:
    """A lightweight global SSIM-style score for vector features."""
    mu_x = np.mean(y_true)
    mu_y = np.mean(y_pred)
    sigma_x = np.var(y_true)
    sigma_y = np.var(y_pred)
    sigma_xy = np.mean((y_true - mu_x) * (y_pred - mu_y))

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(num / max(den, 1e-12))


def reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(y_true, y_pred),
        "ssim_like": global_ssim_like(y_true, y_pred),
    }


def clustering_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy via Hungarian alignment."""
    classes = np.unique(y_true)
    cls2id = {c: i for i, c in enumerate(classes)}
    yt = np.array([cls2id[c] for c in y_true])
    k = len(classes)

    conf = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(yt.astype(int), y_pred.astype(int)):
        if 0 <= a < k and 0 <= b < k:
            conf[b, a] += 1

    r, c = linear_sum_assignment(conf.max() - conf)
    return float(conf[r, c].sum() / max(len(y_true), 1))


def clustering_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": clustering_acc(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }


def cluster_metrics(features: np.ndarray, labels: np.ndarray, n_clusters: int | None = None) -> dict[str, float]:
    """Run k-means on features, then return ACC/NMI/ARI against labels."""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = km.fit_predict(features)
    return clustering_metrics_from_predictions(labels, pred)
