from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RoutingMetrics:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    under_routing_rate: float
    over_routing_rate: float
    confusion: np.ndarray
    precision: dict[int, float]
    recall: dict[int, float]


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_samples, len(values)))
    stats = np.array([stat(values[row]) for row in idx])
    alpha = (1 - ci) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def confusion_matrix(y_true: list[int], y_pred: list[int], tiers: list[int]) -> np.ndarray:
    pos = {t: i for i, t in enumerate(tiers)}
    matrix = np.zeros((len(tiers), len(tiers)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[pos[t], pos[p]] += 1
    return matrix


def routing_metrics(
    y_true: list[int],
    y_pred: list[int],
    tiers: list[int],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> RoutingMetrics:
    true = np.array(y_true)
    pred = np.array(y_pred)
    n = len(true)
    correct = (true == pred).astype(float)
    matrix = confusion_matrix(y_true, y_pred, tiers)
    precision, recall = {}, {}
    for i, tier in enumerate(tiers):
        col, row = matrix[:, i].sum(), matrix[i, :].sum()
        precision[tier] = float(matrix[i, i] / col) if col else float("nan")
        recall[tier] = float(matrix[i, i] / row) if row else float("nan")
    return RoutingMetrics(
        n=n,
        accuracy=float(correct.mean()) if n else float("nan"),
        accuracy_ci=bootstrap_ci(correct, n_samples=n_bootstrap, ci=ci, seed=seed),
        under_routing_rate=float((pred < true).mean()) if n else float("nan"),
        over_routing_rate=float((pred > true).mean()) if n else float("nan"),
        confusion=matrix,
        precision=precision,
        recall=recall,
    )
