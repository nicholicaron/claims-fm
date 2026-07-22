"""Metric suite (SPEC §6): discrimination, calibration, capacity-constrained
decision metrics, and bootstrap CIs. Pure numpy/sklearn on prediction vectors."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            err += mask.sum() / total * abs(p[mask].mean() - y[mask].mean())
    return float(err)


def capture_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Share of all positives captured in the top `frac` by predicted risk."""
    k = max(1, int(round(frac * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    denom = y.sum()
    return float(y[top].sum() / denom) if denom else float("nan")


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    top = np.argsort(-p, kind="stable")[: min(k, len(p))]
    return float(y[top].mean())


def recall_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    top = np.argsort(-p, kind="stable")[: min(k, len(p))]
    denom = y.sum()
    return float(y[top].sum() / denom) if denom else float("nan")


def core_metrics(y: np.ndarray, p: np.ndarray, ece_bins: int = 10) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p, ece_bins),
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile 95% CIs via unit-level resampling."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {
        name: {"point": float(fn(y, p))} for name, fn in metric_fns.items()
    }
    samples: dict[str, list[float]] = {name: [] for name in metric_fns}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() in (0, n):  # degenerate resample
            continue
        for name, fn in metric_fns.items():
            samples[name].append(fn(yb, pb))
    for name, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name]["ci_lo"], out[name]["ci_hi"] = float(lo), float(hi)
    return out


def subgroup_report(
    y: np.ndarray, p: np.ndarray, groups: dict[str, np.ndarray], ece_bins: int = 10
) -> list[dict[str, Any]]:
    """Per-subgroup discrimination + calibration (equity slice, SPEC §6)."""
    rows = []
    for field, values in groups.items():
        for g in sorted(np.unique(values).tolist()):
            mask = values == g
            yg, pg = y[mask], p[mask]
            row: dict[str, Any] = {
                "field": field,
                "group": str(g),
                "n": int(mask.sum()),
                "prevalence": float(yg.mean()) if mask.any() else float("nan"),
                "mean_predicted": float(pg.mean()) if mask.any() else float("nan"),
            }
            if mask.sum() >= 100 and 0 < yg.sum() < mask.sum():
                row["auroc"] = float(roc_auc_score(yg, pg))
                row["ece"] = ece(yg, pg, ece_bins)
            rows.append(row)
    return rows
