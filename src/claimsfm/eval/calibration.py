"""Recalibration (SPEC §6 Task A): Platt and isotonic fit on VAL predictions,
winner chosen by val Brier, test reported before/after."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class PlattCalibrator:
    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e10, max_iter=1000)

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> "PlattCalibrator":
        z = _logit(p_val)
        self.lr.fit(z.reshape(-1, 1), y_val)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(p_val, y_val)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_best_calibrator(p_val: np.ndarray, y_val: np.ndarray) -> tuple[str, Any]:
    """Fit Platt + isotonic on val; return (name, calibrator) with lower val Brier."""
    candidates = {
        "platt": PlattCalibrator().fit(p_val, y_val),
        "isotonic": IsotonicCalibrator().fit(p_val, y_val),
    }
    scores = {
        name: brier_score_loss(y_val, np.clip(c.transform(p_val), 0, 1))
        for name, c in candidates.items()
    }
    best = min(scores, key=scores.get)
    return best, candidates[best]


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    return {"mean_predicted": xs, "fraction_positive": ys, "bin_counts": ns}
