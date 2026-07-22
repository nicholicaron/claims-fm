"""Baseline models: L2 logistic regression (small grid) and XGBoost (seeded
random search, early stopping). All selection on VAL AUPRC — test is never
touched here."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def tune_lr(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, cfg: dict[str, Any]
) -> tuple[Pipeline, dict[str, Any]]:
    best, best_ap, best_params = None, -1.0, None
    for C in cfg["C_grid"]:
        for cw in cfg["class_weight"]:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            C=C, class_weight=cw, max_iter=cfg["max_iter"], solver="lbfgs"
                        ),
                    ),
                ]
            )
            model.fit(Xtr, ytr)
            ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
            log.info("LR C=%g cw=%s -> val AUPRC %.4f", C, cw, ap)
            if ap > best_ap:
                best, best_ap, best_params = model, ap, {"C": C, "class_weight": cw}
    return best, {"val_auprc": best_ap, "params": best_params}


def _draw_params(rng: np.random.Generator, space: dict[str, Any], neg_over_pos: float) -> dict[str, Any]:
    lo_d, hi_d = space["max_depth"]
    spw_choices = [1.0 if v == 1.0 else neg_over_pos for v in space["scale_pos_weight"]]
    return {
        "max_depth": int(rng.integers(lo_d, hi_d + 1)),
        "learning_rate": float(np.exp(rng.uniform(*np.log(space["learning_rate"])))),
        "min_child_weight": int(rng.choice(space["min_child_weight"])),
        "subsample": float(rng.uniform(*space["subsample"])),
        "colsample_bytree": float(rng.uniform(*space["colsample_bytree"])),
        "reg_lambda": float(np.exp(rng.uniform(*np.log(space["reg_lambda"])))),
        "scale_pos_weight": float(rng.choice(spw_choices)),
    }


def tune_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[XGBClassifier, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    neg_over_pos = float((ytr == 0).sum() / max(1, ytr.sum()))
    space = {k: v for k, v in cfg["space"].items()}

    best, best_ap, best_params = None, -1.0, None
    for i in range(cfg["n_search"]):
        params = _draw_params(rng, space, neg_over_pos)
        model = XGBClassifier(
            n_estimators=cfg["max_estimators"],
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=cfg["early_stopping_rounds"],
            n_jobs=0,
            random_state=seed,
            **params,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
        log.info("XGB draw %d/%d %s -> val AUPRC %.4f (trees %d)",
                 i + 1, cfg["n_search"], params, ap, model.best_iteration + 1)
        if ap > best_ap:
            best, best_ap, best_params = model, ap, {
                **params, "best_iteration": int(model.best_iteration)
            }
    return best, {"val_auprc": best_ap, "params": best_params}


def refit_xgb(
    params: dict[str, Any],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> XGBClassifier:
    """Refit the selected config on a (sub)set — used for label-efficiency fits."""
    p = {k: v for k, v in params.items() if k != "best_iteration"}
    model = XGBClassifier(
        n_estimators=cfg["max_estimators"],
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=cfg["early_stopping_rounds"],
        n_jobs=0,
        random_state=seed,
        **p,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model
