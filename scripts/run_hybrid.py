"""M5.5 hybrid experiment: XGBoost re-tuned with the *identical* M2 protocol
(same search space, draws, and seed) on engineered features augmented with
frozen pretrained [CLS] embeddings. Same model family, same tuning, only the
feature set changes — any delta is attributable to the encoder.

Also runs the Task B label-efficiency grid for the hybrid (same subsample
protocol as M2), completing the money chart's practical fourth line.

This is a post-freeze follow-up experiment: the M2/M4/M5 frozen results are
untouched; the hybrid's own test numbers come from a single scoring pass at
the end of this script.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from claimsfm.baselines.models import refit_xgb, tune_xgb
from claimsfm.config import REPO_ROOT, data_path, load_config
from claimsfm.eval.calibration import fit_best_calibrator
from claimsfm.eval.metrics import bootstrap_ci, capture_at, core_metrics, precision_at_k, recall_at_k

log = logging.getLogger(__name__)

A_LABELS = ["label_ip", "label_cost"]


def _matrix(df: pl.DataFrame, feature_cols: list[str], emb: np.ndarray) -> np.ndarray:
    X = df.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    return np.hstack([np.nan_to_num(X, copy=False), emb])


def _emb_for(df: pl.DataFrame, emb_df: pl.DataFrame, id_col: str) -> np.ndarray:
    joined = df.select(id_col).join(emb_df, on=id_col, how="left", maintain_order="left")
    return np.stack(joined["emb"].to_list()).astype(np.float32)


def run_task_a(bl: dict, cfg_data: dict, results: dict) -> None:
    proc = data_path(cfg_data, "processed")
    feats = pl.read_parquet(proc / "task_a_features.parquet")
    meta = json.loads((proc / "task_a_meta.json").read_text())
    emb_df = pl.read_parquet(proc / "task_a_pack" / "cls_embeddings.parquet")
    caps = bl["eval"]["lift_capacities"]

    for label in A_LABELS:
        parts = {}
        for split in ("train", "val", "test"):
            sub = feats.filter(pl.col("split") == split)
            parts[split] = (
                _matrix(sub, meta["feature_cols"], _emb_for(sub, emb_df, "DESYNPUF_ID")),
                sub[label].to_numpy().astype(np.int8),
            )
        (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
        model, info = tune_xgb(Xtr, ytr, Xva, yva, bl["models"]["xgb"], bl["seed"])
        p_val = model.predict_proba(Xva)[:, 1]
        cal_name, cal = fit_best_calibrator(p_val, yva)
        p_te = np.clip(cal.transform(model.predict_proba(Xte)[:, 1]), 0, 1)

        fns = {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss}
        for c in caps:
            fns[f"capture_at_{int(c*100)}pct"] = (lambda c: lambda y, p: capture_at(y, p, c))(c)
        results[f"task_a/{label}"] = {
            "val_auprc": info["val_auprc"],
            "calibrator": cal_name,
            "test": core_metrics(yte, p_te, bl["eval"]["ece_bins"]),
            "ci": bootstrap_ci(yte, p_te, fns, bl["eval"]["bootstrap_n"], bl["seed"]),
        }
        log.info("task_a/%s hybrid: val AUPRC %.4f", label, info["val_auprc"])


def run_task_b(bl: dict, cfg_data: dict, results: dict) -> None:
    proc = data_path(cfg_data, "processed")
    feats = pl.read_parquet(proc / "task_b_features.parquet")
    meta = json.loads((proc / "task_b_meta.json").read_text())
    emb_df = pl.read_parquet(proc / "task_b_pack" / "cls_embeddings.parquet")
    ks = bl["task_b"]["precision_at_k"]

    parts = {}
    for split in ("train", "val", "test"):
        sub = feats.filter(pl.col("split") == split)
        parts[split] = (
            _matrix(sub, meta["feature_cols"], _emb_for(sub, emb_df, "Provider")),
            sub["label"].to_numpy().astype(np.int8),
        )
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    model, info = tune_xgb(Xtr, ytr, Xva, yva, bl["models"]["xgb"], bl["seed"])
    p_te = model.predict_proba(Xte)[:, 1]  # raw: frozen Task B protocol

    fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
    for k in ks:
        fns[f"precision_at_{k}"] = (lambda k: lambda y, p: precision_at_k(y, p, k))(k)
    results["task_b"] = {
        "val_auprc": info["val_auprc"],
        "test": core_metrics(yte, p_te),
        "ci": bootstrap_ci(yte, p_te, fns, bl["eval"]["bootstrap_n"], bl["seed"]),
        "operating_point": {
            "k": bl["task_b"]["operating_point_k"],
            "precision": precision_at_k(yte, p_te, bl["task_b"]["operating_point_k"]),
            "recall": recall_at_k(yte, p_te, bl["task_b"]["operating_point_k"]),
        },
    }

    # label-efficiency grid, same subsample protocol as M2
    rng = np.random.default_rng(bl["seed"])
    le: dict[str, list[float]] = {"1.0": [float(average_precision_score(yte, p_te))]}
    for frac in bl["task_b"]["label_efficiency_fracs"]:
        if frac == 1.0:
            continue
        vals = []
        for _ in range(bl["task_b"]["label_efficiency_seeds"]):
            seed = int(rng.integers(0, 2**31))
            idx = _stratified(ytr, frac, seed)
            m = refit_xgb(info["params"], Xtr[idx], ytr[idx], Xva, yva, bl["models"]["xgb"], seed)
            vals.append(float(average_precision_score(yte, m.predict_proba(Xte)[:, 1])))
        le[str(frac)] = vals
    results["task_b_label_efficiency_test_auprc"] = le
    log.info("task_b hybrid: val AUPRC %.4f, LE %s", info["val_auprc"], le)


def _stratified(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        keep.append(idx[: max(1, int(round(frac * len(idx))))])
    return np.sort(np.concatenate(keep))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bl = load_config("configs/baselines.yaml")
    cfg_data = load_config("configs/data.yaml")
    results: dict = {}
    run_task_b(bl, cfg_data, results)
    run_task_a(bl, cfg_data, results)
    out = REPO_ROOT / "reports" / "metrics_hybrid.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
