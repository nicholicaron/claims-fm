"""Baseline orchestrator.

Default mode: tune LR + XGBoost on train, select/calibrate on val, fit Task B
label-efficiency variants, and freeze model artifacts under data/models/.
Test data is never read in this mode.

--final-eval: the ONE test pass. Loads frozen artifacts, scores test, writes
baselines/results_task_{a,b}.md + metrics JSON + figures. Run once.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from claimsfm.baselines.models import refit_xgb, tune_lr, tune_xgb
from claimsfm.config import REPO_ROOT, data_path, load_config
from claimsfm.eval.calibration import fit_best_calibrator, reliability_curve
from claimsfm.eval.metrics import (
    bootstrap_ci,
    capture_at,
    core_metrics,
    precision_at_k,
    recall_at_k,
    subgroup_report,
)
from claimsfm.eval import report

log = logging.getLogger(__name__)

A_LABELS = ["label_ip", "label_cost"]


def _matrices(df: pl.DataFrame, feature_cols: list[str], label: str):
    out = {}
    for split in ("train", "val", "test"):
        sub = df.filter(pl.col("split") == split)
        X = sub.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
        X = np.nan_to_num(X, copy=False)
        out[split] = (X, sub[label].to_numpy().astype(np.int8), sub)
    return out


def _load_task(cfg, name: str, label: str):
    proc = data_path(cfg, "processed")
    df = pl.read_parquet(proc / f"{name}_features.parquet")
    meta = json.loads((proc / f"{name}_meta.json").read_text())
    return _matrices(df, meta["feature_cols"], label), meta


def tune_all(cfg, bl, models_dir: Path) -> None:
    summary = {}
    for label in A_LABELS:
        mats, _ = _load_task(cfg, "task_a", label)
        (Xtr, ytr, _), (Xva, yva, _) = mats["train"], mats["val"]
        d = models_dir / "task_a" / label
        d.mkdir(parents=True, exist_ok=True)

        lr, lr_info = tune_lr(Xtr, ytr, Xva, yva, bl["models"]["lr"])
        xgb, xgb_info = tune_xgb(Xtr, ytr, Xva, yva, bl["models"]["xgb"], bl["seed"])
        joblib.dump(lr, d / "lr.joblib")
        joblib.dump(xgb, d / "xgb.joblib")

        cals = {}
        for mname, model in (("lr", lr), ("xgb", xgb)):
            cal_name, cal = fit_best_calibrator(model.predict_proba(Xva)[:, 1], yva)
            joblib.dump(cal, d / f"{mname}_calibrator.joblib")
            cals[mname] = cal_name
        summary[f"task_a/{label}"] = {"lr": lr_info, "xgb": xgb_info, "calibrators": cals}

    mats, _ = _load_task(cfg, "task_b", "label")
    (Xtr, ytr, tr_df), (Xva, yva, _) = mats["train"], mats["val"]
    d = models_dir / "task_b"
    d.mkdir(parents=True, exist_ok=True)

    lr, lr_info = tune_lr(Xtr, ytr, Xva, yva, bl["models"]["lr"])
    xgb, xgb_info = tune_xgb(Xtr, ytr, Xva, yva, bl["models"]["xgb"], bl["seed"])
    joblib.dump(lr, d / "lr.joblib")
    joblib.dump(xgb, d / "xgb.joblib")

    le = {}
    rng = np.random.default_rng(bl["seed"])
    for frac in bl["task_b"]["label_efficiency_fracs"]:
        if frac == 1.0:
            joblib.dump(xgb, d / "le_1.0_s0.joblib")
            le["1.0"] = {"seeds": [0], "val_auprc": [xgb_info["val_auprc"]]}
            continue
        seeds, aps = [], []
        for s in range(bl["task_b"]["label_efficiency_seeds"]):
            seed = int(rng.integers(0, 2**31))
            idx = _stratified_subsample(ytr, frac, seed)
            m = refit_xgb(xgb_info["params"], Xtr[idx], ytr[idx], Xva, yva, bl["models"]["xgb"], seed)
            joblib.dump(m, d / f"le_{frac}_s{s}.joblib")
            aps.append(float(average_precision_score(yva, m.predict_proba(Xva)[:, 1])))
            seeds.append(s)
        le[str(frac)] = {"seeds": seeds, "val_auprc": aps}
    summary["task_b"] = {"lr": lr_info, "xgb": xgb_info, "label_efficiency": le}

    (models_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("tuning complete -> %s", models_dir / "tuning_summary.json")


def _stratified_subsample(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        keep.append(idx[: max(1, int(round(frac * len(idx))))])
    return np.sort(np.concatenate(keep))


def final_eval(cfg, bl, models_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    ece_bins = bl["eval"]["ece_bins"]
    n_boot = bl["eval"]["bootstrap_n"]
    caps = bl["eval"]["lift_capacities"]

    # ---------------- Task A ----------------
    results_a: dict = {"labels": {}}
    for label in A_LABELS:
        mats, meta = _load_task(cfg, "task_a", label)
        Xte, yte, te_df = mats["test"]
        d = models_dir / "task_a" / label
        entry: dict = {"models": {}}

        for mname in ("lr", "xgb"):
            model = joblib.load(d / f"{mname}.joblib")
            cal = joblib.load(d / f"{mname}_calibrator.joblib")
            p_raw = model.predict_proba(Xte)[:, 1]
            p_cal = np.clip(cal.transform(p_raw), 0, 1)

            fns = {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss}
            for c in caps:
                fns[f"capture_at_{int(c * 100)}pct"] = (lambda c: lambda y, p: capture_at(y, p, c))(c)

            entry["models"][mname] = {
                "raw": core_metrics(yte, p_raw, ece_bins),
                "calibrated": core_metrics(yte, p_cal, ece_bins),
                "ci": bootstrap_ci(yte, p_cal, fns, n_boot, bl["seed"]),
                "reliability_raw": reliability_curve(yte, p_raw, ece_bins),
                "reliability_calibrated": reliability_curve(yte, p_cal, ece_bins),
            }
            if mname == "xgb":
                age = te_df["age"].to_numpy()
                bands = bl["task_a"]["age_bands"]
                age_band = np.digitize(age, bands[1:-1])
                band_names = np.array(
                    [f"{bands[i]}-{bands[i+1]-1}" if i else f"<{bands[1]}" for i in range(len(bands) - 1)]
                )
                groups = {
                    "sex": te_df["meta_sex"].to_numpy(),
                    "race": te_df["meta_race"].to_numpy(),
                    "age_band": band_names[age_band],
                }
                entry["subgroups_xgb_calibrated"] = subgroup_report(yte, p_cal, groups, ece_bins)
                entry["capture_curve_xgb"] = {
                    "fracs": list(np.round(np.arange(0.01, 0.31, 0.01), 2)),
                    "capture": [capture_at(yte, p_cal, f) for f in np.arange(0.01, 0.31, 0.01)],
                }
        entry["meta"] = {
            "cost_threshold_train": meta.get("cost_threshold_train"),
            "splits_sha256": meta["splits_sha256"],
        }
        results_a["labels"][label] = entry

    # ---------------- Task B ----------------
    mats, meta_b = _load_task(cfg, "task_b", "label")
    Xte, yte, te_df = mats["test"]
    (Xva, yva, _) = mats["val"]
    d = models_dir / "task_b"
    results_b: dict = {"models": {}, "meta": {
        "splits_sha256": meta_b["splits_sha256"],
        "bene_overlap": meta_b["bene_overlap"],
    }}
    ks = bl["task_b"]["precision_at_k"]
    op_k = bl["task_b"]["operating_point_k"]

    for mname in ("lr", "xgb"):
        model = joblib.load(d / f"{mname}.joblib")
        p = model.predict_proba(Xte)[:, 1]
        fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
        for k in ks:
            fns[f"precision_at_{k}"] = (lambda k: lambda y, pp: precision_at_k(y, pp, k))(k)
        results_b["models"][mname] = {
            "core": core_metrics(yte, p, ece_bins),
            "ci": bootstrap_ci(yte, p, fns, n_boot, bl["seed"]),
            "precision_at_k": {k: precision_at_k(yte, p, k) for k in ks},
            "recall_at_k": {k: recall_at_k(yte, p, k) for k in ks},
            "operating_point": {
                "k": op_k,
                "precision": precision_at_k(yte, p, op_k),
                "recall": recall_at_k(yte, p, op_k),
            },
            "reliability": reliability_curve(yte, p, ece_bins),
        }

    le_test = {}
    for frac in bl["task_b"]["label_efficiency_fracs"]:
        aps = []
        n_seeds = 1 if frac == 1.0 else bl["task_b"]["label_efficiency_seeds"]
        for s in range(n_seeds):
            m = joblib.load(d / f"le_{frac}_s{s}.joblib")
            aps.append(float(average_precision_score(yte, m.predict_proba(Xte)[:, 1])))
        le_test[str(frac)] = aps
    tuning = json.loads((models_dir / "tuning_summary.json").read_text())
    results_b["label_efficiency"] = {
        "val": tuning["task_b"]["label_efficiency"],
        "test_auprc": le_test,
    }

    (out_dir / "metrics_task_a.json").write_text(json.dumps(results_a, indent=1))
    (out_dir / "metrics_task_b.json").write_text(json.dumps(results_b, indent=1))
    report.render_task_a(results_a, bl, out_dir)
    report.render_task_b(results_b, bl, out_dir)
    log.info("final eval frozen -> %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.data_config)
    bl = load_config(args.config)
    models_dir = data_path(cfg, "processed").parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.final_eval:
        final_eval(cfg, bl, models_dir, REPO_ROOT / "baselines")
    else:
        tune_all(cfg, bl, models_dir)


if __name__ == "__main__":
    main()
