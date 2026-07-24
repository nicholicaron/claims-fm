"""M4 single test pass: score all frozen fine-tune variants against the
frozen baselines and render reports/task_a_transformer.md.

Everything upstream (training, selection, calibration fitting) used train+val
only; this script is the one place test labels meet predictions. Baseline
numbers are loaded from baselines/metrics_task_a.json — never recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.eval.calibration import fit_best_calibrator, reliability_curve
from claimsfm.eval.metrics import bootstrap_ci, capture_at, core_metrics, subgroup_report
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

MODES = ["probe", "full", "scratch"]
MODE_TITLES = {
    "probe": "Pretrained, frozen (linear probe)",
    "full": "Pretrained, full fine-tune",
    "scratch": "From scratch (identical arch/budget)",
}
LABEL_TITLES = {
    "label_ip": "any inpatient admission in 2010",
    "label_cost": "top-decile 2010 total cost",
}


def _ci_str(ci: dict, name: str, pct: bool = False) -> str:
    m = ci[name]
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(m['point'])} [{f(m['ci_lo'])}, {f(m['ci_hi'])}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_a.yaml")
    parser.add_argument("--baselines-config", default="configs/baselines.yaml")
    parser.add_argument("--modes", default=None,
                        help="comma-separated subset of probe,full,scratch "
                             "(default: all — unchanged v1.0 behavior)")
    parser.add_argument("--report-suffix", default="",
                        help="appended to report/metrics/figure filenames "
                             "(e.g. _17m18s); default output unchanged")
    args = parser.parse_args()
    modes = args.modes.split(",") if args.modes else MODES
    suffix = args.report_suffix

    cfg = load_config(args.config)
    bl = load_config(args.baselines_config)
    caps = bl["eval"]["lift_capacities"]
    ece_bins = bl["eval"]["ece_bins"]
    n_boot = bl["eval"]["bootstrap_n"]

    ft_dir = REPO_ROOT / cfg["out_dir"]
    sidecar = pl.read_parquet(REPO_ROOT / cfg["data"]["pack_dir"] / "sidecar.parquet")
    baselines = json.loads((REPO_ROOT / "baselines/metrics_task_a.json").read_text())

    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    lines = [
        "# Task A — transformer vs baselines" + (f" ({suffix.lstrip('_')})" if suffix else " (M4)"),
        "",
        "Same cohort, same committed splits, same observation-window inputs as the",
        "frozen baselines (tag `v0.2-baselines`). Selection and calibration on val;",
        "test scored once, here. 95% CIs: member-level bootstrap. Prediction head",
        "reads the final-layer `[CLS]` state (design choice; no pooling search).",
        "",
    ]

    for label in cfg["labels"]:
        sub_val = sidecar.filter(pl.col("split") == "val").sort("DESYNPUF_ID")
        sub_test = sidecar.filter(pl.col("split") == "test").sort("DESYNPUF_ID")
        y_val = sub_val[label].to_numpy()
        y_test = sub_test[label].to_numpy()
        prev = y_test.mean()
        entry: dict = {"models": {}}

        for mode in modes:
            pv = pl.read_parquet(ft_dir / f"{mode}_{label}_val.parquet").sort("DESYNPUF_ID")
            pt = pl.read_parquet(ft_dir / f"{mode}_{label}_test.parquet").sort("DESYNPUF_ID")
            assert pv["DESYNPUF_ID"].to_list() == sub_val["DESYNPUF_ID"].to_list()
            assert pt["DESYNPUF_ID"].to_list() == sub_test["DESYNPUF_ID"].to_list()
            p_val, p_raw = pv["p"].to_numpy(), pt["p"].to_numpy()

            cal_name, cal = fit_best_calibrator(p_val, y_val)
            p_cal = np.clip(cal.transform(p_raw), 0, 1)
            meta = json.loads((ft_dir / f"{mode}_{label}_meta.json").read_text())

            fns = {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss}
            for c in caps:
                fns[f"capture_at_{int(c * 100)}pct"] = (lambda c: lambda y, p: capture_at(y, p, c))(c)
            entry["models"][mode] = {
                "val_auprc": meta["best_val_auprc"],
                "calibrator": cal_name,
                "raw": core_metrics(y_test, p_raw, ece_bins),
                "calibrated": core_metrics(y_test, p_cal, ece_bins),
                "ci": bootstrap_ci(y_test, p_cal, fns, n_boot, cfg["seed"]),
                "reliability_raw": reliability_curve(y_test, p_raw, ece_bins),
                "reliability_calibrated": reliability_curve(y_test, p_cal, ece_bins),
                "p_cal": p_cal,  # dropped before json dump
            }

        best_mode = max(modes, key=lambda m: entry["models"][m]["val_auprc"])
        entry["best_mode_by_val"] = best_mode
        best = entry["models"][best_mode]

        age = sub_test["age"].to_numpy()
        bands = bl["task_a"]["age_bands"]
        band_names = np.array(
            [f"{bands[i]}-{bands[i+1]-1}" if i else f"<{bands[1]}" for i in range(len(bands) - 1)]
        )
        groups = {
            "sex": sub_test["meta_sex"].to_numpy(),
            "race": sub_test["meta_race"].to_numpy(),
            "age_band": band_names[np.digitize(age, bands[1:-1])],
        }
        entry["subgroups_best"] = subgroup_report(y_test, best["p_cal"], groups, ece_bins)
        entry["capture_curve_best"] = {
            "fracs": [round(f, 2) for f in np.arange(0.01, 0.31, 0.01)],
            "capture": [capture_at(y_test, best["p_cal"], f) for f in np.arange(0.01, 0.31, 0.01)],
        }

        xgb = baselines["labels"][label]["models"]["xgb"]
        lr = baselines["labels"][label]["models"]["lr"]

        # ---- markdown ----
        lines += [
            f"## {LABEL_TITLES[label]} (test prevalence {prev:.1%})",
            "",
            "| Model | AUROC | AUPRC | Brier | ECE |"
            + "".join(f" Capture@{int(c*100)}% |" for c in caps),
            "|---" * (5 + len(caps)) + "|",
        ]
        for name, m in (("XGBoost (baseline)", xgb), ("Logistic regression (baseline)", lr)):
            row = (
                f"| {name} | {_ci_str(m['ci'], 'auroc')} | {_ci_str(m['ci'], 'auprc')} "
                f"| {_ci_str(m['ci'], 'brier')} | {m['calibrated']['ece']:.4f} |"
            )
            for c in caps:
                row += f" {_ci_str(m['ci'], f'capture_at_{int(c*100)}pct', pct=True)} |"
            lines.append(row)
        for mode in modes:
            m = entry["models"][mode]
            marker = " **←**" if mode == best_mode else ""
            row = (
                f"| {MODE_TITLES[mode]}{marker} | {_ci_str(m['ci'], 'auroc')} | {_ci_str(m['ci'], 'auprc')} "
                f"| {_ci_str(m['ci'], 'brier')} | {m['calibrated']['ece']:.4f} |"
            )
            for c in caps:
                row += f" {_ci_str(m['ci'], f'capture_at_{int(c*100)}pct', pct=True)} |"
            lines.append(row)

        xgb_auroc = xgb["ci"]["auroc"]["point"]
        best_auroc = best["ci"]["auroc"]["point"]
        lines += [""]
        if "full" in modes and "scratch" in modes:
            full_ap = entry["models"]["full"]["ci"]["auprc"]["point"]
            scratch_ap = entry["models"]["scratch"]["ci"]["auprc"]["point"]
            lines.append(
                f"Pretraining transfer: full fine-tune test AUPRC {full_ap:.3f} vs from-scratch "
                f"{scratch_ap:.3f} ({'+' if full_ap >= scratch_ap else ''}{full_ap - scratch_ap:.3f})."
            )
        lines += [
            f"Best transformer ({MODE_TITLES[best_mode]}) AUROC {best_auroc:.3f} vs XGBoost "
            f"{xgb_auroc:.3f}; calibrated with {best['calibrator']} "
            f"(ECE {best['calibrated']['ece']:.4f}).",
            "",
            "### Subgroup slices (best transformer, calibrated)",
            "",
            "| Field | Group | n | Prevalence | Mean predicted | AUROC | ECE |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["subgroups_best"]:
            auroc_s = f"{row['auroc']:.3f}" if "auroc" in row else "—"
            ece_s = f"{row['ece']:.4f}" if "ece" in row else "—"
            lines.append(
                f"| {row['field']} | {row['group']} | {row['n']:,} | {row['prevalence']:.1%} "
                f"| {row['mean_predicted']:.1%} | {auroc_s} | {ece_s} |"
            )

        _fig(label, entry, baselines["labels"][label].get("capture_curve_xgb"), fig_dir, suffix)
        lines += [
            "",
            f"![reliability](figures/task_a_tf_{label}_reliability{suffix}.png)",
            f"![capture](figures/task_a_tf_{label}_capture{suffix}.png)",
            "",
        ]
        for m in entry["models"].values():
            m.pop("p_cal")
        results[label] = entry

    (REPO_ROOT / "reports" / f"metrics_task_a_transformer{suffix}.json").write_text(
        json.dumps(results, indent=1, default=float)
    )
    (REPO_ROOT / "reports" / f"task_a_transformer{suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote reports/task_a_transformer{suffix}.md")


def _fig(label: str, entry: dict, xgb_curve: dict | None, fig_dir: Path, suffix: str = "") -> None:
    best = entry["models"][entry["best_mode_by_val"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, which in zip(axes, ("reliability_raw", "reliability_calibrated")):
        r = best[which]
        ax.plot(r["mean_predicted"], r["fraction_positive"], "o-", ms=4)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="mean predicted", title=which.split("_")[1], xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("fraction positive")
    fig.suptitle(f"Best transformer reliability — {LABEL_TITLES[label]}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"task_a_tf_{label}_reliability{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    c = entry["capture_curve_best"]
    ax.plot([f * 100 for f in c["fracs"]], [v * 100 for v in c["capture"]], lw=2,
            label=MODE_TITLES[entry["best_mode_by_val"]])
    if xgb_curve:
        ax.plot([f * 100 for f in xgb_curve["fracs"]], [v * 100 for v in xgb_curve["capture"]],
                lw=2, ls="--", color="tab:gray", label="XGBoost baseline")
    ax.plot([0, 30], [0, 30], "k:", lw=1, label="random")
    ax.set(xlabel="% of members outreached", ylabel="% of true positives captured",
           title=f"Capture — {LABEL_TITLES[label]}")
    ax.legend()
    fig.savefig(fig_dir / f"task_a_tf_{label}_capture{suffix}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
