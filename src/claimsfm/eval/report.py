"""Render frozen baseline results: markdown tables + figures.

Input is the metrics dicts produced by run_baselines.final_eval — this module
does formatting only, no metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

LABEL_TITLES = {
    "label_ip": "any inpatient admission in 2010",
    "label_cost": "top-decile 2010 total cost",
}


def _ci(entry: dict[str, Any], name: str, pct: bool = False) -> str:
    m = entry["ci"][name]
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    if "ci_lo" in m:
        return f"{f(m['point'])} [{f(m['ci_lo'])}, {f(m['ci_hi'])}]"
    return f(m["point"])


def render_task_a(res: dict[str, Any], bl: dict[str, Any], out_dir: Path) -> None:
    caps = bl["eval"]["lift_capacities"]
    lines = [
        "# Task A baselines — member next-year risk",
        "",
        "Cohort, features, and protocol: see [README.md](README.md). Test set scored once;",
        "95% CIs from member-level bootstrap. Calibrated = val-selected Platt/isotonic.",
        "",
    ]
    for label, entry in res["labels"].items():
        prev = entry["models"]["xgb"]["raw"]["prevalence"]
        lines += [
            f"## {LABEL_TITLES[label]} (test prevalence {prev:.1%})",
            "",
            "| Model | AUROC | AUPRC | Brier | ECE |"
            + "".join(f" Capture@{int(c*100)}% |" for c in caps),
            "|---" * (5 + len(caps)) + "|",
        ]
        for mname in ("lr", "xgb"):
            m = entry["models"][mname]
            row = (
                f"| {mname.upper()} (calibrated) | {_ci(m, 'auroc')} | {_ci(m, 'auprc')} "
                f"| {_ci(m, 'brier')} | {m['calibrated']['ece']:.4f} |"
            )
            for c in caps:
                row += f" {_ci(m, f'capture_at_{int(c*100)}pct', pct=True)} |"
            lines.append(row)
        lines += [
            "",
            f"Uncalibrated XGB: Brier {entry['models']['xgb']['raw']['brier']:.4f}, "
            f"ECE {entry['models']['xgb']['raw']['ece']:.4f} → calibrated "
            f"{entry['models']['xgb']['calibrated']['brier']:.4f} / "
            f"{entry['models']['xgb']['calibrated']['ece']:.4f}.",
            "",
            "### Subgroup slices (XGB calibrated)",
            "",
            "| Field | Group | n | Prevalence | Mean predicted | AUROC | ECE |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["subgroups_xgb_calibrated"]:
            auroc = f"{row['auroc']:.3f}" if "auroc" in row else "—"
            ece_s = f"{row['ece']:.4f}" if "ece" in row else "—"
            lines.append(
                f"| {row['field']} | {row['group']} | {row['n']:,} | {row['prevalence']:.1%} "
                f"| {row['mean_predicted']:.1%} | {auroc} | {ece_s} |"
            )
        lines.append("")
        _fig_reliability(entry, label, out_dir)
        _fig_capture(entry, label, out_dir)
        lines += [
            f"![reliability](figures/task_a_{label}_reliability.png)",
            f"![capture](figures/task_a_{label}_capture.png)",
            "",
        ]
    lines.append(f"Split integrity: `task_a_splits.parquet` sha256 = `{res['labels']['label_ip']['meta']['splits_sha256'][:16]}…`; "
                 f"train-only cost threshold ${res['labels']['label_cost']['meta']['cost_threshold_train']:,.0f}.")
    (out_dir / "results_task_a.md").write_text("\n".join(lines) + "\n")


def _fig_reliability(entry: dict[str, Any], label: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, which in zip(axes, ("reliability_raw", "reliability_calibrated")):
        for mname, color in (("lr", "tab:blue"), ("xgb", "tab:orange")):
            r = entry["models"][mname][which]
            ax.plot(r["mean_predicted"], r["fraction_positive"], "o-", ms=4, color=color, label=mname.upper())
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="mean predicted", title=which.split("_")[1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction positive")
    axes[0].legend()
    fig.suptitle(f"Task A reliability — {LABEL_TITLES[label]}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"task_a_{label}_reliability.png", bbox_inches="tight")
    plt.close(fig)


def _fig_capture(entry: dict[str, Any], label: str, out_dir: Path) -> None:
    c = entry["capture_curve_xgb"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f * 100 for f in c["fracs"]], [v * 100 for v in c["capture"]], "-", lw=2)
    ax.plot([0, 30], [0, 30], "k--", lw=1, label="random")
    ax.set(xlabel="% of members outreached (by predicted risk)", ylabel="% of true positives captured",
           title=f"Capture curve — {LABEL_TITLES[label]}")
    ax.legend()
    fig.savefig(out_dir / "figures" / f"task_a_{label}_capture.png", bbox_inches="tight")
    plt.close(fig)


def render_task_b(res: dict[str, Any], bl: dict[str, Any], out_dir: Path) -> None:
    ks = bl["task_b"]["precision_at_k"]
    prev = res["models"]["xgb"]["core"]["prevalence"]
    ov = res["meta"]["bene_overlap"]
    lines = [
        "# Task B baselines — provider fraud detection",
        "",
        f"Provider-level splits; test = 812 providers, prevalence {prev:.1%}. Test scored once;",
        "95% CIs from provider-level bootstrap.",
        "",
        "| Model | AUROC | AUPRC |" + "".join(f" P@{k} |" for k in ks),
        "|---" * (3 + len(ks)) + "|",
    ]
    for mname in ("lr", "xgb"):
        m = res["models"][mname]
        row = f"| {mname.upper()} | {_ci(m, 'auroc')} | {_ci(m, 'auprc')} |"
        for k in ks:
            row += f" {_ci(m, f'precision_at_{k}', pct=True)} |"
        lines.append(row)
    op = res["models"]["xgb"]["operating_point"]
    lines += [
        "",
        f"**Operating point (XGB):** review the top {op['k']} providers per period "
        f"(~an SIU caseload) → precision {op['precision']:.1%}, recall {op['recall']:.1%}. "
        f"Rationale: precision stays high enough that most referrals are actionable while "
        f"capturing the bulk of flagged-provider fraud; adjust k to actual SIU capacity.",
        "",
        "### Beneficiary overlap caveat (SPEC §5)",
        "",
        f"- {ov['benes_with_multiple_providers']:.1%} of beneficiaries appear under more than one provider.",
        f"- {ov['val_claims_with_train_bene']:.1%} of val-provider claims and "
        f"{ov['test_claims_with_train_bene']:.1%} of test-provider claims involve a beneficiary "
        "also seen under some train provider. Splits are clean at the provider level (the unit "
        "of prediction), but member-level information is not fully disjoint — stated here rather "
        "than pretended away.",
        "",
        "### Label efficiency (XGBoost at 10% / 25% / 100% of labeled providers)",
        "",
        "| Fraction | Val AUPRC (mean over seeds) | Test AUPRC (mean) |",
        "|---|---|---|",
    ]
    le = res["label_efficiency"]
    for frac in map(str, [0.1, 0.25, 1.0]):
        va = le["val"].get(frac) or le["val"].get(frac.rstrip("0")) or {}
        te = le["test_auprc"].get(frac) or le["test_auprc"].get(frac.rstrip("0")) or []
        va_mean = sum(va.get("val_auprc", [])) / max(1, len(va.get("val_auprc", [])))
        te_mean = sum(te) / max(1, len(te))
        lines.append(f"| {float(frac):.0%} | {va_mean:.3f} | {te_mean:.3f} |")
    _fig_label_efficiency(res, out_dir)
    lines += ["", "![label efficiency](figures/task_b_label_efficiency.png)", ""]
    (out_dir / "results_task_b.md").write_text("\n".join(lines) + "\n")


def _fig_label_efficiency(res: dict[str, Any], out_dir: Path) -> None:
    le = res["label_efficiency"]["test_auprc"]
    fracs = sorted(float(f) for f in le)
    means = [sum(le[str(f)]) / len(le[str(f)]) for f in fracs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in fracs:
        ax.plot([f * 100] * len(le[str(f)]), le[str(f)], "o", color="tab:gray", ms=4, alpha=0.6)
    ax.plot([f * 100 for f in fracs], means, "o-", color="tab:orange", lw=2, label="XGB (test AUPRC)")
    ax.set(xlabel="% of labeled providers used for training", ylabel="test AUPRC",
           title="Task B label efficiency — XGBoost baseline")
    ax.set_xscale("log"); ax.set_xticks([10, 25, 100]); ax.set_xticklabels(["10%", "25%", "100%"])
    ax.legend()
    fig.savefig(out_dir / "figures" / "task_b_label_efficiency.png", bbox_inches="tight")
    plt.close(fig)
