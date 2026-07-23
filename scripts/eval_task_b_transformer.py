"""M5 single test pass: Task B transformer vs frozen baselines + the
label-efficiency money chart. Baseline numbers come from
baselines/metrics_task_b.json — never recomputed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.eval.metrics import bootstrap_ci, core_metrics, precision_at_k, recall_at_k

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def _ci_str(ci: dict, name: str, pct: bool = False) -> str:
    m = ci[name]
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(m['point'])} [{f(m['ci_lo'])}, {f(m['ci_hi'])}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_b.yaml")
    parser.add_argument("--baselines-config", default="configs/baselines.yaml")
    parser.add_argument("--report-suffix", default="",
                        help="appended to report/metrics/figure filenames "
                             "(e.g. _hier); default output is unchanged")
    args = parser.parse_args()
    suffix = args.report_suffix

    cfg = load_config(args.config)
    bl = load_config(args.baselines_config)
    ks = bl["task_b"]["precision_at_k"]
    op_k = bl["task_b"]["operating_point_k"]
    n_boot = bl["eval"]["bootstrap_n"]

    ft_dir = REPO_ROOT / cfg["out_dir"]
    sidecar = pl.read_parquet(REPO_ROOT / cfg["data"]["pack_dir"] / "sidecar.parquet")
    pack_meta = json.loads((REPO_ROOT / cfg["data"]["pack_dir"] / "meta.json").read_text())
    baselines = json.loads((REPO_ROOT / "baselines/metrics_task_b.json").read_text())
    hybrid_path = REPO_ROOT / "reports" / "metrics_hybrid.json"
    hybrid = json.loads(hybrid_path.read_text()) if hybrid_path.exists() else None

    sub_val = sidecar.filter(pl.col("split") == "val").sort("Provider")
    sub_test = sidecar.filter(pl.col("split") == "test").sort("Provider")
    y_val = sub_val["label"].to_numpy()
    y_test = sub_test["label"].to_numpy()

    def load_probs(run: str) -> tuple[np.ndarray, np.ndarray]:
        pv = pl.read_parquet(ft_dir / f"{run}_val.parquet").sort("Provider")
        pt = pl.read_parquet(ft_dir / f"{run}_test.parquet").sort("Provider")
        assert pt["Provider"].to_list() == sub_test["Provider"].to_list()
        return pv["p"].to_numpy(), pt["p"].to_numpy()

    # raw probabilities throughout: the frozen M2 Task B protocol is
    # uncalibrated (ranking metrics; SPEC puts recalibration under Task A),
    # and small-val isotonic introduces ranking ties that distort AUPRC
    model_runs = [("full_1.0", "Pretrained, full fine-tune"), ("scratch_1.0", "From scratch")]
    if (ft_dir / "probe_1.0_test.parquet").exists():  # hier runs add a probe arm
        model_runs.append(("probe_1.0", "Probe (frozen encoder)"))

    results: dict = {"models": {}}
    for run, title in model_runs:
        _, p_raw = load_probs(run)
        fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
        for k in ks:
            fns[f"precision_at_{k}"] = (lambda k: lambda y, p: precision_at_k(y, p, k))(k)
        results["models"][run] = {
            "title": title,
            "core": core_metrics(y_test, p_raw),
            "ci": bootstrap_ci(y_test, p_raw, fns, n_boot, cfg["seed"]),
            "operating_point": {
                "k": op_k,
                "precision": precision_at_k(y_test, p_raw, op_k),
                "recall": recall_at_k(y_test, p_raw, op_k),
            },
        }

    def le_curve(prefix: str, at_100: str) -> dict[str, list[float]]:
        curve = {"1.0": [float(average_precision_score(y_test, load_probs(at_100)[1]))]}
        for frac in cfg["label_efficiency"]["fracs"]:
            vals = []
            for s in range(cfg["label_efficiency"]["seeds"]):
                path = ft_dir / f"{prefix}_{frac}_s{s}_test.parquet"
                if path.exists():
                    _, pt = load_probs(f"{prefix}_{frac}_s{s}")
                    vals.append(float(average_precision_score(y_test, pt)))
            if vals:
                curve[str(frac)] = vals
        return curve

    le = le_curve("full", "full_1.0")
    le_scratch = le_curve("scratch", "scratch_1.0")
    results["label_efficiency_test_auprc"] = le
    results["label_efficiency_scratch_test_auprc"] = le_scratch

    xgb = baselines["models"]["xgb"]
    lr = baselines["models"]["lr"]
    xgb_le = baselines["label_efficiency"]["test_auprc"]

    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _money_chart(le, le_scratch, xgb_le,
                 hybrid["task_b_label_efficiency_test_auprc"] if hybrid else None,
                 fig_dir / f"task_b_money_chart{suffix}.png")

    kept_pct = 100 * pack_meta["claims_kept"] / pack_meta["claims_total"]
    prev = y_test.mean()
    if suffix:
        design_lines = [
            "Provider fraud on the Kaggle dataset, same committed provider splits as the",
            "frozen baselines. Hierarchical encoding (Phase 2): ALL of a provider's",
            "claims in 512-token claim-aligned chunks, each encoded independently;",
            "chunk `[CLS]` vectors pooled per provider with gated attention (MIL).",
            "Selection on val; test scored once, here.",
            "",
            f"**No truncation:** {kept_pct:.0f}% of claims retained "
            f"({pack_meta['n_truncated_providers']:,} providers truncated). The v1.0",
            "512-token information asymmetry vs the all-claims baselines is gone;",
            "chunk count still implicitly encodes claim volume (as do the baselines'",
            "volume features; the scratch arm shares the architecture).",
            "",
        ]
    else:
        design_lines = [
            "Provider fraud on the Kaggle dataset, same committed provider splits as the",
            "frozen baselines. Provider = one sequence of its claims (claim = `[VISIT]`",
            "span of DX/PX codes) encoded by the DE-SynPUF-pretrained encoder; `[CLS]`",
            "pooled. Selection/calibration on val; test scored once, here.",
            "",
            "**Stated information asymmetry:** sequences truncate keep-most-recent at 512",
            f"tokens — {pack_meta['n_truncated_providers']:,} high-volume providers truncated; "
            f"{kept_pct:.0f}% of claims retained overall. The XGBoost baseline sees",
            "all-claims aggregate features (including volume), so the baseline comparison",
            "is not information-equal; the pretrained-vs-scratch comparison is (both arms",
            "share the constraint).",
            "",
        ]
    lines = [
        "# Task B — cross-dataset transfer & label efficiency"
        + (" — hierarchical (Phase 2)" if suffix else " (M5)"),
        "",
        *design_lines,
        f"## Headline comparison (test, prevalence {prev:.1%})",
        "",
        "| Model | AUROC | AUPRC |" + "".join(f" P@{k} |" for k in ks),
        "|---" * (3 + len(ks)) + "|",
    ]
    for name, m in (("XGBoost (baseline)", xgb), ("Logistic regression (baseline)", lr)):
        row = f"| {name} | {_ci_str(m['ci'], 'auroc')} | {_ci_str(m['ci'], 'auprc')} |"
        for k in ks:
            row += f" {_ci_str(m['ci'], f'precision_at_{k}', pct=True)} |"
        lines.append(row)
    for run, _ in model_runs:
        m = results["models"][run]
        row = f"| {m['title']} | {_ci_str(m['ci'], 'auroc')} | {_ci_str(m['ci'], 'auprc')} |"
        for k in ks:
            row += f" {_ci_str(m['ci'], f'precision_at_{k}', pct=True)} |"
        lines.append(row)
    if hybrid:
        h = hybrid["task_b"]
        row = (f"| Hybrid: XGBoost + frozen embeddings (M5.5) | {_ci_str(h['ci'], 'auroc')} "
               f"| {_ci_str(h['ci'], 'auprc')} |")
        for k in ks:
            row += f" {_ci_str(h['ci'], f'precision_at_{k}', pct=True)} |"
        lines.append(row)

    op = results["models"]["full_1.0"]["operating_point"]
    lines += [
        "",
        f"Operating point (pretrained transformer, top {op['k']}): precision "
        f"{op['precision']:.1%}, recall {op['recall']:.1%}.",
        "",
        "## Label efficiency (test AUPRC; ± is std over subsample seeds)",
        "",
        "| Labeled providers | XGBoost | Pretrained transformer | Transformer from scratch | Hybrid (XGB+emb) |",
        "|---|---|---|---|---|",
    ]

    def _cell(vals: list[float]) -> str:
        if not vals:
            return "—"
        s = f"{np.mean(vals):.3f}"
        return s + (f" (±{np.std(vals):.3f})" if len(vals) > 1 else "")

    hybrid_le = hybrid["task_b_label_efficiency_test_auprc"] if hybrid else {}
    for frac in ("0.1", "0.25", "1.0"):
        lines.append(
            f"| {float(frac):.0%} | {_cell(xgb_le.get(frac, []))} "
            f"| {_cell(le.get(frac, []))} | {_cell(le_scratch.get(frac, []))} "
            f"| {_cell(hybrid_le.get(frac, []))} |"
        )
    lines += ["", f"![money chart](figures/task_b_money_chart{suffix}.png)", ""]

    (REPO_ROOT / "reports" / f"metrics_task_b_transformer{suffix}.json").write_text(
        json.dumps(results, indent=1, default=float)
    )
    (REPO_ROOT / "reports" / f"task_b_transformer{suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote reports/task_b_transformer{suffix}.md")


def _money_chart(le: dict, le_scratch: dict, xgb_le: dict, hybrid_le: dict | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = ("0.1", "0.25", "1.0")
    x = [10, 25, 100]
    series_list = [
        (le, "tab:orange", "o-", "Pretrained transformer"),
        (le_scratch, "tab:blue", "D--", "Transformer from scratch"),
    ]
    if hybrid_le:
        series_list.append((hybrid_le, "tab:green", "^-.", "Hybrid: XGB + frozen embeddings"))
    for series, color, marker, label in series_list:
        pts = [(xi, series[k]) for xi, k in zip(x, keys) if k in series]
        for xi, vals in pts:
            ax.plot([xi] * len(vals), vals, ".", color=color, ms=6, alpha=0.5)
        ax.plot([p[0] for p in pts], [np.mean(p[1]) for p in pts], marker, color=color, lw=2, label=label)
    xgb_means = [np.mean(xgb_le[k]) for k in keys]
    ax.plot(x, xgb_means, "s:", color="tab:gray", lw=2, label="XGBoost (engineered features)")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 100])
    ax.set_xticklabels(["10%", "25%", "100%"])
    ax.set(xlabel="share of labeled providers used for training", ylabel="test AUPRC",
           title="Provider fraud: label efficiency")
    ax.legend()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
