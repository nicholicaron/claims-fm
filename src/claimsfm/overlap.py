"""The M1 transfer gate: Kaggle<->DE-SynPUF vocabulary overlap.

Gate metric (SPEC §3/§7 M1): share of Kaggle diagnosis-code *occurrences*
covered by the pretraining vocabulary — the occurrence-weighted view is what
the encoder actually sees at fine-tune time. Type-level coverage, procedure
coverage, and a coverage-vs-floor sweep are reported alongside so a failure
comes with its diagnosis attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from claimsfm.config import data_path
from claimsfm.etl.kaggle_tables import DX_COLS, PX_COLS, code_occurrences

GATE = 0.60


def _coverage(occ: pl.DataFrame, vocab_tokens: set[str]) -> dict[str, float]:
    total_occ = occ["count"].sum()
    covered = occ.filter(pl.col("token").is_in(list(vocab_tokens)))
    return {
        "occurrence_coverage": covered["count"].sum() / max(1, total_occ),
        "type_coverage": covered.height / max(1, occ.height),
        "n_types": occ.height,
        "n_occurrences": int(total_occ),
    }


def run_overlap(cfg: dict[str, Any], kaggle_parquet_dir: Path) -> dict[str, Any]:
    processed = data_path(cfg, "processed")
    with open(processed / "vocab.json") as f:
        vocab = json.load(f)
    vocab_tokens = set(vocab["tokens"])
    counts = pl.read_parquet(processed / "token_counts.parquet")

    dx_occ = code_occurrences(kaggle_parquet_dir, DX_COLS, "DX_")
    px_occ = code_occurrences(kaggle_parquet_dir, PX_COLS, "PX_")

    report: dict[str, Any] = {
        "dx": _coverage(dx_occ, vocab_tokens),
        "px": _coverage(px_occ, vocab_tokens),
        "vocab_size": len(vocab_tokens),
        "vocab_min_count": vocab["meta"]["min_count"],
    }

    # Coverage if the floor had been set differently — the first lever if the
    # gate fails, computable without rebuilding anything.
    sweep = {}
    for mc in [1] + list(cfg["etl"]["vocab_min_count_sweep"]):
        toks = set(counts.filter(pl.col("count") >= mc)["token"])
        sweep[mc] = _coverage(dx_occ, toks)["occurrence_coverage"]
    report["dx_occurrence_coverage_by_floor"] = sweep
    report["gate"] = GATE
    report["gate_passed"] = report["dx"]["occurrence_coverage"] >= GATE
    return report


def write_report(report: dict[str, Any], out_path: Path) -> None:
    dx, px = report["dx"], report["px"]
    lines = [
        "# Kaggle ↔ DE-SynPUF vocabulary overlap",
        "",
        f"Gate: ≥ {report['gate']:.0%} of Kaggle dx-code occurrences covered by the "
        f"pretraining vocab (size {report['vocab_size']}, floor {report['vocab_min_count']}).",
        "",
        f"**Result: {'PASS' if report['gate_passed'] else 'FAIL'} — "
        f"{dx['occurrence_coverage']:.1%} dx occurrence coverage**",
        "",
        "| Metric | Diagnosis | Procedure |",
        "|---|---|---|",
        f"| Occurrence-weighted coverage | {dx['occurrence_coverage']:.1%} | {px['occurrence_coverage']:.1%} |",
        f"| Unique-type coverage | {dx['type_coverage']:.1%} | {px['type_coverage']:.1%} |",
        f"| Kaggle distinct codes | {dx['n_types']:,} | {px['n_types']:,} |",
        f"| Kaggle code occurrences | {dx['n_occurrences']:,} | {px['n_occurrences']:,} |",
        "",
        "Dx occurrence coverage at alternative vocab floors:",
        "",
        "| min_count | coverage |",
        "|---|---|",
    ]
    for mc, cov in report["dx_occurrence_coverage_by_floor"].items():
        lines.append(f"| {mc} | {cov:.1%} |")
    out_path.write_text("\n".join(lines) + "\n")
