"""Kaggle Healthcare Provider Fraud CSVs -> typed parquet + code counts.

The Kaggle data follows the DE-SynPUF schema (ICD-9, dot-less codes); codes
are normalized identically to the SynPUF side (strip, uppercase) so vocab
overlap is measured apples-to-apples.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

DX_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PX_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]


def find_csvs(raw_dir: Path) -> dict[str, list[Path]]:
    """Kaggle CSVs by kind; both Train_* and Test_* claim files count for overlap."""
    groups: dict[str, list[Path]] = {"inpatient": [], "outpatient": [], "beneficiary": [], "labels": []}
    for p in sorted(raw_dir.glob("*.csv")):
        name = p.name.lower()
        if "inpatient" in name:
            groups["inpatient"].append(p)
        elif "outpatient" in name:
            groups["outpatient"].append(p)
        elif "beneficiary" in name:
            groups["beneficiary"].append(p)
        else:
            groups["labels"].append(p)
    return groups


def _normalize(col: str) -> pl.Expr:
    # ClmProcedureCode_* parse as floats ("9904.0") if read naively; all-string
    # scan + suffix strip keeps them aligned with SynPUF's dot-less codes.
    # The CSVs write missing values as the literal string "NA".
    return (
        pl.col(col)
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace(r"\.0$", "")
        .replace(["", "NA"], None)
    )


def build_kaggle_tables(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, paths in find_csvs(raw_dir).items():
        if not paths:
            raise FileNotFoundError(f"no Kaggle {kind} CSVs under {raw_dir}")
        for p in paths:
            out = out_dir / (p.stem.lower() + ".parquet")
            if out.exists():
                continue
            lf = pl.scan_csv(p, infer_schema_length=0)
            code_cols = [c for c in lf.collect_schema().names() if c in DX_COLS + PX_COLS]
            lf = lf.with_columns([_normalize(c) for c in code_cols])
            lf.sink_parquet(out, compression="zstd")
            log.info("wrote %s", out)


def code_occurrences(out_dir: Path, cols: list[str], prefix: str) -> pl.DataFrame:
    """Occurrence counts of (prefixed) codes across all Kaggle claim files."""
    frames = []
    for p in sorted(out_dir.glob("*patient*.parquet")):
        lf = pl.scan_parquet(p)
        present = [c for c in cols if c in lf.collect_schema().names()]
        if not present:
            continue
        frames.append(
            lf.select(present)
            .unpivot(on=present, value_name="code")
            .filter(pl.col("code").is_not_null())
            .select((pl.lit(prefix) + pl.col("code")).alias("token"))
        )
    return (
        pl.concat(frames)
        .group_by("token")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")
    )
