"""Post-download integrity checks.

The critical one: sample 1's 2010 beneficiary file had to be recovered from a
Wayback capture because every live CMS URL 404s (the portal's own sample-1
page mislinks sample 20's file). Authenticity is decided by beneficiary-ID
overlap: DE-SynPUF samples are disjoint member populations, so the recovered
file's DESYNPUF_IDs must overlap sample 1's 2008/2009 files heavily and other
samples' files not at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from claimsfm.config import data_path, load_lock

log = logging.getLogger(__name__)

BENE_ID = "DESYNPUF_ID"


def _bene_ids(cfg: dict[str, Any], sample: int, year: int) -> set[str]:
    lock = load_lock()
    key = f"sample_{sample:02d}:beneficiary_{year}"
    entry = lock["synpuf"][key]
    csv = data_path(cfg, "raw") / "synpuf" / f"sample_{sample:02d}" / entry["csv_file"]
    return set(
        pl.scan_csv(csv, infer_schema_length=0).select(BENE_ID).collect()[BENE_ID]
    )


def cross_year_consistency(cfg: dict[str, Any], sample: int) -> dict[str, float]:
    """Jaccard-style overlap of member IDs across years within one sample.

    Members persist across years (minus deaths/new enrollees), so overlap of
    consecutive years should be high. Near-zero means the file belongs to a
    different sample.
    """
    ids = {y: _bene_ids(cfg, sample, y) for y in (2008, 2009, 2010)}
    out = {}
    for a, b in ((2008, 2009), (2009, 2010), (2008, 2010)):
        inter = len(ids[a] & ids[b])
        out[f"{a}_vs_{b}"] = inter / max(1, min(len(ids[a]), len(ids[b])))
    return out


def verify_all(cfg: dict[str, Any], min_overlap: float = 0.5) -> None:
    failures = []
    for sample_str in cfg["synpuf"]["samples"]:
        sample = int(sample_str)
        overlaps = cross_year_consistency(cfg, sample)
        log.info("sample %d cross-year ID overlap: %s", sample, overlaps)
        for pair, frac in overlaps.items():
            if frac < min_overlap:
                failures.append(f"sample {sample} {pair}: overlap {frac:.3f} < {min_overlap}")
    if failures:
        raise RuntimeError(
            "beneficiary ID consistency failures (possible wrong-sample file):\n"
            + "\n".join(failures)
        )
