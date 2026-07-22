"""Raw DE-SynPUF CSVs -> typed parquet, one file per (sample, table).

Typing is pattern-based rather than a hardcoded schema so a header change in
a re-downloaded file surfaces as a loud diff, not a silent miscast: *_DT
columns become dates, known money/count prefixes become floats, everything
else (IDs, ICD-9/NDC codes) stays string.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import polars as pl

from claimsfm.config import data_path, load_lock

log = logging.getLogger(__name__)

DATE_RE = re.compile(r"(_DT$|_DT_)")
FLOAT_RE = re.compile(
    r"^(MEDREIMB_|BENRES_|PPPYMT_|CLM_PMT_|NCH_|CLM_PASS_THRU_|PTNT_PAY_|TOT_RX_CST_"
    r"|QTY_DSPNSD_|DAYS_SUPLY_|CLM_UTLZTN_|BENE_.*_MONS$|PLAN_CVRG_MOS_NUM$)"
)


def type_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs = []
    for col in lf.collect_schema().names():
        if DATE_RE.search(col):
            exprs.append(pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False))
        elif FLOAT_RE.search(col):
            exprs.append(pl.col(col).cast(pl.Float64, strict=False))
        else:
            exprs.append(
                pl.col(col).str.strip_chars().str.to_uppercase().replace("", None)
            )
    return lf.with_columns(exprs)


def build_tables(cfg: dict[str, Any], delete_csv: bool = False) -> None:
    lock = load_lock()
    raw_root = data_path(cfg, "raw") / "synpuf"
    out_root = data_path(cfg, "interim") / "synpuf"

    for key, entry in sorted(lock.get("synpuf", {}).items()):
        sample_dir, file_type = key.split(":")
        csv_path = raw_root / sample_dir / entry["csv_file"]
        out_dir = out_root / sample_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{file_type}.parquet"

        if out_path.exists():
            log.info("skip %s (parquet exists)", key)
            continue
        if not csv_path.exists():
            raise FileNotFoundError(f"{key}: {csv_path} missing; run `make download`")

        lf = type_columns(pl.scan_csv(csv_path, infer_schema_length=0))
        lf.sink_parquet(out_path, compression="zstd")

        n_out = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        if n_out != entry["rows"]:
            raise RuntimeError(f"{key}: parquet rows {n_out} != lock rows {entry['rows']}")
        log.info("wrote %s (%d rows)", out_path, n_out)

        if delete_csv:
            csv_path.unlink()
