"""Member event sequences from DE-SynPUF claims tables.

Stage 1 (events): each claim explodes into long-format coded events —
diagnosis (DX_), ICD-9 procedure (PX_), drug (RX_) — dated per claim.
HCPCS columns are intentionally excluded: enormous vocabulary and absent
from the Kaggle fraud schema, so they contribute nothing to transfer.

Stage 2 (sequences): events group per member, sorted by (date, claim type,
claim id, token) for determinism. A visit is a (date, claim type) group.
The parquet keeps parallel lists (tokens/dates/claim types/visit ids) plus
demographics; flattening into model inputs is the tokenizer's job, so one
sequence store serves any window or masking scheme.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from claimsfm.config import data_path

log = logging.getLogger(__name__)

BENE_ID = "DESYNPUF_ID"

DX_COLS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
PX_COLS = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]


def _code_events(
    lf: pl.LazyFrame, cols: list[str], prefix: str, date_expr: pl.Expr, claim_type: str
) -> pl.LazyFrame:
    present = [c for c in cols if c in lf.collect_schema().names()]
    return (
        lf.select(
            pl.col(BENE_ID),
            pl.col("CLM_ID").alias("claim_id"),
            date_expr.alias("event_date"),
            *[pl.col(c) for c in present],
        )
        .unpivot(index=[BENE_ID, "claim_id", "event_date"], on=present, value_name="code")
        .filter(pl.col("code").is_not_null() & pl.col("event_date").is_not_null())
        .select(
            pl.col(BENE_ID),
            pl.col("event_date"),
            pl.col("claim_id"),
            pl.lit(claim_type).alias("claim_type"),
            (pl.lit(prefix) + pl.col("code")).alias("token"),
        )
        .unique()  # same code in multiple positions of one claim collapses
    )


def sample_events(interim_dir: Path) -> pl.LazyFrame:
    """All coded events for one sample, from its typed parquet tables."""
    ip = pl.scan_parquet(interim_dir / "inpatient.parquet")
    op = pl.scan_parquet(interim_dir / "outpatient.parquet")
    rx = pl.scan_parquet(interim_dir / "pde.parquet")

    ip_date = pl.coalesce(pl.col("CLM_ADMSN_DT"), pl.col("CLM_FROM_DT"))
    op_date = pl.col("CLM_FROM_DT")

    return pl.concat(
        [
            _code_events(ip, DX_COLS, "DX_", ip_date, "IP"),
            _code_events(ip, PX_COLS, "PX_", ip_date, "IP"),
            _code_events(op, DX_COLS, "DX_", op_date, "OP"),
            _code_events(op, PX_COLS, "PX_", op_date, "OP"),
            rx.filter(pl.col("PROD_SRVC_ID").is_not_null() & pl.col("SRVC_DT").is_not_null())
            .select(
                pl.col(BENE_ID),
                pl.col("SRVC_DT").alias("event_date"),
                pl.col("PDE_ID").alias("claim_id"),
                pl.lit("RX").alias("claim_type"),
                # NDC truncated to 9-digit labeler+product: full NDC-11 yields
                # ~278k distinct tokens (94% of the raw vocab); package-size
                # granularity is not worth that. Decision logged in DATA.md.
                (pl.lit("RX_") + pl.col("PROD_SRVC_ID").str.slice(0, 9)).alias("token"),
            )
            .unique(),
        ]
    )


def _demographics(interim_dir: Path) -> pl.LazyFrame:
    """One row per member: sex, race, birth year from the earliest year seen."""
    frames = []
    for year in (2008, 2009, 2010):
        p = interim_dir / f"beneficiary_{year}.parquet"
        if p.exists():
            frames.append(
                pl.scan_parquet(p).select(
                    BENE_ID,
                    pl.col("BENE_BIRTH_DT").dt.year().alias("birth_year"),
                    pl.col("BENE_SEX_IDENT_CD").alias("sex"),
                    pl.col("BENE_RACE_CD").alias("race"),
                )
            )
    return pl.concat(frames).unique(subset=[BENE_ID], keep="first", maintain_order=True)


def assemble_sequences(events: pl.LazyFrame, demo: pl.LazyFrame) -> pl.DataFrame:
    ordered = events.sort(BENE_ID, "event_date", "claim_type", "claim_id", "token")
    return (
        ordered.with_columns(
            pl.struct("event_date", "claim_type")
            .rank("dense")
            .over(BENE_ID)
            .cast(pl.UInt32)
            .alias("visit_id")
        )
        .group_by(BENE_ID, maintain_order=True)
        .agg(
            pl.col("token").alias("tokens"),
            pl.col("event_date").alias("dates"),
            pl.col("claim_type").alias("claim_types"),
            pl.col("visit_id").alias("visit_ids"),
            pl.len().alias("n_events"),
            pl.col("visit_id").n_unique().alias("n_visits"),
        )
        .join(demo, on=BENE_ID, how="left", maintain_order="left")
        .collect(engine="streaming")
    )


def build_sequences(
    cfg: dict[str, Any],
    roles: list[str] | None = None,
    window: tuple[int, int] | None = None,
) -> list[Path]:
    """Build per-role sequence parquets; window=(y0, y1) keeps event years y0..y1."""
    interim_root = data_path(cfg, "interim") / "synpuf"
    events_root = data_path(cfg, "interim") / "events"
    out_root = data_path(cfg, "processed")
    out_root.mkdir(parents=True, exist_ok=True)
    events_root.mkdir(parents=True, exist_ok=True)

    by_role: dict[str, list[str]] = {}
    for sample, role in cfg["synpuf"]["samples"].items():
        by_role.setdefault(role, []).append(sample)

    outputs = []
    for role, samples in by_role.items():
        if roles and role not in roles:
            continue
        suffix = f"_window_{window[0]}_{window[1]}" if window else ""
        out_path = out_root / f"sequences_{role}{suffix}.parquet"
        if out_path.exists():
            log.info("skip %s (exists)", out_path)
            outputs.append(out_path)
            continue

        # per-sample sequence caches, then a lazy concat -> sink: peak RAM is
        # one sample's frame, not the whole role's (18 samples would not fit
        # the 8GB build machine). Caches also make the 18-sample build
        # resumable per sample.
        seq_cache = data_path(cfg, "interim") / f"sequences{suffix}"
        seq_cache.mkdir(parents=True, exist_ok=True)
        cache_paths = []
        for sample in samples:
            sample_dir = interim_root / f"sample_{int(sample):02d}"
            events_path = events_root / f"sample_{int(sample):02d}.parquet"
            if not events_path.exists():
                sample_events(sample_dir).sink_parquet(events_path, compression="zstd")
            seq_path = seq_cache / f"sample_{int(sample):02d}.parquet"
            if not seq_path.exists():
                ev = pl.scan_parquet(events_path)
                if window:
                    ev = ev.filter(
                        pl.col("event_date").dt.year().is_between(window[0], window[1])
                    )
                seq = assemble_sequences(ev, _demographics(sample_dir)).with_columns(
                    pl.lit(int(sample), dtype=pl.Int32).alias("sample_id"),
                    pl.lit(role).alias("role"),
                )
                seq.write_parquet(seq_path, compression="zstd")
                log.info("sample %s (%s%s): %d members", sample, role, suffix, len(seq))
                del seq
            cache_paths.append(seq_path)

        pl.concat([pl.scan_parquet(p) for p in cache_paths]).sink_parquet(
            out_path, compression="zstd"
        )
        outputs.append(out_path)
        log.info("wrote %s", out_path)
    return outputs
