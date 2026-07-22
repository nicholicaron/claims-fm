"""ETL invariants: date ranges, no duplicate events, event-count
reconciliation between the long-format event store and the sequence store,
cross-year beneficiary ID consistency (which doubles as the authenticity
check on sample 1's Wayback-recovered 2010 file), and determinism."""

import datetime as dt

import polars as pl
import pytest

from claimsfm.config import data_path
from claimsfm.data.verify import cross_year_consistency
from claimsfm.etl.sequences import assemble_sequences, sample_events


def _event_paths(cfg):
    root = data_path(cfg, "interim") / "events"
    paths = sorted(root.glob("sample_*.parquet"))
    if not paths:
        pytest.skip("events not built yet (make sequences)")
    return paths


def test_event_dates_within_study_period(cfg):
    # IP/OP claims that opened in late 2007 and ran into the 2008-2010 window
    # legitimately carry pre-2008 dates (~0.03% of events, verified across
    # samples); drug events are strictly within the window.
    lo, hi = dt.date(2007, 11, 1), dt.date(2010, 12, 31)
    for p in _event_paths(cfg):
        stats = (
            pl.scan_parquet(p)
            .select(
                pl.col("event_date").min().alias("lo"),
                pl.col("event_date").max().alias("hi"),
                (pl.col("event_date") < dt.date(2008, 1, 1)).mean().alias("pre_frac"),
                pl.col("event_date").filter(pl.col("claim_type") == "RX").min().alias("rx_lo"),
            )
            .collect()
        )
        assert stats["lo"][0] >= lo, p.name
        assert stats["hi"][0] <= hi, p.name
        assert stats["pre_frac"][0] < 0.001, p.name
        assert stats["rx_lo"][0] >= dt.date(2008, 1, 1), p.name


def test_no_duplicate_events(cfg):
    for p in _event_paths(cfg):
        n, n_unique = (
            pl.scan_parquet(p)
            .select(
                pl.len().alias("n"),
                pl.struct("DESYNPUF_ID", "claim_id", "claim_type", "token", "event_date")
                .n_unique()
                .alias("u"),
            )
            .collect()
            .row(0)
        )
        assert n == n_unique, p.name


def test_sequence_event_counts_reconcile(cfg, processed):
    seq_counts = (
        pl.concat(
            [
                pl.scan_parquet(processed / "sequences_pretrain.parquet"),
                pl.scan_parquet(processed / "sequences_eval_only.parquet"),
            ]
        )
        .group_by("sample_id")
        .agg(pl.col("n_events").sum())
        .collect()
    )
    for row in seq_counts.iter_rows(named=True):
        events_path = data_path(cfg, "interim") / "events" / f"sample_{row['sample_id']:02d}.parquet"
        n_events = pl.scan_parquet(events_path).select(pl.len()).collect().item()
        assert row["n_events"] == n_events, f"sample {row['sample_id']}"


def test_cross_year_beneficiary_consistency(cfg, lock):
    for sample_str in cfg["synpuf"]["samples"]:
        overlaps = cross_year_consistency(cfg, int(sample_str))
        for pair, frac in overlaps.items():
            assert frac >= 0.5, f"sample {sample_str} {pair}: {frac:.3f}"


def test_sequence_build_deterministic(cfg):
    interim = data_path(cfg, "interim") / "synpuf" / "sample_03"
    if not (interim / "inpatient.parquet").exists():
        pytest.skip("sample 3 tables not built yet")
    members = (
        pl.scan_parquet(interim / "beneficiary_2008.parquet")
        .select("DESYNPUF_ID")
        .head(200)
        .collect()["DESYNPUF_ID"]
    )
    demo = pl.scan_parquet(interim / "beneficiary_2008.parquet").select(
        "DESYNPUF_ID",
        pl.col("BENE_BIRTH_DT").dt.year().alias("birth_year"),
        pl.col("BENE_SEX_IDENT_CD").alias("sex"),
        pl.col("BENE_RACE_CD").alias("race"),
    )
    ev = sample_events(interim).filter(pl.col("DESYNPUF_ID").is_in(list(members)))
    a = assemble_sequences(ev, demo)
    b = assemble_sequences(ev, demo)
    assert a.equals(b)
