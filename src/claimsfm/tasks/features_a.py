"""Task A member feature matrix — strictly observation-window inputs.

Every event-derived block flows through `window_events()`, which filters to
the observation years and then *asserts* the bound, so a mis-wired caller
fails loudly rather than leaking 2010 signal. Beneficiary-summary blocks use
only 2008/2009 files (plus static demographics from 2008).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import polars as pl

from claimsfm.config import data_path

log = logging.getLogger(__name__)

BENE_ID = "DESYNPUF_ID"
SP_COLS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", "SP_DEPRESSN",
    "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA",
]
COST_FIELDS = [
    f"{kind}_{setting}"
    for kind in ("MEDREIMB", "BENRES", "PPPYMT")
    for setting in ("IP", "OP", "CAR")
]


class WindowViolation(RuntimeError):
    pass


def check_window(ev: pl.LazyFrame, bound: dt.date) -> pl.LazyFrame:
    """Raise if any event falls after the observation-window bound."""
    hi = ev.select(pl.col("event_date").max()).collect().item()
    if hi is not None and hi > bound:
        raise WindowViolation(f"event beyond observation window: {hi} > {bound}")
    return ev


def window_events(cfg: dict[str, Any], bl: dict[str, Any]) -> pl.LazyFrame:
    years = bl["task_a"]["observation_years"]
    frames = [
        pl.scan_parquet(data_path(cfg, "interim") / "events" / f"sample_{s:02d}.parquet")
        for s in bl["task_a"]["eval_samples"]
    ]
    ev = pl.concat(frames).filter(pl.col("event_date").dt.year().is_in(years))
    return check_window(ev, dt.date(max(years), 12, 31))


def _bene(cfg: dict[str, Any], sample: int, year: int) -> pl.LazyFrame:
    p = data_path(cfg, "interim") / "synpuf" / f"sample_{sample:02d}" / f"beneficiary_{year}.parquet"
    return pl.scan_parquet(p)


def _bene_all(cfg: dict[str, Any], bl: dict[str, Any], year: int) -> pl.LazyFrame:
    return pl.concat([_bene(cfg, s, year) for s in bl["task_a"]["eval_samples"]])


def demographic_block(cohort: pl.DataFrame, bl: dict[str, Any]) -> pl.DataFrame:
    asof = dt.date(bl["task_a"]["prediction_year"], 1, 1)
    df = cohort.select(
        BENE_ID,
        ((pl.lit(asof) - pl.col("birth_date")).dt.total_days() / 365.25).alias("age"),
        (pl.col("sex") == "2").cast(pl.Int8).alias("sex_female"),
        pl.col("race"),
        pl.col("state"),
        (pl.col("esrd_2010") == "Y").cast(pl.Int8).alias("esrd"),
    )
    df = df.to_dummies(columns=["race", "state"], drop_first=True)
    return df


def chronic_block(cfg: dict[str, Any], bl: dict[str, Any]) -> pl.LazyFrame:
    b09 = _bene_all(cfg, bl, 2009)
    flags = [(pl.col(c) == "1").cast(pl.Int8).alias(f"cc_{c.removeprefix('SP_').lower()}") for c in SP_COLS]
    return b09.select(BENE_ID, *flags).with_columns(
        pl.sum_horizontal([f"cc_{c.removeprefix('SP_').lower()}" for c in SP_COLS]).alias("cc_count")
    )


def cost_block(cfg: dict[str, Any], bl: dict[str, Any]) -> pl.LazyFrame:
    out = None
    for year in bl["task_a"]["observation_years"]:
        yy = str(year)[2:]
        b = _bene_all(cfg, bl, year).select(
            BENE_ID,
            *[pl.col(c).alias(f"cost{yy}_{c.lower()}") for c in COST_FIELDS],
            pl.sum_horizontal(COST_FIELDS).alias(f"cost{yy}_total"),
        )
        out = b if out is None else out.join(b, on=BENE_ID, how="full", coalesce=True)
    return out.with_columns(
        pl.col("cost08_total").log1p().alias("cost08_log_total"),
        pl.col("cost09_total").log1p().alias("cost09_log_total"),
        (pl.col("cost09_total") - pl.col("cost08_total")).alias("cost_trend"),
    )


def utilization_block(ev: pl.LazyFrame, cfg: dict[str, Any], bl: dict[str, Any]) -> pl.LazyFrame:
    year_type_counts = (
        ev.group_by(BENE_ID, pl.col("event_date").dt.year().alias("y"), "claim_type")
        .agg(pl.col("claim_id").n_unique().alias("n"))
        .collect(engine="streaming")
        .pivot(on=["y", "claim_type"], index=BENE_ID, values="n")
        .lazy()
    )
    distincts = ev.group_by(BENE_ID).agg(
        pl.col("token").filter(pl.col("token").str.starts_with("DX_")).n_unique().alias("n_dx_codes"),
        pl.col("token").filter(pl.col("token").str.starts_with("PX_")).n_unique().alias("n_px_codes"),
        pl.col("token").filter(pl.col("token").str.starts_with("RX_")).n_unique().alias("n_rx_codes"),
        pl.col("event_date").n_unique().alias("n_active_days"),
        pl.col("event_date").dt.strftime("%Y-%m").n_unique().alias("n_active_months"),
        pl.len().alias("n_events"),
    )
    ip_days = (
        pl.concat(
            [
                pl.scan_parquet(
                    data_path(cfg, "interim") / "synpuf" / f"sample_{s:02d}" / "inpatient.parquet"
                )
                for s in bl["task_a"]["eval_samples"]
            ]
        )
        .with_columns(pl.coalesce(pl.col("CLM_ADMSN_DT"), pl.col("CLM_FROM_DT")).alias("adm"))
        .filter(pl.col("adm").dt.year().is_in(bl["task_a"]["observation_years"]))
        .group_by(BENE_ID)
        .agg(pl.col("CLM_UTLZTN_DAY_CNT").sum().alias("ip_days"))
    )
    return year_type_counts.join(distincts, on=BENE_ID, how="full", coalesce=True).join(
        ip_days, on=BENE_ID, how="left"
    )


def recency_block(ev: pl.LazyFrame, bl: dict[str, Any]) -> pl.LazyFrame:
    anchor = dt.date(max(bl["task_a"]["observation_years"]), 12, 31)
    windows = bl["task_a"]["recency_windows_days"]
    aggs = [
        (pl.lit(anchor) - pl.col("event_date").max()).dt.total_days().alias("days_since_last_event"),
        (pl.lit(anchor) - pl.col("event_date").filter(pl.col("claim_type") == "IP").max())
        .dt.total_days()
        .alias("days_since_last_ip"),
    ]
    for w in windows:
        start = anchor - dt.timedelta(days=w)
        aggs.append((pl.col("event_date") > start).sum().alias(f"events_last_{w}d"))
    return ev.group_by(BENE_ID).agg(aggs)


def dx_category_block(
    ev: pl.LazyFrame, train_ids: pl.Series, top_n: int
) -> tuple[pl.LazyFrame, list[str]]:
    """Counts over top-N 3-digit dx categories, ranked by TRAIN prevalence only."""
    dx = ev.filter(pl.col("token").str.starts_with("DX_")).with_columns(
        pl.when(pl.col("token").str.starts_with("DX_E"))
        .then(pl.col("token").str.slice(3, 4))
        .otherwise(pl.col("token").str.slice(3, 3))
        .alias("cat")
    )
    top = (
        dx.filter(pl.col(BENE_ID).is_in(train_ids.implode()))
        .group_by("cat")
        .agg(pl.col(BENE_ID).n_unique().alias("n_members"))
        .sort("n_members", "cat", descending=[True, False])
        .head(top_n)
        .collect(engine="streaming")["cat"]
        .to_list()
    )
    block = (
        dx.filter(pl.col("cat").is_in(top))
        .group_by(BENE_ID, "cat")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
        .pivot(on="cat", index=BENE_ID, values="n")
        .rename({c: f"dxcat_{c}" for c in top})
        .lazy()
    )
    return block, top


def build_features(
    cfg: dict[str, Any], bl: dict[str, Any], cohort_with_splits: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    ev_all = window_events(cfg, bl)
    cohort_ids = cohort_with_splits[BENE_ID]
    ev = ev_all.filter(pl.col(BENE_ID).is_in(cohort_ids.implode()))
    train_ids = cohort_with_splits.filter(pl.col("split") == "train")[BENE_ID]

    dxcat, top_cats = dx_category_block(ev, train_ids, bl["task_a"]["dx_category_top_n"])

    base = cohort_with_splits.select(
        BENE_ID, "sample_id", "split", "label_ip", "label_cost", "cost_2010",
        # raw demographics as meta_ (not features) for subgroup slice reporting
        pl.col("sex").alias("meta_sex"),
        pl.col("race").alias("meta_race"),
    )
    out = (
        base.lazy()
        .join(demographic_block(cohort_with_splits, bl).lazy(), on=BENE_ID, how="left")
        .join(chronic_block(cfg, bl), on=BENE_ID, how="left")
        .join(cost_block(cfg, bl), on=BENE_ID, how="left")
        .join(utilization_block(ev, cfg, bl), on=BENE_ID, how="left")
        .join(recency_block(ev, bl), on=BENE_ID, how="left")
        .join(dxcat, on=BENE_ID, how="left")
        .collect(engine="streaming")
    )

    meta_cols = {BENE_ID, "sample_id", "split", "label_ip", "label_cost", "cost_2010"}
    feature_cols = [c for c in out.columns if c not in meta_cols and not c.startswith("meta_")]
    # no-claims members: zero counts; recency gets an "over two years ago" sentinel
    sentinel = 365 * 2
    out = out.with_columns(
        [
            pl.col(c).fill_null(sentinel)
            if c.startswith("days_since_")
            else pl.col(c).fill_null(0)
            for c in feature_cols
        ]
    )
    meta = {
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "top_dx_categories_train_ranked": top_cats,
    }
    return out, meta
