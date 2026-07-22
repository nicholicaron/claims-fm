"""Task A cohort, labels, and splits (SPEC §5, samples 1-2 only).

Cohort rule (each step is a waterfall row, reported and persisted):
  1. present in all three beneficiary-summary years;
  2. full Part A+B coverage (HI==12 and SMI==12) in both observation years;
  3. zero HMO months in 2008-2010 (HMO months hide claims from FFS files);
  4. alive entering 2010 (deaths during 2010 remain in the cohort — they are
     exactly who care management needs to find).

Labels:
  ip    — any inpatient admission dated in 2010;
  cost  — 2010 total (all nine beneficiary cost fields) at or above the 90th
          percentile computed on the TRAIN split only.

Splits are member-level 70/15/15, stratified on the joint (ip x provisional
cost bucket) key. The provisional bucket (full-cohort q90) is used only to
balance the splits; the committed label threshold comes from train rows.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from claimsfm.config import data_path

log = logging.getLogger(__name__)

BENE_ID = "DESYNPUF_ID"
COST_FIELDS = [
    f"{kind}_{setting}"
    for kind in ("MEDREIMB", "BENRES", "PPPYMT")
    for setting in ("IP", "OP", "CAR")
]


def _bene(cfg: dict[str, Any], sample: int, year: int) -> pl.LazyFrame:
    p = data_path(cfg, "interim") / "synpuf" / f"sample_{sample:02d}" / f"beneficiary_{year}.parquet"
    return pl.scan_parquet(p)


def build_cohort(cfg: dict[str, Any], bl: dict[str, Any]) -> tuple[pl.DataFrame, list[dict]]:
    """Returns (cohort frame with demographics + ip label + 2010 cost, waterfall)."""
    a = bl["task_a"]
    samples = a["eval_samples"]

    frames = {}
    for year in (2008, 2009, 2010):
        frames[year] = pl.concat(
            [
                _bene(cfg, s, year).with_columns(pl.lit(s, dtype=pl.Int32).alias("sample_id"))
                for s in samples
            ]
        )

    waterfall = []
    ids = frames[2008].select(BENE_ID).collect()[BENE_ID]
    waterfall.append({"step": "in 2008 beneficiary file", "members": len(ids)})

    present_all = (
        frames[2008]
        .select(BENE_ID, "sample_id")
        .join(frames[2009].select(BENE_ID), on=BENE_ID, how="inner")
        .join(frames[2010].select(BENE_ID), on=BENE_ID, how="inner")
    )
    n = present_all.select(pl.len()).collect().item()
    waterfall.append({"step": "present all three years", "members": n})

    cov = present_all
    for year in a["cohort"]["require_full_ab_coverage_years"]:
        cov = cov.join(
            frames[year]
            .filter(
                (pl.col("BENE_HI_CVRAGE_TOT_MONS") == 12)
                & (pl.col("BENE_SMI_CVRAGE_TOT_MONS") == 12)
            )
            .select(BENE_ID),
            on=BENE_ID,
            how="inner",
        )
    n = cov.select(pl.len()).collect().item()
    waterfall.append({"step": "full A+B coverage 2008 & 2009", "members": n})

    for year in a["cohort"]["require_zero_hmo_years"]:
        cov = cov.join(
            frames[year].filter(pl.col("BENE_HMO_CVRAGE_TOT_MONS") == 0).select(BENE_ID),
            on=BENE_ID,
            how="inner",
        )
    n = cov.select(pl.len()).collect().item()
    waterfall.append({"step": "zero HMO months 2008-2010", "members": n})

    alive_from = dt.date(a["cohort"]["alive_entering"], 1, 1)
    demo = frames[2008].select(
        BENE_ID,
        pl.col("BENE_BIRTH_DT").alias("birth_date"),
        pl.col("BENE_SEX_IDENT_CD").alias("sex"),
        pl.col("BENE_RACE_CD").alias("race"),
        pl.col("SP_STATE_CODE").alias("state"),
    )
    death = frames[2010].select(BENE_ID, pl.col("BENE_DEATH_DT").alias("death_date"))
    cohort = (
        cov.join(death, on=BENE_ID, how="left")
        .filter(pl.col("death_date").is_null() | (pl.col("death_date") >= alive_from))
        .join(demo, on=BENE_ID, how="left")
        .join(
            frames[2010].select(
                BENE_ID,
                pl.sum_horizontal(COST_FIELDS).alias("cost_2010"),
                pl.col("BENE_ESRD_IND").alias("esrd_2010"),
            ),
            on=BENE_ID,
            how="left",
        )
    )
    n = cohort.select(pl.len()).collect().item()
    waterfall.append({"step": "alive entering 2010", "members": n})

    ip_2010 = pl.concat(
        [
            pl.scan_parquet(
                data_path(cfg, "interim") / "synpuf" / f"sample_{s:02d}" / "inpatient.parquet"
            )
            for s in samples
        ]
    )
    admitted = (
        ip_2010.filter(
            pl.coalesce(pl.col("CLM_ADMSN_DT"), pl.col("CLM_FROM_DT")).dt.year()
            == a["prediction_year"]
        )
        .select(BENE_ID)
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label_ip"))
    )
    out = (
        cohort.join(admitted, on=BENE_ID, how="left")
        .with_columns(pl.col("label_ip").fill_null(0))
        .collect(engine="streaming")
        .sort(BENE_ID)
    )
    return out, waterfall


def assign_splits(cohort: pl.DataFrame, bl: dict[str, Any]) -> pl.DataFrame:
    """70/15/15 member-level splits, stratified on (ip x provisional cost bucket)."""
    seed = bl["seed"]
    fr = bl["splits"]
    q = bl["task_a"]["cost_label_quantile"]

    provisional = (cohort["cost_2010"] >= cohort["cost_2010"].quantile(q)).cast(pl.Int8)
    strat = (cohort["label_ip"] * 2 + provisional).to_numpy()
    idx = np.arange(len(cohort))

    tr, rest = train_test_split(
        idx, test_size=1 - fr["train"], stratify=strat, random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=fr["test"] / (fr["val"] + fr["test"]),
        stratify=strat[rest],
        random_state=seed,
    )
    split = np.empty(len(cohort), dtype=object)
    split[tr], split[val], split[test] = "train", "val", "test"

    out = cohort.with_columns(pl.Series("split", split, dtype=pl.String))

    # committed label threshold: train rows only
    thresh = out.filter(pl.col("split") == "train")["cost_2010"].quantile(q)
    out = out.with_columns(
        (pl.col("cost_2010") >= thresh).cast(pl.Int8).alias("label_cost"),
        pl.lit(thresh).alias("cost_threshold_train"),
    )
    log.info("train-only cost threshold (q%.2f): $%.2f", q, thresh)
    return out


def splits_hash(df: pl.DataFrame) -> str:
    key = "\n".join(f"{i}:{s}" for i, s in df.select(BENE_ID, "split").iter_rows())
    return hashlib.sha256(key.encode()).hexdigest()
