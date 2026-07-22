"""Task B dataset invariants: split hygiene, within-provider feature locality,
overlap stats presence."""

import datetime as dt
import json

import polars as pl
import pytest

from claimsfm.config import data_path
from claimsfm.tasks.provider_b import provider_features


@pytest.fixture(scope="module")
def task_b(cfg):
    p = data_path(cfg, "processed") / "task_b_features.parquet"
    if not p.exists():
        pytest.skip("task B not built yet (make task-b)")
    return pl.read_parquet(p)


@pytest.fixture(scope="module")
def meta_b(cfg):
    p = data_path(cfg, "processed") / "task_b_meta.json"
    if not p.exists():
        pytest.skip("task B not built yet")
    return json.loads(p.read_text())


def test_splits_disjoint_and_stratified(task_b):
    assert task_b["Provider"].n_unique() == len(task_b)
    stats = task_b.group_by("split").agg(pl.col("label").mean())
    vals = stats["label"].to_list()
    assert max(vals) - min(vals) < 0.01


def test_overlap_stats_present(meta_b):
    ov = meta_b["bene_overlap"]
    assert 0 <= ov["benes_with_multiple_providers"] <= 1
    assert "test_claims_with_train_bene" in ov


def _toy_claims(providers: list[str]) -> pl.DataFrame:
    rows = []
    for i, prov in enumerate(providers):
        for j in range(3):
            rows.append(
                {
                    "Provider": prov,
                    "BeneID": f"B{i}_{j}",
                    "ClaimID": f"C{i}_{j}",
                    "start": dt.date(2009, 1 + j, 15),
                    "end": dt.date(2009, 1 + j, 18),
                    "reimb": 100.0 * (i + 1),
                    "deductible": 10.0,
                    "attending": f"PHY{i}",
                    "operating": None,
                    "dx_concat": "4019|25000",
                    "dx_list": ["4019", "25000"],
                    "px_list": [],
                    "is_ip": False,
                    "admit": None,
                    "discharge": None,
                    "drg": None,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("admit").cast(pl.Date),
        pl.col("discharge").cast(pl.Date),
        pl.col("drg").cast(pl.String),
        pl.col("operating").cast(pl.String),
        pl.col("px_list").cast(pl.List(pl.String)),
    )


def _toy_bene(claims: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "BeneID": claims["BeneID"].unique().sort(),
        }
    ).with_columns(
        pl.lit(dt.date(1940, 1, 1)).alias("dob"),
        pl.lit(None, dtype=pl.Date).alias("dod"),
        pl.lit(2, dtype=pl.Int8).alias("chronic_count"),
        pl.lit(0, dtype=pl.Int8).alias("renal"),
        pl.lit(5000.0).alias("annual_reimb"),
    )


def test_provider_features_are_within_provider(cfg):
    """A provider's feature vector must not change when other providers appear."""
    two = _toy_claims(["P1", "P2"])
    three = _toy_claims(["P1", "P2", "P3"])
    f2 = provider_features(two, _toy_bene(two)).filter(pl.col("Provider") == "P1")
    f3 = provider_features(three, _toy_bene(three)).filter(pl.col("Provider") == "P1")
    assert f2.equals(f3)


def test_no_label_derived_features(task_b, meta_b):
    for col in meta_b["feature_cols"]:
        assert "fraud" not in col.lower() and col != "label", col
