"""Task A dataset invariants: split hygiene, leakage guards, threshold provenance."""

import datetime as dt
import json

import polars as pl
import pytest

from claimsfm.config import data_path, load_config
from claimsfm.tasks.features_a import WindowViolation, check_window, window_events


@pytest.fixture(scope="module")
def bl():
    return load_config("configs/baselines.yaml")


@pytest.fixture(scope="module")
def task_a(cfg):
    p = data_path(cfg, "processed") / "task_a_features.parquet"
    if not p.exists():
        pytest.skip("task A not built yet (make task-a)")
    return pl.read_parquet(p)


@pytest.fixture(scope="module")
def meta(cfg):
    p = data_path(cfg, "processed") / "task_a_meta.json"
    if not p.exists():
        pytest.skip("task A not built yet")
    return json.loads(p.read_text())


def test_splits_disjoint_and_complete(task_a):
    assert set(task_a["split"].unique()) == {"train", "val", "test"}
    assert task_a["DESYNPUF_ID"].n_unique() == len(task_a)


def test_split_stratification_holds(task_a):
    stats = task_a.group_by("split").agg(
        pl.col("label_ip").mean(), pl.col("label_cost").mean()
    )
    for col in ("label_ip", "label_cost"):
        vals = stats[col].to_list()
        assert max(vals) - min(vals) < 0.01, f"{col} prevalence drift across splits"


def test_cost_threshold_from_train_only(task_a, meta, bl):
    q = bl["task_a"]["cost_label_quantile"]
    train_thresh = task_a.filter(pl.col("split") == "train")["cost_2010"].quantile(q)
    assert train_thresh == pytest.approx(meta["cost_threshold_train"])
    relabeled = (task_a["cost_2010"] >= train_thresh).cast(pl.Int8)
    assert (relabeled == task_a["label_cost"]).all()


def test_waterfall_monotonic(meta, task_a):
    counts = [row["members"] for row in meta["waterfall"]]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == len(task_a)


def test_window_guard_rejects_out_of_window_events():
    leaky = pl.LazyFrame(
        {"event_date": [dt.date(2009, 6, 1), dt.date(2010, 5, 1)], "token": ["DX_1", "DX_2"]}
    )
    with pytest.raises(WindowViolation):
        check_window(leaky, dt.date(2009, 12, 31))


def test_window_events_respects_bound(cfg, bl):
    ev = window_events(cfg, bl)  # raises WindowViolation internally if leaky
    hi = ev.select(pl.col("event_date").max()).collect().item()
    assert hi <= dt.date(2009, 12, 31)


def test_no_2010_signal_in_features(task_a, meta):
    # feature list must not contain anything derived from the prediction year
    for col in meta["feature_cols"]:
        assert "2010" not in col and not col.startswith("meta_"), col
    assert "cost_2010" not in meta["feature_cols"]


def test_features_no_nulls(task_a, meta):
    nulls = task_a.select(
        [pl.col(c).null_count().alias(c) for c in meta["feature_cols"]]
    ).row(0)
    assert sum(nulls) == 0
