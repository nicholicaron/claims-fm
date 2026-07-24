"""The clean-split rules from SPEC §3/§5, enforced as tests.

Samples 1-2 are Task A eval; they must never appear in the pretraining
sequences or vocabulary, and Task A observation sequences must not contain
prediction-window (2010) events.
"""

import json

import polars as pl

from claimsfm.config import load_config

_MANIFEST = load_config("configs/data.yaml")["synpuf"]["samples"]
EVAL_SAMPLES = {int(s) for s, role in _MANIFEST.items() if role == "eval_only"}
PRETRAIN_SAMPLES = {int(s) for s, role in _MANIFEST.items() if role == "pretrain"}


def test_pretrain_contains_only_pretrain_samples(processed):
    samples = set(
        pl.scan_parquet(processed / "sequences_pretrain.parquet")
        .select(pl.col("sample_id").unique())
        .collect()["sample_id"]
    )
    assert samples <= PRETRAIN_SAMPLES
    assert not (samples & EVAL_SAMPLES)


def test_no_member_overlap_between_pretrain_and_eval(processed):
    pretrain_ids = set(
        pl.scan_parquet(processed / "sequences_pretrain.parquet")
        .select("DESYNPUF_ID")
        .collect()["DESYNPUF_ID"]
    )
    eval_ids = set(
        pl.scan_parquet(processed / "sequences_eval_only.parquet")
        .select("DESYNPUF_ID")
        .collect()["DESYNPUF_ID"]
    )
    assert not (pretrain_ids & eval_ids)


def test_vocab_built_from_pretrain_roles_only(processed):
    with open(processed / "vocab.json") as f:
        meta = json.load(f)["meta"]
    assert meta["source_roles"] == ["pretrain"]
    # Phase 2 freezes the vocab at the v1.0 build (samples 3-7) while the
    # manifest's pretrain role grew to 3-20; the leakage invariant is subset,
    # not equality: vocab sources are pretrain-role samples and never eval.
    assert set(meta["source_samples"]) <= PRETRAIN_SAMPLES
    assert not (set(meta["source_samples"]) & EVAL_SAMPLES)


def test_windowed_eval_sequences_exclude_prediction_year(processed):
    path = processed / "sequences_eval_only_window_2008_2009.parquet"
    max_date = (
        pl.scan_parquet(path)
        .select(pl.col("dates").explode().max())
        .collect()
        .item()
    )
    assert max_date.year <= 2009
