"""Task A fine-tuning pack: cohort-aligned sequences + label sidecar.

The pack reuses the M3 array format; the sidecar parquet carries labels,
splits, and slice demographics in *identical member order* (both sorted by
DESYNPUF_ID). Split assignments come from the committed M2 contract
(`task_a_splits.parquet`) and are verified against its recorded sha256.
Cohort members with no observation-window claims pack as `[CLS]`-only rows —
healthy members are part of the population XGBoost scored, so the
transformer scores them too.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from claimsfm.pretrain.data import pack_frame

log = logging.getLogger(__name__)

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SIDECAR_COLS = ["DESYNPUF_ID", "split", "label_ip", "label_cost", "age", "meta_sex", "meta_race"]


def build_task_a_pack(
    features_path: Path,
    sequences_path: Path,
    vocab_path: Path,
    meta_path: Path,
    out_dir: Path,
    max_len: int,
) -> dict:
    with open(vocab_path) as f:
        vocab = json.load(f)
    task_meta = json.loads(meta_path.read_text())

    cohort = pl.read_parquet(features_path, columns=SIDECAR_COLS).sort("DESYNPUF_ID")

    # split integrity: reproduce the committed M2 hash exactly
    key = "\n".join(f"{i}:{s}" for i, s in cohort.select("DESYNPUF_ID", "split").iter_rows())
    got = hashlib.sha256(key.encode()).hexdigest()
    if got != task_meta["splits_sha256"]:
        raise RuntimeError(
            f"split hash mismatch: pack {got[:16]} vs contract {task_meta['splits_sha256'][:16]}"
        )

    seqs = pl.read_parquet(
        sequences_path,
        columns=["DESYNPUF_ID", "tokens", "dates", "claim_types", "visit_ids", "birth_year"],
    )
    df = cohort.select("DESYNPUF_ID").join(seqs, on="DESYNPUF_ID", how="left").sort("DESYNPUF_ID")
    assert df["DESYNPUF_ID"].to_list() == cohort["DESYNPUF_ID"].to_list()

    stats = pack_frame(df, vocab["tokens"], out_dir, max_len)
    split_arr = np.array([SPLIT_CODE[s] for s in cohort["split"]], dtype=np.uint8)
    np.save(out_dir / "split.npy", split_arr)
    cohort.write_parquet(out_dir / "sidecar.parquet", compression="zstd")

    n_empty = int(df["tokens"].is_null().sum())
    meta = {
        **stats,
        "n_empty_history": n_empty,
        "splits_sha256": got,
        "vocab_size": len(vocab["tokens"]),
        "vocab_counts_hash": vocab["meta"]["counts_hash"],
        "source": str(sequences_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(
        "task A pack: %d members (%d empty-history), %d positions -> %s",
        stats["n_members"], n_empty, stats["n_positions"], out_dir,
    )
    return meta
