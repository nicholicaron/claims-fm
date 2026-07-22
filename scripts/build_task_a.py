"""Build the Task A dataset: cohort waterfall, labels, splits, feature matrix.

Outputs (data/processed/):
  task_a_features.parquet  — one row per cohort member: ids, split, labels, features
  task_a_splits.parquet    — (DESYNPUF_ID, split); the M4 contract
  task_a_meta.json         — waterfall, threshold, split hash/counts, feature list
"""

import argparse
import json
import logging

import polars as pl

from claimsfm.config import data_path, load_config
from claimsfm.tasks.cohort_a import assign_splits, build_cohort, splits_hash
from claimsfm.tasks.features_a import build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.data_config)
    bl = load_config(args.config)

    cohort, waterfall = build_cohort(cfg, bl)
    for row in waterfall:
        logging.info("waterfall | %-35s %8d", row["step"], row["members"])

    cohort = assign_splits(cohort, bl)
    features, feat_meta = build_features(cfg, bl, cohort)

    out_dir = data_path(cfg, "processed")
    features.write_parquet(out_dir / "task_a_features.parquet", compression="zstd")
    splits = cohort.select("DESYNPUF_ID", "split").sort("DESYNPUF_ID")
    splits.write_parquet(out_dir / "task_a_splits.parquet", compression="zstd")

    label_stats = (
        features.group_by("split")
        .agg(
            pl.len().alias("n"),
            pl.col("label_ip").mean().alias("prev_ip"),
            pl.col("label_cost").mean().alias("prev_cost"),
        )
        .sort("split")
    )
    logging.info("split label stats:\n%s", label_stats)

    meta = {
        "waterfall": waterfall,
        "cost_threshold_train": float(cohort["cost_threshold_train"][0]),
        "splits_sha256": splits_hash(cohort),
        "split_stats": label_stats.to_dicts(),
        "seed": bl["seed"],
        **feat_meta,
    }
    with open(out_dir / "task_a_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"task A: {len(features)} members, {feat_meta['n_features']} features")


if __name__ == "__main__":
    main()
