"""Build the Task B dataset: provider features, splits, bene-overlap stats.

Outputs (data/processed/):
  task_b_features.parquet — one row per labeled provider: label, split, features
  task_b_splits.parquet   — (Provider, split); the M5 contract
  task_b_meta.json        — split hash/counts, bene-overlap stats, feature list
"""

import argparse
import json
import logging

import polars as pl

from claimsfm.config import REPO_ROOT, data_path, load_config
from claimsfm.tasks.provider_b import (
    assign_splits,
    bene_overlap_stats,
    load_bene,
    load_claims,
    provider_features,
    splits_hash,
    _find,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.data_config)
    bl = load_config(args.config)
    kaggle_dir = REPO_ROOT / cfg["paths"]["interim"] / "kaggle_fraud"

    labels = pl.read_parquet(_find(kaggle_dir, "train-*.parquet"))
    claims = load_claims(kaggle_dir)
    bene = load_bene(kaggle_dir)

    labeled = assign_splits(labels, bl)
    feats = provider_features(claims, bene)
    out = labeled.join(feats, on="Provider", how="left")
    assert out["n_claims"].null_count() == 0, "labeled provider with no claims"

    overlap = bene_overlap_stats(claims, labeled)
    logging.info("bene overlap stats: %s", overlap)

    out_dir = data_path(cfg, "processed")
    out.write_parquet(out_dir / "task_b_features.parquet", compression="zstd")
    labeled.select("Provider", "split").sort("Provider").write_parquet(
        out_dir / "task_b_splits.parquet", compression="zstd"
    )

    stats = (
        out.group_by("split")
        .agg(pl.len().alias("n"), pl.col("label").mean().alias("prev_fraud"))
        .sort("split")
    )
    logging.info("split stats:\n%s", stats)

    feature_cols = [c for c in out.columns if c not in {"Provider", "PotentialFraud", "label", "split"}]
    meta = {
        "splits_sha256": splits_hash(labeled),
        "split_stats": stats.to_dicts(),
        "bene_overlap": overlap,
        "seed": bl["seed"],
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }
    with open(out_dir / "task_b_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"task B: {len(out)} providers, {len(feature_cols)} features")


if __name__ == "__main__":
    main()
