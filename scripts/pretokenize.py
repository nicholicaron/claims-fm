"""Pretokenize the pretraining corpus into the packed array format.

Usage:
  python scripts/pretokenize.py --config configs/pretrain.yaml
  python scripts/pretokenize.py --config configs/pretrain_smoke.yaml --limit 5000
"""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.pretrain.data import pretokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--limit", type=int, default=None, help="member cap (smoke packs)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    meta = pretokenize(
        sequences_path=REPO_ROOT / "data/processed/sequences_pretrain.parquet",
        vocab_path=REPO_ROOT / "data/processed/vocab.json",
        out_dir=REPO_ROOT / cfg["data"]["pack_dir"],
        max_len=cfg["data"]["max_len"],
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["seed"],
        limit_members=args.limit,
    )
    print(meta)


if __name__ == "__main__":
    main()
