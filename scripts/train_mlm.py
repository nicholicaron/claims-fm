"""Train the claims encoder with masked-code modeling.

Usage:
  python scripts/train_mlm.py --config configs/pretrain.yaml
  python scripts/train_mlm.py --config configs/pretrain.yaml --resume data/checkpoints/pretrain/last.pt
"""

import argparse
import logging
from pathlib import Path

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.pretrain.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    best = train(cfg, REPO_ROOT, resume=args.resume)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
