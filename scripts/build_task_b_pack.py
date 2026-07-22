"""Build the Task B provider pack for fine-tuning."""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.task_b_data import build_task_b_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_b.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    proc = REPO_ROOT / "data/processed"
    meta = build_task_b_pack(
        kaggle_dir=REPO_ROOT / "data/interim/kaggle_fraud",
        features_path=proc / "task_b_features.parquet",
        meta_path=proc / "task_b_meta.json",
        vocab_path=proc / "vocab.json",
        out_dir=REPO_ROOT / cfg["data"]["pack_dir"],
        max_len=cfg["data"]["max_len"],
    )
    print(meta)


if __name__ == "__main__":
    main()
