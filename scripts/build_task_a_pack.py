"""Build the Task A fine-tuning pack (cohort sequences + label sidecar)."""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.data import build_task_a_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_a.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    proc = REPO_ROOT / "data/processed"
    meta = build_task_a_pack(
        features_path=proc / "task_a_features.parquet",
        sequences_path=proc / "sequences_eval_only_window_2008_2009.parquet",
        vocab_path=proc / "vocab.json",
        meta_path=proc / "task_a_meta.json",
        out_dir=REPO_ROOT / cfg["data"]["pack_dir"],
        max_len=cfg["data"]["max_len"],
    )
    print(meta)


if __name__ == "__main__":
    main()
