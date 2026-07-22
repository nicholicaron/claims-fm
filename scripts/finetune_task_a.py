"""Run Task A fine-tunes: all {probe, full, scratch} x {label_ip, label_cost}
by default, or a single --mode/--label pair.

Usage (on the GPU instance):
  PYTHONPATH=src python scripts/finetune_task_a.py --config configs/finetune_a.yaml
"""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.train import run_finetune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_a.yaml")
    parser.add_argument("--mode", default=None, choices=["probe", "full", "scratch"])
    parser.add_argument("--label", default=None, choices=["label_ip", "label_cost"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    modes = [args.mode] if args.mode else list(cfg["modes"])
    labels = [args.label] if args.label else list(cfg["labels"])
    for label in labels:
        for mode in modes:
            run_finetune(cfg, mode, label, REPO_ROOT)
    print("all fine-tune runs complete")


if __name__ == "__main__":
    main()
