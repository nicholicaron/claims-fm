"""Run the M5 Task B fine-tunes: full + scratch at 100% labels, plus the
label-efficiency curve (pretrained full at 10%/25%, several seeds each).

Usage (GPU instance):
  PYTHONPATH=src python scripts/finetune_task_b.py --config configs/finetune_b.yaml
"""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.train import run_finetune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_b.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    runs = [("full", 1.0, 0, "full_1.0"), ("scratch", 1.0, 0, "scratch_1.0")]
    for frac in cfg["label_efficiency"]["fracs"]:
        for s in range(cfg["label_efficiency"]["seeds"]):
            runs.append(("full", frac, s, f"full_{frac}_s{s}"))

    for mode, frac, seed, name in runs:
        run_finetune(
            cfg, mode, "label", REPO_ROOT,
            train_frac=frac, frac_seed=seed, run_name=name,
        )
    print("all task B runs complete")


if __name__ == "__main__":
    main()
