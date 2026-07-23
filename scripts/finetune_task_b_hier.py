"""Run the Phase 2 hierarchical Task B fine-tunes: full + scratch + probe at
100% labels, plus the label-efficiency grid (registered in
reports/scaling_prereg.md — grid structure mirrors v1.0).

Usage:
  PYTHONPATH=src python scripts/finetune_task_b_hier.py --config configs/finetune_b_hier.yaml
  ... --scratch-le   # from-scratch label-efficiency arm only
  ... --smoke        # one short full-mode run (use the smoke config)
"""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.hier_train import run_finetune_hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_b_hier.yaml")
    parser.add_argument("--scratch-le", action="store_true",
                        help="run only the from-scratch label-efficiency arm")
    parser.add_argument("--smoke", action="store_true",
                        help="one short full-mode run (pair with the smoke config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.smoke:
        runs = [("full", 0.1, 0, "smoke_full")]
    elif args.scratch_le:
        runs = []
        for frac in cfg["label_efficiency"]["fracs"]:
            for s in range(cfg["label_efficiency"]["seeds"]):
                runs.append(("scratch", frac, s, f"scratch_{frac}_s{s}"))
    else:
        runs = [("full", 1.0, 0, "full_1.0"), ("scratch", 1.0, 0, "scratch_1.0"),
                ("probe", 1.0, 0, "probe_1.0")]
        for frac in cfg["label_efficiency"]["fracs"]:
            for s in range(cfg["label_efficiency"]["seeds"]):
                runs.append(("full", frac, s, f"full_{frac}_s{s}"))

    for mode, frac, seed, name in runs:
        run_finetune_hier(
            cfg, mode, "label", REPO_ROOT,
            train_frac=frac, frac_seed=seed, run_name=name,
        )
    print("all hierarchical task B runs complete")


if __name__ == "__main__":
    main()
