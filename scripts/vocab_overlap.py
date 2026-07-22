"""M1 gate: Kaggle<->DE-SynPUF vocab overlap. Exits nonzero on gate failure
so `make m1` stops before any GPU spend gets planned."""

import argparse
import json
import logging
import sys

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.overlap import run_overlap, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    kaggle_dir = REPO_ROOT / cfg["paths"]["interim"] / "kaggle_fraud"
    if not any(kaggle_dir.glob("*.parquet")):
        print(
            "Kaggle parquet missing — download requires ~/.kaggle/kaggle.json, "
            "then: make download && make tables",
            file=sys.stderr,
        )
        sys.exit(2)

    report = run_overlap(cfg, kaggle_dir)
    out = REPO_ROOT / "reports" / "vocab_overlap.md"
    write_report(report, out)
    print(json.dumps({k: v for k, v in report.items() if k != "dx_occurrence_coverage_by_floor"}, indent=2, default=str))
    print(f"report: {out}")
    if not report["gate_passed"]:
        print("OVERLAP GATE FAILED — do not proceed to M3; see report for floor sweep", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
