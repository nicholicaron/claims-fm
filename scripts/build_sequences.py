"""Member event sequences, optionally windowed (e.g. Task A observation years).

Usage:
  python scripts/build_sequences.py --config configs/data.yaml
  python scripts/build_sequences.py --config configs/data.yaml --window 2008:2009 --roles eval_only
"""

import argparse
import logging

from claimsfm.config import load_config
from claimsfm.etl.sequences import build_sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--roles", nargs="*", default=None, help="restrict to roles, e.g. eval_only")
    parser.add_argument("--window", default=None, help="event-year window, e.g. 2008:2009")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    window = None
    if args.window:
        y0, y1 = args.window.split(":")
        window = (int(y0), int(y1))
    build_sequences(load_config(args.config), roles=args.roles, window=window)


if __name__ == "__main__":
    main()
