"""Build the pretraining code vocabulary + tokenizer artifact."""

import argparse
import logging

from claimsfm.config import load_config
from claimsfm.vocab import build_vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = build_vocab(load_config(args.config))
    print(f"vocab written to {out}")


if __name__ == "__main__":
    main()
