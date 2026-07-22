"""Download DE-SynPUF samples per the manifest and verify integrity.

Usage: python scripts/download.py --config configs/data.yaml
"""

import argparse
import logging

from claimsfm.config import load_config
from claimsfm.data.download import download_synpuf
from claimsfm.data.verify import verify_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    download_synpuf(cfg)
    if not args.skip_verify:
        verify_all(cfg)
    print("download + verify complete; see configs/data.lock.yaml")


if __name__ == "__main__":
    main()
