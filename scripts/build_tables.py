"""Raw CSVs -> typed parquet (DE-SynPUF + Kaggle if downloaded)."""

import argparse
import logging

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.etl.synpuf_tables import build_tables
from claimsfm.etl.kaggle_tables import build_kaggle_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--delete-csv", action="store_true", help="remove extracted CSVs after conversion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    build_tables(cfg, delete_csv=args.delete_csv)

    kaggle_raw = REPO_ROOT / cfg["kaggle"]["dest"]
    if any(kaggle_raw.glob("*.csv")):
        build_kaggle_tables(kaggle_raw, REPO_ROOT / cfg["paths"]["interim"] / "kaggle_fraud")
    else:
        logging.warning("no Kaggle CSVs under %s; skipping (needs ~/.kaggle/kaggle.json)", kaggle_raw)


if __name__ == "__main__":
    main()
