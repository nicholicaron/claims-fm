"""Extract frozen pretrained [CLS] embeddings for every member/provider in a
fine-tuning pack. Inference-only — runs locally (MPS/CPU), no GPU spend.

Usage:
  python scripts/extract_embeddings.py --pack data/processed/task_a_pack
  python scripts/extract_embeddings.py --pack data/processed/task_b_pack
"""

import argparse
import logging
import time

import numpy as np
import polars as pl
import torch

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.model.encoder import ClaimsEncoder, EncoderConfig
from claimsfm.pretrain.data import LengthBucketBatches, Pack, collate

log = logging.getLogger(__name__)


@torch.no_grad()
def extract(pack: Pack, encoder: ClaimsEncoder, device: torch.device, tokens_per_batch: int) -> np.ndarray:
    encoder.eval()
    d = encoder.cfg.d_model
    out = np.zeros((len(pack.lengths), d), dtype=np.float32)
    all_idx = np.arange(len(pack.lengths))
    batches = LengthBucketBatches(pack.lengths, all_idx, tokens_per_batch, seed=0)
    t0 = time.time()
    for i, b in enumerate(batches.batches):
        batch = {k: v.to(device) for k, v in collate(pack, b).items()}
        hidden = encoder(batch["input_ids"], batch["claim_type_ids"], batch["age_years"],
                         batch["month_idx"], batch["visit_ids"])
        out[b] = hidden[:, 0].float().cpu().numpy()
        if (i + 1) % 50 == 0:
            log.info("batch %d/%d (%.0fs)", i + 1, len(batches.batches), time.time() - t0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--checkpoint", default="data/checkpoints/pretrain/best.pt")
    parser.add_argument("--tokens-per-batch", type=int, default=8192)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

    pretrain_cfg = load_config("configs/pretrain.yaml")
    pack = Pack(REPO_ROOT / args.pack)
    state = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    encoder = ClaimsEncoder(
        EncoderConfig.from_dict(pack.meta["vocab_size"], pretrain_cfg["model"])
    ).to(device)
    encoder.load_state_dict(state["model"])
    assert state["vocab_counts_hash"] == pack.meta["vocab_counts_hash"], "vocab mismatch"

    emb = extract(pack, encoder, device, args.tokens_per_batch)
    sidecar = pl.read_parquet(REPO_ROOT / args.pack / "sidecar.parquet")
    id_col = sidecar.columns[0]
    df = pl.DataFrame({id_col: sidecar[id_col]}).with_columns(
        pl.Series("emb", list(emb))
    )
    out_path = REPO_ROOT / args.pack / "cls_embeddings.parquet"
    df.write_parquet(out_path, compression="zstd")
    print(f"wrote {out_path} ({emb.shape})")


if __name__ == "__main__":
    main()
