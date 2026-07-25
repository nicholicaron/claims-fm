"""Registered P1 scoring: masked-prediction quality of a checkpoint on the
v1.0 5-sample pack's val split (identical members val in every pack — the
prereg's common yardstick across scaling cells).

  PYTHONPATH=src python scripts/eval_masked_val.py <checkpoint> <pretrain_config> [pack_dir]
"""

import json
import sys

import torch

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.model.encoder import ClaimsEncoder, EncoderConfig
from claimsfm.pretrain.data import LengthBucketBatches, Pack, kind_lut
from claimsfm.pretrain.train import evaluate, pick_device


def main() -> None:
    ckpt_path, cfg_path = sys.argv[1], sys.argv[2]
    pack_dir = sys.argv[3] if len(sys.argv) > 3 else "data/processed/pretrain_pack"
    cfg = load_config(cfg_path)
    device = pick_device()

    pack = Pack(REPO_ROOT / pack_dir)
    model = ClaimsEncoder(EncoderConfig.from_dict(pack.meta["vocab_size"], cfg["model"])).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    lut = kind_lut(REPO_ROOT / "data/processed/vocab.json")

    batches = LengthBucketBatches(
        pack.lengths, pack.indices("val"), cfg["train"]["tokens_per_batch"], cfg["seed"]
    )
    with torch.no_grad():
        out = evaluate(model, pack, batches, cfg, device, lut)
    print(json.dumps({"checkpoint": ckpt_path, "pack": pack_dir, **out}))


if __name__ == "__main__":
    main()
