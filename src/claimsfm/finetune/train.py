"""Fine-tuning loop for Task A: one (mode, label) run at a time.

Selection on val AUPRC only; test probabilities are written alongside but
never evaluated here — the single test pass happens in the local eval script.
Runs are minutes long, so crash recovery is rerun-the-mode (documented in the
runbook), not mid-run resume.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from claimsfm.finetune.model import build_model, param_groups
from claimsfm.pretrain.data import LengthBucketBatches, Pack, collate

log = logging.getLogger(__name__)


@torch.no_grad()
def score_split(model, pack, indices, device, tokens_per_batch, seed) -> np.ndarray:
    """Probabilities for pack members at `indices`, returned in indices order."""
    model.eval()
    batches = LengthBucketBatches(pack.lengths, indices, tokens_per_batch, seed)
    probs = np.zeros(len(pack.lengths), dtype=np.float64)
    for b in batches.batches:
        batch = {k: v.to(device) for k, v in collate(pack, b).items()}
        p = torch.sigmoid(model(batch)).float().cpu().numpy()
        probs[b] = p
    return probs[indices]


def run_finetune(
    cfg: dict, mode: str, label: str, repo_root: Path, device: torch.device | None = None
) -> Path:
    torch.manual_seed(cfg["seed"])
    device = device or (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    use_bf16 = device.type == "cuda"

    pack = Pack(repo_root / cfg["data"]["pack_dir"])
    sidecar = pl.read_parquet(repo_root / cfg["data"]["pack_dir"] / "sidecar.parquet")
    y = sidecar[label].to_numpy().astype(np.float32)

    tr_idx = pack.indices("train")
    va_idx = pack.indices("val")
    te_idx = pack.indices("test")

    model = build_model(
        mode, cfg["model"], pack.meta["vocab_size"],
        repo_root / cfg["checkpoint"], cfg["train"]["head_dropout"],
    ).to(device)
    lrs = cfg["modes"][mode]
    opt = torch.optim.AdamW(param_groups(model, lrs["head_lr"], lrs["encoder_lr"]), weight_decay=0.01)

    tpb = cfg["train"]["tokens_per_batch"]
    train_batches = LengthBucketBatches(pack.lengths, tr_idx, tpb, cfg["seed"])
    best_ap, best_state, bad = -1.0, None, 0
    t0 = time.time()

    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        if mode == "probe":
            model.encoder.eval()  # frozen encoder: no dropout noise in features
        for k, bi in enumerate(train_batches.epoch_order(epoch)):
            b = train_batches.batches[bi]
            batch = {kk: v.to(device) for kk, v in collate(pack, b).items()}
            target = torch.from_numpy(y[b]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            if (k + 1) % cfg["train"]["log_every_steps"] == 0:
                log.info("%s/%s epoch %d step %d loss %.4f", mode, label, epoch, k + 1, loss.item())

        val_p = score_split(model, pack, va_idx, device, tpb, cfg["seed"])
        ap = float(average_precision_score(y[va_idx], val_p))
        log.info("%s/%s epoch %d val AUPRC %.4f (%.0fs)", mode, label, epoch, ap, time.time() - t0)
        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                log.info("%s/%s early stop after epoch %d", mode, label, epoch)
                break

    model.load_state_dict(best_state)
    out_dir = repo_root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idx in (("val", va_idx), ("test", te_idx)):
        p = score_split(model, pack, idx, device, tpb, cfg["seed"])
        pl.DataFrame(
            {"DESYNPUF_ID": sidecar["DESYNPUF_ID"].gather(idx), "p": p}
        ).write_parquet(out_dir / f"{mode}_{label}_{split_name}.parquet")

    torch.save({"model": best_state, "mode": mode, "label": label, "config": cfg},
               out_dir / f"{mode}_{label}.pt")
    (out_dir / f"{mode}_{label}_meta.json").write_text(
        json.dumps({"mode": mode, "label": label, "best_val_auprc": best_ap,
                    "seconds": round(time.time() - t0)}, indent=1)
    )
    return out_dir
