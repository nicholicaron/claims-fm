"""Hierarchical Task B fine-tuning loop (Phase 2).

Mirrors finetune/train.py's contract exactly — val-AUPRC selection, patience,
test probs written but never scored here, artifacts {run}_val/_test.parquet
(Provider, p) + {run}.pt + {run}_meta.json — so scripts/eval_task_b_transformer.py
and the label-efficiency machinery work unchanged.

Training batches whole providers with a random-K chunk cap (collate_hier);
evaluation is two-phase: (1) encode ALL chunks of the split in fixed-shape
micro-batches under no_grad — shapes identical to training's L buckets, so no
new kernel shapes on MPS and no VRAM spike from monster providers — then
(2) pool + head per provider in chunk-count-sorted groups.
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

from claimsfm.finetune.hier_data import HierPack, ProviderBatches, collate_chunks, collate_hier
from claimsfm.finetune.hier_model import build_hier_model, hier_param_groups
from claimsfm.finetune.train import stratified_subsample
from claimsfm.pretrain.data import LengthBucketBatches

log = logging.getLogger(__name__)


@torch.no_grad()
def score_split_hier(model, hpack, provider_indices, device, tokens_per_batch, seed) -> np.ndarray:
    """Probabilities for providers at `indices` (all chunks, no cap)."""
    model.eval()
    d_model = model.encoder.cfg.d_model

    # phase 1: encode every chunk of these providers
    chunk_ids = np.concatenate([hpack.chunks_of(int(i)) for i in provider_indices])
    cls_store = torch.zeros((len(hpack.chunk_lengths), d_model), dtype=torch.float32)
    batches = LengthBucketBatches(hpack.chunk_lengths, chunk_ids, tokens_per_batch, seed)
    for b in batches.batches:
        batch = {k: v.to(device) for k, v in collate_chunks(hpack, b).items()}
        cls = model.encode_chunks(batch)
        cls_store[torch.from_numpy(b)] = cls.float().cpu()

    # phase 2: pool + head per provider, grouped by chunk count (K_pad mult of 4)
    probs = np.zeros(len(provider_indices), dtype=np.float64)
    n_chunks = np.asarray([hpack.n_chunks_of(int(i)) for i in provider_indices])
    order = np.argsort(n_chunks, kind="stable")
    group = 512
    for g0 in range(0, len(order), group):
        rows = order[g0 : g0 + group]
        K = ((int(n_chunks[rows].max()) + 3) // 4) * 4
        chunk_row, chunk_slot, flat = [], [], []
        for local, r in enumerate(rows):
            ids = hpack.chunks_of(int(provider_indices[r]))
            flat.append(torch.from_numpy(ids.astype(np.int64)))
            chunk_row.append(torch.full((len(ids),), local, dtype=torch.long))
            chunk_slot.append(torch.arange(len(ids), dtype=torch.long))
        flat = torch.cat(flat)
        chunk_row, chunk_slot = torch.cat(chunk_row), torch.cat(chunk_slot)
        mask = torch.zeros((len(rows), K), dtype=torch.bool)
        mask[chunk_row, chunk_slot] = True
        logits = model.head_from_cls(
            cls_store[flat].to(device), chunk_row.to(device), chunk_slot.to(device), mask.to(device)
        )
        probs[rows] = torch.sigmoid(logits).float().cpu().numpy()
    return probs


def run_finetune_hier(
    cfg: dict,
    mode: str,
    label: str,
    repo_root: Path,
    device: torch.device | None = None,
    train_frac: float = 1.0,
    frac_seed: int = 0,
    run_name: str | None = None,
) -> Path:
    torch.manual_seed(cfg["seed"])
    device = device or (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    use_bf16 = device.type == "cuda"

    hpack = HierPack(repo_root / cfg["data"]["pack_dir"])
    sidecar = pl.read_parquet(repo_root / cfg["data"]["pack_dir"] / "sidecar.parquet")
    y = sidecar[label].to_numpy().astype(np.float32)

    tr_idx = hpack.indices("train")
    va_idx = hpack.indices("val")
    te_idx = hpack.indices("test")
    if train_frac < 1.0:
        tr_idx = stratified_subsample(tr_idx, y.astype(np.int8), train_frac, frac_seed)
        log.info("label-efficiency: training on %d providers (frac %.2f, seed %d)",
                 len(tr_idx), train_frac, frac_seed)
    run_name = run_name or f"{mode}_{label}"

    model = build_hier_model(
        mode, cfg["model"], hpack.meta["vocab_size"],
        repo_root / cfg["checkpoint"], cfg["train"]["head_dropout"], cfg["pooling"],
    ).to(device)
    lrs = cfg["modes"][mode]
    opt = torch.optim.AdamW(
        hier_param_groups(model, lrs["head_lr"], lrs["encoder_lr"]), weight_decay=0.01
    )

    tpb = cfg["train"]["tokens_per_batch"]
    k_train = cfg["pooling"]["k_train"]
    max_steps = cfg["train"].get("max_steps")
    train_batches = ProviderBatches(hpack, tr_idx, tpb, cfg["seed"], k_train)
    best_ap, best_state, bad = -1.0, None, 0
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        if mode == "probe":
            model.encoder.eval()  # frozen encoder: no dropout noise in features
        for k, bi in enumerate(train_batches.epoch_order(epoch)):
            b = train_batches.batches[bi]
            batch = {kk: v.to(device) for kk, v in
                     collate_hier(hpack, b, epoch, cfg["seed"], k_train).items()}
            target = torch.from_numpy(y[b]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            global_step += 1
            if (k + 1) % cfg["train"]["log_every_steps"] == 0:
                log.info("%s/%s epoch %d step %d loss %.4f", mode, label, epoch, k + 1, loss.item())
            if max_steps and global_step >= max_steps:
                break

        val_p = score_split_hier(model, hpack, va_idx, device, tpb, cfg["seed"])
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
        if max_steps and global_step >= max_steps:
            log.info("%s/%s max_steps %d reached", mode, label, max_steps)
            break

    model.load_state_dict(best_state)
    out_dir = repo_root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idx in (("val", va_idx), ("test", te_idx)):
        p = score_split_hier(model, hpack, idx, device, tpb, cfg["seed"])
        pl.DataFrame(
            {"Provider": sidecar["Provider"].gather(idx), "p": p}
        ).write_parquet(out_dir / f"{run_name}_{split_name}.parquet")

    torch.save({"model": best_state, "mode": mode, "label": label, "config": cfg},
               out_dir / f"{run_name}.pt")
    (out_dir / f"{run_name}_meta.json").write_text(
        json.dumps({"mode": mode, "label": label, "train_frac": train_frac,
                    "frac_seed": frac_seed, "best_val_auprc": best_ap,
                    "k_train": k_train, "seconds": round(time.time() - t0)}, indent=1)
    )
    return out_dir
