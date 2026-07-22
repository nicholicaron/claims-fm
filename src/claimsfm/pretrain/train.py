"""Masked-code pretraining loop. Plain PyTorch, config-driven, seeded.

Built for preemptible GPUs (SPEC §10): checkpoints carry model/optimizer/
scheduler/epoch/batch-cursor/best-val state and are written atomically;
--resume replays the identical batch and mask streams from the cursor.
Metrics go to metrics.jsonl (committed figures render from it); W&B attaches
only if WANDB_API_KEY is set.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.nn.functional as F

from claimsfm.model.encoder import ClaimsEncoder, EncoderConfig
from claimsfm.pretrain.data import (
    LengthBucketBatches,
    Pack,
    apply_mlm_mask,
    collate,
    kind_lut,
    mask_generator,
)

log = logging.getLogger(__name__)

KIND_NAMES = {1: "dx", 2: "px", 3: "rx"}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(step: int, total: int, warmup: int, final_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


class JsonlLogger:
    def __init__(self, path: Path, wandb_run=None):
        self.f = open(path, "a")
        self.wandb = wandb_run

    def log(self, record: dict) -> None:
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        if self.wandb:
            self.wandb.log(record)


def maybe_wandb(cfg: dict):
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb  # optional dependency, present only if user opted in

    return wandb.init(project="claims-fm", config=cfg)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


@torch.no_grad()
def evaluate(model, pack, batches, cfg, device, lut) -> dict:
    model.eval()
    vocab_size = pack.meta["vocab_size"]
    total_loss, total_masked, total_correct = 0.0, 0, 0
    kind_masked = {k: 0 for k in KIND_NAMES}
    kind_correct = {k: 0 for k in KIND_NAMES}
    for bi in range(len(batches)):
        batch = collate(pack, batches.batches[bi])
        masked, labels = apply_mlm_mask(
            batch["input_ids"], vocab_size, cfg["masking"]["rate"],
            cfg["masking"]["prob_mask"], cfg["masking"]["prob_random"],
            mask_generator(cfg["seed"], -1, bi),  # fixed eval mask stream
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        masked, labels = masked.to(device), labels.to(device)
        hidden = model(masked, batch["claim_type_ids"], batch["age_years"],
                       batch["month_idx"], batch["visit_ids"])
        logits = model.mlm_logits(hidden)
        sel = labels != -100
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
        )
        pred = logits.argmax(-1)
        hit = (pred == labels) & sel
        n_sel = sel.sum().item()
        total_loss += loss.item() * n_sel
        total_masked += n_sel
        total_correct += hit.sum().item()
        gold_kind = lut.to(device)[labels.clamp(min=0)]
        for k in KIND_NAMES:
            km = (gold_kind == k) & sel
            kind_masked[k] += km.sum().item()
            kind_correct[k] += (km & hit).sum().item()
    model.train()
    out = {
        "val_loss": total_loss / max(1, total_masked),
        "val_masked_acc": total_correct / max(1, total_masked),
    }
    for k, name in KIND_NAMES.items():
        out[f"val_masked_acc_{name}"] = kind_correct[k] / max(1, kind_masked[k])
    return out


def train(
    cfg: dict, repo_root: Path, resume: Path | None = None, device: torch.device | None = None
) -> Path:
    torch.manual_seed(cfg["seed"])
    device = device or pick_device()
    use_bf16 = device.type == "cuda"
    log.info("device=%s bf16=%s", device, use_bf16)

    pack = Pack(repo_root / cfg["data"]["pack_dir"])
    vocab_size = pack.meta["vocab_size"]
    lut = kind_lut(repo_root / "data/processed/vocab.json")

    train_batches = LengthBucketBatches(
        pack.lengths, pack.indices("train"), cfg["train"]["tokens_per_batch"], cfg["seed"]
    )
    val_batches = LengthBucketBatches(
        pack.lengths, pack.indices("val"), cfg["train"]["tokens_per_batch"], cfg["seed"]
    )
    steps_per_epoch = len(train_batches)
    total_steps = steps_per_epoch * cfg["train"]["max_epochs"]
    warmup = int(cfg["optim"]["warmup_frac"] * total_steps)

    model = ClaimsEncoder(EncoderConfig.from_dict(vocab_size, cfg["model"])).to(device)
    log.info("model params: %.2fM", model.num_params() / 1e6)
    if device.type == "cuda" and os.environ.get("CLAIMSFM_COMPILE") == "1":
        # opt-in only: on hosts with a broken inductor toolchain, per-shape
        # compile retries stall training far worse than eager ever could
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optim"]["peak_lr"],
        betas=tuple(cfg["optim"]["betas"]), weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, warmup, cfg["optim"]["final_lr_frac"])
    )

    out_dir = repo_root / cfg["train"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, start_batch, global_step = 0, 0, 0
    best_val, bad_evals = float("inf"), 0
    if resume:
        # always CPU: load_state_dict moves params/opt state to the live device,
        # and set_rng_state requires a CPU ByteTensor
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, start_batch = state["epoch"], state["batch_idx"]
        global_step, best_val, bad_evals = state["global_step"], state["best_val"], state["bad_evals"]
        torch.set_rng_state(state["torch_rng"].cpu())
        log.info("resumed from %s at epoch %d batch %d step %d", resume, start_epoch, start_batch, global_step)

    logger = JsonlLogger(out_dir / "metrics.jsonl", maybe_wandb(cfg))
    kept: list[Path] = sorted(out_dir.glob("step_*.pt"))

    def checkpoint(epoch: int, batch_idx: int, tag: str | None = None) -> None:
        state = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch, "batch_idx": batch_idx,
            "global_step": global_step, "best_val": best_val, "bad_evals": bad_evals,
            "torch_rng": torch.get_rng_state(), "config": cfg,
            "vocab_counts_hash": pack.meta["vocab_counts_hash"],
        }
        path = out_dir / (f"{tag}.pt" if tag else f"step_{global_step:07d}.pt")
        save_checkpoint(path, state)
        if not tag:
            kept.append(path)
            while len(kept) > cfg["train"]["keep_checkpoints"]:
                kept.pop(0).unlink(missing_ok=True)

    model.train()
    t0, tokens_seen = time.time(), 0
    stop = False
    for epoch in range(start_epoch, cfg["train"]["max_epochs"]):
        order = train_batches.epoch_order(epoch)
        for k in range(start_batch, len(order)):
            bi = order[k]
            batch = collate(pack, train_batches.batches[bi])
            masked, labels = apply_mlm_mask(
                batch["input_ids"], vocab_size, cfg["masking"]["rate"],
                cfg["masking"]["prob_mask"], cfg["masking"]["prob_random"],
                mask_generator(cfg["seed"], epoch, bi),
            )
            batch = {kk: v.to(device) for kk, v in batch.items()}
            masked, labels = masked.to(device), labels.to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                hidden = model(masked, batch["claim_type_ids"], batch["age_years"],
                               batch["month_idx"], batch["visit_ids"])
                logits = model.mlm_logits(hidden)
                # fixed-shape loss (no boolean gather): keeps MPS/CUDA graphs static
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size).float(), labels.view(-1), ignore_index=-100
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()
            global_step += 1
            tokens_seen += int(batch["input_ids"].numel())

            max_steps = cfg["train"].get("max_steps")
            if max_steps and global_step >= max_steps:
                checkpoint(epoch, k + 1, tag="last")
                log.info("max_steps %d reached; checkpointed for resume", max_steps)
                return out_dir / "last.pt"

            if global_step % cfg["train"]["log_every_steps"] == 0:
                with torch.no_grad():
                    sel = labels != -100
                    hits = (logits.argmax(-1) == labels) & sel
                    acc = (hits.sum() / sel.sum().clamp(min=1)).item()
                dt_s = time.time() - t0
                logger.log({
                    "step": global_step, "epoch": epoch, "loss": round(loss.item(), 4),
                    "masked_acc": round(acc, 4), "lr": sched.get_last_lr()[0],
                    "tokens_per_s": round(tokens_seen / max(dt_s, 1e-6)),
                })
                t0, tokens_seen = time.time(), 0
            if global_step % cfg["train"]["checkpoint_every_steps"] == 0:
                checkpoint(epoch, k + 1)

        start_batch = 0
        val = evaluate(model, pack, val_batches, cfg, device, lut)
        logger.log({"step": global_step, "epoch": epoch, **{k2: round(v, 4) for k2, v in val.items()}})
        if val["val_loss"] < best_val:
            best_val, bad_evals = val["val_loss"], 0
            checkpoint(epoch + 1, 0, tag="best")
        else:
            bad_evals += 1
            if bad_evals >= cfg["train"]["early_stop_patience"]:
                log.info("early stop after epoch %d (best val %.4f)", epoch, best_val)
                stop = True
        checkpoint(epoch + 1, 0, tag="last")
        if stop:
            break

    return out_dir / "best.pt"
