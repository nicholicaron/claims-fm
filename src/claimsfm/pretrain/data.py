"""Packed pretraining data: pretokenization, loading, batching, MLM masking.

Pack layout (directory of .npy files, mmap-friendly):
  input_ids u16 · claim_type_ids u8 · age_years u8 · month_idx u8 ·
  visit_ids u16  — flat concatenations over members
  offsets u64    — member i occupies [offsets[i], offsets[i+1])
  split u8       — 0 train / 1 val, per member (seeded hash)
  meta.json      — counts, vocab hash, seed, config

Masking is derived from (seed, epoch, batch_idx) so a resumed run replays the
identical mask stream without storing RNG blobs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

log = logging.getLogger(__name__)

N_SPECIALS = 5  # [PAD] [UNK] [MASK] [CLS] [VISIT]
PAD_ID, UNK_ID, MASK_ID, CLS_ID, VISIT_ID = range(5)
CLAIM_TYPE_IDS = {"IP": 1, "OP": 2, "RX": 3}
STUDY_START = dt.date(2008, 1, 1)

ARRAYS = ["input_ids", "claim_type_ids", "age_years", "month_idx", "visit_ids"]


def pretokenize(
    sequences_path: Path,
    vocab_path: Path,
    out_dir: Path,
    max_len: int,
    val_frac: float,
    seed: int,
    limit_members: int | None = None,
) -> dict:
    """sequences parquet + vocab -> packed arrays. Pure numpy per member row."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    tok2id = vocab["tokens"]

    df = pl.read_parquet(sequences_path)
    if limit_members:
        df = df.head(limit_members)

    cols = {name: [] for name in ARRAYS}
    offsets = [0]
    splits = []
    salt = str(seed).encode()

    for row in df.iter_rows(named=True):
        ids = [CLS_ID]
        ctypes = [0]
        ages = [0]
        months = [0]
        visits = [0]
        prev_visit = None
        birth_year = row["birth_year"] or 1935
        for tok, date, ctype, visit in zip(
            row["tokens"], row["dates"], row["claim_types"], row["visit_ids"]
        ):
            age = min(max(date.year - birth_year, 0), 110)
            month = max(0, (date.year - STUDY_START.year) * 12 + date.month - 1)
            if prev_visit is not None and visit != prev_visit:
                ids.append(VISIT_ID)
                ctypes.append(0)
                ages.append(age)
                months.append(month)
                visits.append(visit)
            ids.append(tok2id.get(tok, UNK_ID))
            ctypes.append(CLAIM_TYPE_IDS[ctype])
            ages.append(age)
            months.append(month)
            visits.append(visit)
            prev_visit = visit
        if len(ids) > max_len:  # keep most recent history, [CLS] anchor stays
            ids = [ids[0]] + ids[-(max_len - 1):]
            ctypes = [ctypes[0]] + ctypes[-(max_len - 1):]
            ages = [ages[0]] + ages[-(max_len - 1):]
            months = [months[0]] + months[-(max_len - 1):]
            visits = [visits[0]] + visits[-(max_len - 1):]

        cols["input_ids"].append(np.asarray(ids, dtype=np.uint16))
        cols["claim_type_ids"].append(np.asarray(ctypes, dtype=np.uint8))
        cols["age_years"].append(np.asarray(ages, dtype=np.uint8))
        cols["month_idx"].append(np.asarray(months, dtype=np.uint8))
        cols["visit_ids"].append(np.asarray(visits, dtype=np.uint16))
        offsets.append(offsets[-1] + len(ids))
        h = hashlib.sha256(salt + row["DESYNPUF_ID"].encode()).digest()[0] / 255
        splits.append(1 if h < val_frac else 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(out_dir / f"{name}.npy", np.concatenate(cols[name]))
    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(out_dir / "split.npy", np.asarray(splits, dtype=np.uint8))

    meta = {
        "n_members": len(splits),
        "n_val_members": int(sum(splits)),
        "n_positions": offsets[-1],
        "max_len": max_len,
        "val_frac": val_frac,
        "seed": seed,
        "vocab_size": len(tok2id),
        "vocab_counts_hash": vocab["meta"]["counts_hash"],
        "source": str(sequences_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("pack: %s members, %s positions -> %s", len(splits), offsets[-1], out_dir)
    return meta


class Pack:
    def __init__(self, pack_dir: Path):
        self.dir = Path(pack_dir)
        self.arrays = {n: np.load(self.dir / f"{n}.npy", mmap_mode="r") for n in ARRAYS}
        self.offsets = np.load(self.dir / "offsets.npy")
        self.split = np.load(self.dir / "split.npy")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.lengths = np.diff(self.offsets).astype(np.int64)

    def indices(self, split: str) -> np.ndarray:
        want = 0 if split == "train" else 1
        return np.flatnonzero(self.split == want)

    def member(self, i: int) -> dict[str, np.ndarray]:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return {n: np.asarray(a[lo:hi]) for n, a in self.arrays.items()}


class LengthBucketBatches:
    """Deterministic length-bucketed batches: cost = n_seqs * batch_max_len.

    Batch composition is fixed (sorted by length); batch ORDER is shuffled per
    epoch with a seed derived from (seed, epoch), so `resume(epoch, k)` can
    skip the first k batches and replay the identical stream.
    """

    def __init__(self, lengths: np.ndarray, indices: np.ndarray, tokens_per_batch: int, seed: int):
        self.seed = seed
        order = indices[np.argsort(lengths[indices], kind="stable")]
        self.batches: list[np.ndarray] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            new_max = max(cur_max, int(lengths[i]))
            if cur and new_max * (len(cur) + 1) > tokens_per_batch:
                self.batches.append(np.asarray(cur))
                cur, cur_max = [], 0
                new_max = int(lengths[i])
            cur.append(int(i))
            cur_max = new_max
        if cur:
            self.batches.append(np.asarray(cur))

    def __len__(self) -> int:
        return len(self.batches)

    def epoch_order(self, epoch: int) -> list[int]:
        rng = np.random.default_rng((self.seed, epoch))
        return rng.permutation(len(self.batches)).tolist()


def collate(pack: Pack, member_ids: np.ndarray) -> dict[str, torch.Tensor]:
    rows = [pack.member(i) for i in member_ids]
    L = max(len(r["input_ids"]) for r in rows)
    out = {}
    for name, dtype in [
        ("input_ids", torch.long), ("claim_type_ids", torch.long),
        ("age_years", torch.long), ("month_idx", torch.long), ("visit_ids", torch.long),
    ]:
        padded = np.zeros((len(rows), L), dtype=np.int64)
        for j, r in enumerate(rows):
            padded[j, : len(r[name])] = r[name]
        out[name] = torch.from_numpy(padded)
    return out


def apply_mlm_mask(
    input_ids: torch.Tensor,
    vocab_size: int,
    rate: float,
    prob_mask: float,
    prob_random: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (masked_input_ids, labels). labels = -100 where not masked."""
    eligible = input_ids >= N_SPECIALS
    scores = torch.rand(input_ids.shape, generator=generator)
    selected = eligible & (scores < rate)

    labels = torch.where(selected, input_ids, torch.full_like(input_ids, -100))
    action = torch.rand(input_ids.shape, generator=generator)
    masked = input_ids.clone()
    masked[selected & (action < prob_mask)] = MASK_ID
    rand_slot = selected & (action >= prob_mask) & (action < prob_mask + prob_random)
    random_codes = torch.randint(
        N_SPECIALS, vocab_size, input_ids.shape, generator=generator
    )
    masked[rand_slot] = random_codes[rand_slot]
    return masked, labels


def mask_generator(seed: int, epoch: int, batch_idx: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(hash((seed, epoch, batch_idx)) % (2**63))
    return g


def frequency_prior(counts: "pl.DataFrame") -> dict[str, float]:
    """Top-1 accuracy of always predicting the modal code — the SPEC §7 M3
    bar the model must clear 'meaningfully'. Overall and per code kind."""
    out = {}
    total = counts["count"].sum()
    out["overall"] = counts["count"].max() / total
    for prefix, name in (("DX_", "dx"), ("PX_", "px"), ("RX_", "rx")):
        sub = counts.filter(pl.col("token").str.starts_with(prefix))
        # baseline predicts the *global* mode; within-kind accuracy is the
        # global mode's share of that kind's occurrences (0 unless mode is of
        # this kind) — report the more generous per-kind mode instead.
        out[name] = sub["count"].max() / sub["count"].sum()
    return out


def kind_lut(vocab_path: Path) -> torch.Tensor:
    """Per-token-id code kind: 0 special, 1 DX, 2 PX, 3 RX."""
    with open(vocab_path) as f:
        tokens = json.load(f)["tokens"]
    lut = torch.zeros(len(tokens), dtype=torch.uint8)
    for tok, i in tokens.items():
        if tok.startswith("DX_"):
            lut[i] = 1
        elif tok.startswith("PX_"):
            lut[i] = 2
        elif tok.startswith("RX_"):
            lut[i] = 3
    return lut
