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
import os
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


_DTYPES = {"input_ids": np.uint16, "claim_type_ids": np.uint8, "age_years": np.uint8,
           "month_idx": np.uint8, "visit_ids": np.uint16}


def _encode_member(row: dict, tok2id: dict[str, int], max_len: int) -> dict[str, np.ndarray]:
    """One member's sequences-row -> packed arrays. Members with no events pack
    as a lone [CLS] — real cohort members with empty observation history."""
    ids = [CLS_ID]
    ctypes = [0]
    ages = [0]
    months = [0]
    visits = [0]
    prev_visit = None
    birth_year = row["birth_year"] or 1935
    for tok, date, ctype, visit in zip(
        row["tokens"] or [], row["dates"] or [], row["claim_types"] or [], row["visit_ids"] or []
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
    return {
        "input_ids": np.asarray(ids, dtype=np.uint16),
        "claim_type_ids": np.asarray(ctypes, dtype=np.uint8),
        "age_years": np.asarray(ages, dtype=np.uint8),
        "month_idx": np.asarray(months, dtype=np.uint8),
        "visit_ids": np.asarray(visits, dtype=np.uint16),
    }


def pack_frame(df: pl.DataFrame, tok2id: dict[str, int], out_dir: Path, max_len: int) -> dict:
    """Encode a sequences frame into the packed array files (in-memory path;
    fine for smoke packs — the corpus-scale path is pretokenize's two-pass
    memmap build)."""
    cols = {name: [] for name in ARRAYS}
    offsets = [0]
    for row in df.iter_rows(named=True):
        arrs = _encode_member(row, tok2id, max_len)
        for name in ARRAYS:
            cols[name].append(arrs[name])
        offsets.append(offsets[-1] + len(arrs["input_ids"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(out_dir / f"{name}.npy", np.concatenate(cols[name]))
    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    return {"n_members": len(df), "n_positions": offsets[-1], "max_len": max_len}


def pretokenize(
    sequences_path: Path,
    vocab_path: Path,
    out_dir: Path,
    max_len: int,
    val_frac: float,
    seed: int,
    limit_members: int | None = None,
    slice_size: int = 100_000,
) -> dict:
    """sequences parquet + vocab -> packed arrays with a hash-based val split.

    Two-pass memmap build so corpus size is bounded by `slice_size` members in
    RAM, not the whole parquet (the 18-sample corpus would not fit the 8GB
    build machine): pass 1 computes exact packed lengths from the count
    columns (1 + n_events + (n_visits-1), capped at max_len — asserted against
    the encoder per member in pass 2); pass 2 writes members slice by slice
    into preallocated .npy memmaps.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    tok2id = vocab["tokens"]

    lf = pl.scan_parquet(sequences_path)
    if limit_members:
        lf = lf.head(limit_members)

    head = lf.select(
        "DESYNPUF_ID", "sample_id",
        pl.min_horizontal(
            1 + pl.col("n_events").cast(pl.Int64)
            + (pl.col("n_visits").cast(pl.Int64) - 1).clip(0),
            max_len,
        ).alias("packed_len"),
    ).collect(engine="streaming")
    n_members = head.height
    lens = head["packed_len"].to_numpy().astype(np.int64)
    offsets = np.zeros(n_members + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(lens)
    total = int(offsets[-1])
    source_samples = sorted(head["sample_id"].unique().to_list())

    # frozen-pack guard: a pack dir is a corpus artifact; refuse to silently
    # rebuild it from a different corpus (e.g. the 5-sample pretrain_pack
    # after data.yaml grew to samples 3-20)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and os.environ.get("CLAIMSFM_FORCE_REPACK") != "1":
        old = json.loads(meta_path.read_text())
        old_src = old.get("source_samples")
        differs = (old_src is not None and old_src != source_samples) or (
            old_src is None and old.get("n_members") != n_members
        )
        if differs:
            raise RuntimeError(
                f"{out_dir} holds a pack from samples {old_src or 'unknown'} "
                f"({old.get('n_members')} members); refusing to overwrite with samples "
                f"{source_samples} ({n_members} members). Point pack_dir elsewhere, or set "
                "CLAIMSFM_FORCE_REPACK=1 if this is intentional."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    mm = {
        name: np.lib.format.open_memmap(
            out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(total,)
        )
        for name, dtype in _DTYPES.items()
    }
    row_lo = 0
    while row_lo < n_members:
        df = lf.slice(row_lo, slice_size).collect()
        for local, row in enumerate(df.iter_rows(named=True)):
            i = row_lo + local
            arrs = _encode_member(row, tok2id, max_len)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            if hi - lo != len(arrs["input_ids"]):
                raise RuntimeError(
                    f"member {i}: pass-1 length {hi - lo} != encoded {len(arrs['input_ids'])}"
                )
            for name in ARRAYS:
                mm[name][lo:hi] = arrs[name]
        row_lo += df.height
        log.info("pretokenize: %d / %d members", row_lo, n_members)
        if df.height == 0:
            raise RuntimeError("empty slice before reaching n_members")
    for a in mm.values():
        a.flush()
    np.save(out_dir / "offsets.npy", offsets)

    salt = str(seed).encode()
    splits = np.fromiter(
        (
            1 if hashlib.sha256(salt + mid.encode()).digest()[0] / 255 < val_frac else 0
            for mid in head["DESYNPUF_ID"]
        ),
        dtype=np.uint8,
        count=n_members,
    )
    np.save(out_dir / "split.npy", splits)

    ids_arr = mm["input_ids"]
    n_unk = int((ids_arr == UNK_ID).sum())
    n_code = int((ids_arr >= N_SPECIALS).sum())

    meta = {
        "n_members": n_members,
        "n_positions": total,
        "max_len": max_len,
        "n_val_members": int(splits.sum()),
        "val_frac": val_frac,
        "seed": seed,
        "vocab_size": len(tok2id),
        "vocab_counts_hash": vocab["meta"]["counts_hash"],
        "source": str(sequences_path),
        "source_samples": source_samples,
        "unk_rate": round(n_unk / max(1, n_unk + n_code), 6),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(
        "pack: %s members, %s positions, unk %.3f%% -> %s",
        n_members, total, 100 * meta["unk_rate"], out_dir,
    )
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
        want = {"train": 0, "val": 1, "test": 2}[split]
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

    def __init__(
        self,
        lengths: np.ndarray,
        indices: np.ndarray,
        tokens_per_batch: int,
        seed: int,
        pad_multiple: int = 64,
        max_len: int = 512,
    ):
        self.seed = seed
        # cost must use the length collate will actually pad to, or batches of
        # short sequences blow past the token budget by up to pad_multiple x
        eff = np.minimum(max_len, ((lengths + pad_multiple - 1) // pad_multiple) * pad_multiple)
        order = indices[np.argsort(lengths[indices], kind="stable")]
        self.batches: list[np.ndarray] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            new_max = max(cur_max, int(eff[i]))
            if cur and new_max * (len(cur) + 1) > tokens_per_batch:
                self.batches.append(np.asarray(cur))
                cur, cur_max = [], 0
                new_max = int(eff[i])
            cur.append(int(i))
            cur_max = new_max
        if cur:
            self.batches.append(np.asarray(cur))

    def __len__(self) -> int:
        return len(self.batches)

    def epoch_order(self, epoch: int) -> list[int]:
        rng = np.random.default_rng((self.seed, epoch))
        return rng.permutation(len(self.batches)).tolist()


def collate(pack: Pack, member_ids: np.ndarray, pad_multiple: int = 64) -> dict[str, torch.Tensor]:
    rows = [pack.member(i) for i in member_ids]
    L = max(len(r["input_ids"]) for r in rows)
    # quantize the padded length: a handful of distinct tensor shapes instead
    # of one per batch (MPS recompiles kernels per shape; CUDA caches better too)
    L = min(512, ((L + pad_multiple - 1) // pad_multiple) * pad_multiple)
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
