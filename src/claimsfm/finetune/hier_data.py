"""Task B chunked provider pack + loaders for hierarchical encoding (Phase 2).

The v1.0 pack truncates each provider to one 512-token sequence, discarding
52.8% of claims. Here a provider's claims are packed into 512-token,
claim-boundary-aligned chunks instead — every claim is kept. Each chunk is
`[CLS]` + `[VISIT]`-separated claim spans, the same structure the encoder
pretrained on. Chunk capacity uses the v1.0 budget accounting (every claim
charged tokens+1 against chunk_len-1), so "single-chunk provider" here is
exactly "untruncated provider" in v1.0, and that chunk is byte-identical to
the v1.0 pack row (pinned by tests/test_finetune_hier.py).

Granularities: token spans and offsets are chunk-level; labels, splits and the
sidecar stay provider-level, bridged by `provider_chunks` (provider i owns
chunks [pc[i], pc[i+1])). The generic pretrain Pack/LengthBucketBatches are
not reused for training because their one-span-per-member cost model
misprices multi-chunk providers; ProviderBatches keeps providers whole per
batch (the pooler needs every sampled chunk of a provider in one step).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from claimsfm.finetune.task_b_data import SPLIT_CODE, provider_claim_events

log = logging.getLogger(__name__)

ARRAYS = ("input_ids", "claim_type_ids", "age_years", "month_idx", "visit_ids")


def build_task_b_pack_chunked(
    kaggle_dir: Path,
    features_path: Path,
    meta_path: Path,
    vocab_path: Path,
    out_dir: Path,
    chunk_len: int,
) -> dict:
    with open(vocab_path) as f:
        vocab = json.load(f)
    tok2id = vocab["tokens"]
    unk = tok2id["[UNK]"]
    cls_id, visit_id = tok2id["[CLS]"], tok2id["[VISIT]"]
    task_meta = json.loads(meta_path.read_text())

    sidecar = pl.read_parquet(
        features_path, columns=["Provider", "label", "split"]
    ).sort("Provider")
    key = "\n".join(f"{p}:{s}" for p, s in sidecar.select("Provider", "split").iter_rows())
    got = hashlib.sha256(key.encode()).hexdigest()
    if got != task_meta["splits_sha256"]:
        raise RuntimeError(f"split hash mismatch: {got[:16]} vs {task_meta['splits_sha256'][:16]}")

    events = provider_claim_events(kaggle_dir)
    grouped = dict(
        (prov, sub) for prov, sub in events.group_by("Provider", maintain_order=False)
    )

    cols = {n: [] for n in ARRAYS}
    chunk_offsets = [0]
    provider_chunks = [0]
    total_claims = 0
    chunks_per_provider = []
    tokens_per_provider = []

    for prov in sidecar["Provider"]:
        sub = grouped.get((prov,))
        total_claims += 0 if sub is None else len(sub)
        spans = []
        if sub is not None:
            for claim_no, row in enumerate(sub.iter_rows(named=True), start=1):
                toks = row["tokens"] or []
                if toks:
                    spans.append({
                        "ids": [tok2id.get(t, unk) for t in toks],
                        "ct": row["claim_type_id"], "age": row["age"],
                        "mo": row["month"], "v": claim_no % 65536,  # u16 guard; consumer is %2
                    })

        # greedy oldest->newest, claim never split; capacity mirrors the v1.0
        # keep-most-recent budget (cost = tokens+1 vs chunk_len-1) so a
        # single-chunk provider == an untruncated v1.0 provider, byte-exact
        chunk_spans: list[list[dict]] = [[]]
        budget = chunk_len - 1
        for span in spans:
            cost = len(span["ids"]) + 1
            if cost > chunk_len - 1:
                raise ValueError(f"claim longer than a chunk ({cost} > {chunk_len - 1})")
            if budget - cost < 0:
                chunk_spans.append([])
                budget = chunk_len - 1
            budget -= cost
            chunk_spans[-1].append(span)

        prov_tokens = 0
        for spans_in_chunk in chunk_spans:  # empty-history: one lone [CLS] chunk
            ids, ctypes, ages, months, visits = [cls_id], [0], [0], [0], [0]
            for si, span in enumerate(spans_in_chunk):
                if si > 0:
                    ids.append(visit_id)
                    ctypes.append(0)
                    ages.append(span["age"])
                    months.append(span["mo"])
                    visits.append(span["v"])
                for tid in span["ids"]:
                    ids.append(tid)
                    ctypes.append(span["ct"])
                    ages.append(span["age"])
                    months.append(span["mo"])
                    visits.append(span["v"])
            assert len(ids) <= chunk_len
            cols["input_ids"].append(np.asarray(ids, dtype=np.uint16))
            cols["claim_type_ids"].append(np.asarray(ctypes, dtype=np.uint8))
            cols["age_years"].append(np.asarray(ages, dtype=np.uint8))
            cols["month_idx"].append(np.asarray(months, dtype=np.uint8))
            cols["visit_ids"].append(np.asarray(visits, dtype=np.uint16))
            chunk_offsets.append(chunk_offsets[-1] + len(ids))
            prov_tokens += len(ids)
        provider_chunks.append(provider_chunks[-1] + len(chunk_spans))
        chunks_per_provider.append(len(chunk_spans))
        tokens_per_provider.append(prov_tokens)

    out_dir.mkdir(parents=True, exist_ok=True)
    for n, chunks in cols.items():
        np.save(out_dir / f"{n}.npy", np.concatenate(chunks))
    np.save(out_dir / "offsets.npy", np.asarray(chunk_offsets, dtype=np.uint64))
    np.save(out_dir / "provider_chunks.npy", np.asarray(provider_chunks, dtype=np.uint32))
    np.save(
        out_dir / "split.npy",
        np.array([SPLIT_CODE[s] for s in sidecar["split"]], dtype=np.uint8),
    )
    sidecar.write_parquet(out_dir / "sidecar.parquet", compression="zstd")

    cpp = np.asarray(chunks_per_provider)
    tpp = np.asarray(tokens_per_provider)
    meta = {
        "n_members": len(sidecar),
        "n_chunks": len(chunk_offsets) - 1,
        "n_positions": chunk_offsets[-1],
        "chunk_len": chunk_len,
        # compat keys read by the eval report: nothing truncated, all kept
        "max_len": chunk_len,
        "n_truncated_providers": 0,
        "claims_total": total_claims,
        "claims_kept": total_claims,
        "splits_sha256": got,
        "vocab_size": len(tok2id),
        "vocab_counts_hash": vocab["meta"]["counts_hash"],
        "provider_stats": {
            "n_single_chunk": int((cpp == 1).sum()),
            "chunks_p50": float(np.percentile(cpp, 50)),
            "chunks_p90": float(np.percentile(cpp, 90)),
            "chunks_p99": float(np.percentile(cpp, 99)),
            "chunks_max": int(cpp.max()),
            "tokens_p50": float(np.percentile(tpp, 50)),
            "tokens_p99": float(np.percentile(tpp, 99)),
            "tokens_max": int(tpp.max()),
            "chunks_histogram": {
                str(k): int(v) for k, v in zip(*np.unique(np.minimum(cpp, 16), return_counts=True))
            },  # 16 bucket = "16+"
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(
        "chunked task B pack: %d providers, %d chunks, %d positions; all %d claims kept; "
        "single-chunk %d, max chunks %d",
        meta["n_members"], meta["n_chunks"], meta["n_positions"], total_claims,
        meta["provider_stats"]["n_single_chunk"], meta["provider_stats"]["chunks_max"],
    )
    return meta


class HierPack:
    """mmap view over a chunked pack: chunk-level spans, provider-level splits."""

    def __init__(self, pack_dir: Path):
        self.dir = Path(pack_dir)
        self.arrays = {n: np.load(self.dir / f"{n}.npy", mmap_mode="r") for n in ARRAYS}
        self.offsets = np.load(self.dir / "offsets.npy")
        self.provider_chunks = np.load(self.dir / "provider_chunks.npy")
        self.split = np.load(self.dir / "split.npy")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.chunk_lengths = np.diff(self.offsets).astype(np.int64)

    def indices(self, split: str) -> np.ndarray:
        want = {"train": 0, "val": 1, "test": 2}[split]
        return np.flatnonzero(self.split == want)

    def n_chunks_of(self, i: int) -> int:
        return int(self.provider_chunks[i + 1] - self.provider_chunks[i])

    def chunks_of(self, i: int) -> np.ndarray:
        return np.arange(int(self.provider_chunks[i]), int(self.provider_chunks[i + 1]))

    def chunk(self, j: int) -> dict[str, np.ndarray]:
        lo, hi = int(self.offsets[j]), int(self.offsets[j + 1])
        return {n: np.asarray(a[lo:hi]) for n, a in self.arrays.items()}


class ProviderBatches:
    """Deterministic provider-major batches for hierarchical training.

    Whole providers per batch (the pooler needs all of a provider's sampled
    chunks in one step). Cost model mirrors LengthBucketBatches at chunk
    granularity: every chunk in a batch pads to the batch's max quantized
    chunk length, so cost = total_counted_chunks * batch_max_len. A provider
    counts min(n_chunks, k_train) chunks. Composition is fixed (sorted by
    max padded chunk length, so short single-chunk providers co-batch at
    small L); batch ORDER shuffles per epoch via `epoch_order`.
    """

    def __init__(
        self,
        hpack: HierPack,
        provider_indices: np.ndarray,
        tokens_per_batch: int,
        seed: int,
        k_train: int,
        pad_multiple: int = 64,
    ):
        if k_train is None or k_train < 1:
            raise ValueError("ProviderBatches is train-only; k_train must be >= 1")
        self.seed = seed
        chunk_len = int(hpack.meta["chunk_len"])
        if k_train * chunk_len > tokens_per_batch:
            raise ValueError(
                f"k_train*chunk_len ({k_train * chunk_len}) exceeds tokens_per_batch "
                f"({tokens_per_batch}); a single heavy provider would not fit a batch"
            )
        eff_len = np.zeros(len(provider_indices), dtype=np.int64)
        n_counted = np.zeros(len(provider_indices), dtype=np.int64)
        for row, i in enumerate(provider_indices):
            lens = hpack.chunk_lengths[hpack.chunks_of(int(i))]
            top = np.sort(lens)[::-1][: k_train]
            q = np.minimum(chunk_len, ((top + pad_multiple - 1) // pad_multiple) * pad_multiple)
            eff_len[row] = int(q.max())
            n_counted[row] = len(top)
        order = np.argsort(eff_len, kind="stable")
        self.batches: list[np.ndarray] = []
        cur: list[int] = []
        cur_max = 0
        cur_chunks = 0
        for row in order:
            new_max = max(cur_max, int(eff_len[row]))
            new_chunks = cur_chunks + int(n_counted[row])
            if cur and new_max * new_chunks > tokens_per_batch:
                self.batches.append(np.asarray(cur))
                cur, cur_max, cur_chunks = [], 0, 0
                new_max, new_chunks = int(eff_len[row]), int(n_counted[row])
            cur.append(int(provider_indices[row]))
            cur_max, cur_chunks = new_max, new_chunks
        if cur:
            self.batches.append(np.asarray(cur))

    def __len__(self) -> int:
        return len(self.batches)

    def epoch_order(self, epoch: int) -> list[int]:
        rng = np.random.default_rng((self.seed, epoch))
        return rng.permutation(len(self.batches)).tolist()


def collate_chunks(
    hpack: HierPack, chunk_ids: np.ndarray, pad_multiple: int = 64
) -> dict[str, "torch.Tensor"]:
    """Pad a set of chunks to (N, L) tensors — the eval phase-1 collate."""
    import torch

    rows = [hpack.chunk(int(j)) for j in chunk_ids]
    L = max(len(r["input_ids"]) for r in rows)
    L = min(int(hpack.meta["chunk_len"]), ((L + pad_multiple - 1) // pad_multiple) * pad_multiple)
    out = {}
    for name in ARRAYS:
        padded = np.zeros((len(rows), L), dtype=np.int64)
        for j, r in enumerate(rows):
            padded[j, : len(r[name])] = r[name]
        out[name] = torch.from_numpy(padded)
    return out


def collate_hier(
    hpack: HierPack,
    provider_ids: np.ndarray,
    epoch: int,
    seed: int,
    k_train: int | None,
    pad_multiple: int = 64,
) -> dict[str, "torch.Tensor"]:
    """Batch of whole providers -> flattened chunk tensors + pooling indices.

    Training (k_train set): each provider contributes a random-K sample of its
    chunks, without replacement, seeded per (seed, epoch, provider) — the
    stream replays exactly for a given epoch. The pooler's chunk axis pads to
    the constant K = k_train (fixed shape set: {L buckets} x {K}).
    Eval (k_train None): all chunks; K pads to a multiple of 4.
    """
    import torch

    sel: list[tuple[int, np.ndarray]] = []  # (provider_row, chunk ids)
    for row, i in enumerate(provider_ids):
        chunks = hpack.chunks_of(int(i))
        if k_train is not None and len(chunks) > k_train:
            rng = np.random.default_rng((seed, epoch, int(i)))
            chunks = np.sort(rng.choice(chunks, size=k_train, replace=False))
        sel.append((row, chunks))

    flat = np.concatenate([c for _, c in sel])
    out = collate_chunks(hpack, flat, pad_multiple)

    P = len(provider_ids)
    if k_train is not None:
        K = k_train
    else:
        K = max(len(c) for _, c in sel)
        K = ((K + 3) // 4) * 4
    chunk_row = np.concatenate([np.full(len(c), row, dtype=np.int64) for row, c in sel])
    chunk_slot = np.concatenate([np.arange(len(c), dtype=np.int64) for _, c in sel])
    mask = np.zeros((P, K), dtype=bool)
    mask[chunk_row, chunk_slot] = True
    out["chunk_row"] = torch.from_numpy(chunk_row)
    out["chunk_slot"] = torch.from_numpy(chunk_slot)
    out["mask"] = torch.from_numpy(mask)
    return out
