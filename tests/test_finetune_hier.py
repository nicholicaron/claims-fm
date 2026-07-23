"""Hierarchical fine-tuning stack invariants: chunked-pack conservation and
v1.0 byte-equivalence, provider-major batching, gated MIL pooling, flat-model
equivalence on single-chunk providers. CPU, tiny data."""

import json

import numpy as np
import polars as pl
import pytest
import torch

from claimsfm.config import REPO_ROOT, load_config
from claimsfm.finetune.hier_data import (
    ARRAYS,
    HierPack,
    ProviderBatches,
    collate_chunks,
    collate_hier,
)
from claimsfm.finetune.hier_model import (
    GatedAttentionPool,
    build_hier_model,
    hier_param_groups,
)
from claimsfm.finetune.model import build_model
from claimsfm.pretrain.data import Pack


@pytest.fixture(scope="module")
def hier_cfg():
    return load_config("configs/finetune_b_hier.yaml")


@pytest.fixture(scope="module")
def hpack(hier_cfg):
    d = REPO_ROOT / hier_cfg["data"]["pack_dir"]
    if not (d / "sidecar.parquet").exists():
        pytest.skip("chunked task B pack not built (scripts/build_task_b_pack_chunked.py)")
    return HierPack(d)


@pytest.fixture(scope="module")
def old_pack():
    d = REPO_ROOT / "data/processed/task_b_pack"
    if not (d / "sidecar.parquet").exists():
        pytest.skip("v1.0 task B pack not built")
    return Pack(d)


# ---------------- real-pack invariants (skip if not built) ----------------


def test_chunked_pack_invariants(hpack):
    m = hpack.meta
    assert len(hpack.provider_chunks) - 1 == m["n_members"]
    assert len(hpack.split) == m["n_members"]
    assert len(hpack.chunk_lengths) == m["n_chunks"]
    assert hpack.chunk_lengths.max() <= m["chunk_len"]
    assert hpack.chunk_lengths.min() >= 1
    # every claim kept: the whole point of the chunked pack
    assert m["claims_kept"] == m["claims_total"]
    assert m["n_truncated_providers"] == 0
    # every chunk starts with [CLS] (id 3)
    starts = np.asarray(hpack.arrays["input_ids"])[hpack.offsets[:-1].astype(np.int64)]
    assert (starts == 3).all()
    # split contract unchanged
    task_meta = json.loads((REPO_ROOT / "data/processed/task_b_meta.json").read_text())
    assert m["splits_sha256"] == task_meta["splits_sha256"]


def test_single_chunk_equals_v1_truncation_universe(hpack, old_pack):
    # chunk capacity mirrors the v1.0 budget accounting, so single-chunk here
    # must be exactly the untruncated-in-v1.0 providers
    n_single = hpack.meta["provider_stats"]["n_single_chunk"]
    assert n_single == old_pack.meta["n_members"] - old_pack.meta["n_truncated_providers"]


def test_chunk0_byte_equal_to_v1_rows(hpack, old_pack):
    old_sidecar = pl.read_parquet(old_pack.dir / "sidecar.parquet")
    new_sidecar = pl.read_parquet(hpack.dir / "sidecar.parquet")
    assert old_sidecar["Provider"].to_list() == new_sidecar["Provider"].to_list()
    n_chunks = np.diff(hpack.provider_chunks)
    single = np.flatnonzero(n_chunks == 1)
    assert len(single) == hpack.meta["provider_stats"]["n_single_chunk"]
    for i in single:
        old_row = old_pack.member(int(i))
        new_row = hpack.chunk(int(hpack.provider_chunks[i]))
        for name in ARRAYS:
            assert np.array_equal(old_row[name], new_row[name]), (
                f"provider row {i} channel {name} differs from v1.0"
            )


def test_month_nondecreasing_across_chunk_boundaries(hpack):
    months = np.asarray(hpack.arrays["month_idx"])
    n_chunks = np.diff(hpack.provider_chunks)
    multi = np.flatnonzero(n_chunks > 1)[:200]
    for i in multi:
        lo, hi = int(hpack.provider_chunks[i]), int(hpack.provider_chunks[i + 1])
        for j in range(lo, hi - 1):
            last_of_j = months[int(hpack.offsets[j + 1]) - 1]
            # position 0 of the next chunk is [CLS] (month 0); claims start at 1
            first_of_next = months[int(hpack.offsets[j + 1]) + 1]
            assert first_of_next >= last_of_j, f"provider {i} chunk {j - lo}"


# ---------------- synthetic units (no data needed) ----------------


def _write_tiny_pack(tmp_path, provider_chunk_lens, chunk_len=16):
    rng = np.random.default_rng(0)
    cols = {n: [] for n in ARRAYS}
    offsets = [0]
    pc = [0]
    for lens in provider_chunk_lens:
        for L in lens:
            ids = np.concatenate([[3], rng.integers(5, 90, L - 1)]).astype(np.uint16)
            cols["input_ids"].append(ids)
            cols["claim_type_ids"].append(
                np.concatenate([[0], np.ones(L - 1)]).astype(np.uint8))
            cols["age_years"].append(np.full(L, 70, dtype=np.uint8))
            cols["month_idx"].append(np.full(L, 5, dtype=np.uint8))
            cols["visit_ids"].append(np.arange(L, dtype=np.uint16))
            offsets.append(offsets[-1] + L)
        pc.append(pc[-1] + len(lens))
    for n, arrs in cols.items():
        np.save(tmp_path / f"{n}.npy", np.concatenate(arrs))
    np.save(tmp_path / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(tmp_path / "provider_chunks.npy", np.asarray(pc, dtype=np.uint32))
    n = len(provider_chunk_lens)
    np.save(tmp_path / "split.npy", np.zeros(n, dtype=np.uint8))
    pl.DataFrame({
        "Provider": [f"PRV{i:04d}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "split": ["train"] * n,
    }).write_parquet(tmp_path / "sidecar.parquet")
    (tmp_path / "meta.json").write_text(json.dumps({
        "n_members": n, "n_chunks": pc[-1], "n_positions": offsets[-1],
        "chunk_len": chunk_len, "vocab_size": 100,
        "claims_total": 0, "claims_kept": 0, "n_truncated_providers": 0,
        "splits_sha256": "test", "vocab_counts_hash": "test",
        "provider_stats": {},
    }))
    return HierPack(tmp_path)


LENS = [[4], [6, 3], [2], [5, 5, 5, 5, 5], [7], [3, 3], [8], [2, 2, 2]]


def test_provider_batches_cover_each_provider_once(tmp_path):
    hp = _write_tiny_pack(tmp_path, LENS)
    idx = np.arange(len(LENS))
    batches = ProviderBatches(hp, idx, tokens_per_batch=64, seed=1, k_train=3, pad_multiple=4)
    seen = np.concatenate(batches.batches)
    assert sorted(seen.tolist()) == idx.tolist()
    # deterministic composition and epoch order
    again = ProviderBatches(hp, idx, tokens_per_batch=64, seed=1, k_train=3, pad_multiple=4)
    assert all(np.array_equal(a, b) for a, b in zip(batches.batches, again.batches))
    assert batches.epoch_order(2) == again.epoch_order(2)
    # cost bound: counted chunks * max padded len within budget
    for b in batches.batches:
        n_counted = sum(min(hp.n_chunks_of(int(i)), 3) for i in b)
        eff = 0
        for i in b:
            lens = hp.chunk_lengths[hp.chunks_of(int(i))]
            top = np.sort(lens)[::-1][:3]
            eff = max(eff, int(np.minimum(16, ((top + 3) // 4) * 4).max()))
        assert n_counted * eff <= 64


def test_provider_batches_rejects_unfittable_config(tmp_path):
    hp = _write_tiny_pack(tmp_path, LENS)
    with pytest.raises(ValueError):
        ProviderBatches(hp, np.arange(3), tokens_per_batch=16, seed=1, k_train=8)


def test_collate_hier_shapes_and_determinism(tmp_path):
    hp = _write_tiny_pack(tmp_path, LENS)
    ids = np.array([1, 3, 4])  # 2, 5, 1 chunks
    b1 = collate_hier(hp, ids, epoch=0, seed=7, k_train=3, pad_multiple=4)
    assert b1["mask"].shape[1] == 3  # constant K = k_train
    assert b1["mask"].sum(dim=1).tolist() == [2, 3, 1]  # min(n_chunks, K)
    assert b1["input_ids"].shape[0] == 6  # only real chunks hit the encoder
    assert b1["input_ids"].shape[1] % 4 == 0
    b2 = collate_hier(hp, ids, epoch=0, seed=7, k_train=3, pad_multiple=4)
    assert torch.equal(b1["input_ids"], b2["input_ids"])  # replayable stream
    # eval mode: all chunks, K padded to a multiple of 4
    be = collate_hier(hp, ids, epoch=0, seed=7, k_train=None, pad_multiple=4)
    assert be["mask"].sum(dim=1).tolist() == [2, 5, 1]
    assert be["mask"].shape[1] % 4 == 0


def test_gated_pool_properties():
    torch.manual_seed(0)
    pool = GatedAttentionPool(d_model=16, d_att=8).eval()
    cls = torch.randn(6, 16)
    row = torch.tensor([0, 1, 1, 2, 2, 2])
    slot = torch.tensor([0, 0, 1, 0, 1, 2])
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[row, slot] = True
    pooled, attn = pool(cls, row, slot, mask)
    # weights sum to 1 over real slots; empty slots exactly 0
    assert torch.allclose(attn.sum(dim=1), torch.ones(3), atol=1e-6)
    assert (attn[~mask] == 0).all()
    # single-chunk provider: pooled == its chunk cls exactly
    assert torch.allclose(pooled[0], cls[0], atol=1e-6)
    # permutation invariance: shuffling a provider's chunks changes nothing
    perm = torch.tensor([0, 2, 1, 5, 3, 4])
    pooled_p, _ = pool(cls[perm],
                       torch.tensor([0, 1, 1, 2, 2, 2]),
                       torch.tensor([0, 0, 1, 0, 1, 2]), mask)
    assert torch.allclose(pooled_p[2], pooled[2], atol=1e-5)


_TINY_MODEL = dict(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.1,
                   max_positions=512, n_claim_types=4, n_ages=112, n_months=37)
_POOL = {"kind": "gated", "d_att": 16, "k_train": 4}


@pytest.fixture()
def tiny_hier_ckpt(tmp_path):
    m = build_hier_model("scratch", _TINY_MODEL, 100, None, 0.1, _POOL)
    path = tmp_path / "best.pt"
    torch.save({"model": m.encoder.state_dict()}, path)
    return path


def _fake_hier_batch(n=3, L=8):
    g = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.randint(5, 100, (n, L), generator=g),
        "claim_type_ids": torch.randint(0, 4, (n, L), generator=g),
        "age_years": torch.randint(60, 90, (n, L), generator=g),
        "month_idx": torch.randint(0, 24, (n, L), generator=g),
        "visit_ids": torch.randint(0, 5, (n, L), generator=g),
        "chunk_row": torch.arange(n),
        "chunk_slot": torch.zeros(n, dtype=torch.long),
        "mask": torch.zeros(n, 4, dtype=torch.bool).index_fill_(1, torch.tensor([0]), True),
    }


def test_hier_equals_flat_on_single_chunk_providers():
    torch.manual_seed(0)
    hier = build_hier_model("scratch", _TINY_MODEL, 100, None, 0.1, _POOL).eval()
    flat = build_model("scratch", _TINY_MODEL, 100, None, 0.1).eval()
    flat.encoder.load_state_dict(hier.encoder.state_dict())
    flat.head.load_state_dict(hier.head.state_dict())
    batch = _fake_hier_batch()
    logits_hier = hier(batch)
    logits_flat = flat({k: batch[k] for k in
                        ("input_ids", "claim_type_ids", "age_years", "month_idx", "visit_ids")})
    assert torch.allclose(logits_hier, logits_flat, atol=1e-5)


def test_hier_probe_freezes_encoder_but_trains_pooler(tiny_hier_ckpt):
    model = build_hier_model("probe", _TINY_MODEL, 100, tiny_hier_ckpt, 0.1, _POOL)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    pool_before = {k: v.clone() for k, v in model.pool.state_dict().items()}
    opt = torch.optim.AdamW(hier_param_groups(model, 1e-2, 0.0))
    model.train(); model.encoder.eval()
    for _ in range(3):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(_fake_hier_batch()), torch.tensor([1.0, 0.0, 1.0]))
        opt.zero_grad(); loss.backward(); opt.step()
    for k in before:
        assert torch.equal(before[k], model.encoder.state_dict()[k]), f"encoder {k} moved"
    assert any(not torch.equal(pool_before[k], model.pool.state_dict()[k]) for k in pool_before)


def test_gradient_reaches_encoder_through_pooling(tiny_hier_ckpt):
    model = build_hier_model("full", _TINY_MODEL, 100, tiny_hier_ckpt, 0.1, _POOL)
    model.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(_fake_hier_batch()), torch.tensor([1.0, 0.0, 1.0]))
    loss.backward()
    g = model.encoder.blocks[0].qkv.weight.grad
    assert g is not None and g.abs().sum() > 0
