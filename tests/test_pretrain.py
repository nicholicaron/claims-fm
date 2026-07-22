"""Pretraining stack invariants: masking, pack fidelity, resume determinism,
frequency-prior baseline. CPU-only, tiny data — no GPU required."""

import copy
import json

import numpy as np
import polars as pl
import pytest
import torch

from claimsfm.config import REPO_ROOT, data_path, load_config
from claimsfm.pretrain.data import (
    MASK_ID,
    N_SPECIALS,
    Pack,
    apply_mlm_mask,
    frequency_prior,
    mask_generator,
    pretokenize,
)
from claimsfm.tokenizer import ClaimsTokenizer


VOCAB_SIZE = 1000


def _rand_ids(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB_SIZE, (4, n // 4))
    return torch.from_numpy(ids)


def test_masking_rate_and_specials():
    ids = _rand_ids()
    masked, labels = apply_mlm_mask(ids, VOCAB_SIZE, 0.15, 0.8, 0.1, mask_generator(1, 0, 0))
    eligible = ids >= N_SPECIALS
    selected = labels != -100

    assert not (selected & ~eligible).any(), "special token was masked"
    rate = selected.sum().item() / eligible.sum().item()
    assert 0.14 < rate < 0.16

    became_mask = (masked == MASK_ID) & selected
    kept = (masked == ids) & selected
    randomized = selected & ~became_mask & ~kept
    n = selected.sum().item()
    assert 0.77 < became_mask.sum().item() / n < 0.83
    assert 0.07 < randomized.sum().item() / n < 0.13
    # random replacements are valid code tokens, never specials
    assert (masked[randomized] >= N_SPECIALS).all()
    # unmasked positions untouched
    assert (masked[~selected] == ids[~selected]).all()


def test_masking_deterministic_per_seed():
    ids = _rand_ids()
    m1, l1 = apply_mlm_mask(ids, VOCAB_SIZE, 0.15, 0.8, 0.1, mask_generator(1, 2, 3))
    m2, l2 = apply_mlm_mask(ids, VOCAB_SIZE, 0.15, 0.8, 0.1, mask_generator(1, 2, 3))
    m3, _ = apply_mlm_mask(ids, VOCAB_SIZE, 0.15, 0.8, 0.1, mask_generator(1, 2, 4))
    assert torch.equal(m1, m2) and torch.equal(l1, l2)
    assert not torch.equal(m1, m3)


@pytest.fixture(scope="module")
def tiny_pack(tmp_path_factory, cfg):
    seqs = data_path(cfg, "processed") / "sequences_pretrain.parquet"
    if not seqs.exists():
        pytest.skip("pretrain sequences not built")
    out = tmp_path_factory.mktemp("pack")
    pretokenize(
        sequences_path=seqs,
        vocab_path=data_path(cfg, "processed") / "vocab.json",
        out_dir=out,
        max_len=512,
        val_frac=0.1,
        seed=7,
        limit_members=60,
    )
    return out


def test_pack_matches_tokenizer(tiny_pack, cfg):
    """Packed ids must equal ClaimsTokenizer.encode output member-for-member."""
    pack = Pack(tiny_pack)
    tok = ClaimsTokenizer.load(data_path(cfg, "processed") / "vocab.json")
    df = pl.read_parquet(data_path(cfg, "processed") / "sequences_pretrain.parquet").head(60)
    for i in (0, 7, 41):
        row = df.row(i, named=True)
        enc = tok.encode(row["tokens"], row["dates"], row["claim_types"], row["visit_ids"], max_len=512)
        got = pack.member(i)["input_ids"].astype(np.int64)
        assert got.tolist() == enc["input_ids"], f"member {i} id mismatch"


def test_pack_split_reproducible(tiny_pack):
    pack = Pack(tiny_pack)
    assert pack.split.sum() > 0  # some val members at 10%
    assert len(pack.lengths) == pack.meta["n_members"] == 60
    assert int(pack.offsets[-1]) == pack.meta["n_positions"]


def test_frequency_prior_math():
    counts = pl.DataFrame(
        {"token": ["DX_A", "DX_B", "RX_C", "PX_D"], "count": [50, 30, 15, 5]}
    )
    prior = frequency_prior(counts)
    assert prior["overall"] == pytest.approx(0.5)
    assert prior["dx"] == pytest.approx(50 / 80)
    assert prior["rx"] == pytest.approx(1.0)


def test_resume_determinism(tiny_pack, tmp_path):
    """N steps straight == N/2 steps + checkpoint + resume, bitwise on CPU."""
    from claimsfm.pretrain.train import train

    base = load_config("configs/pretrain_smoke.yaml")
    base["model"].update(d_model=32, n_layers=1, n_heads=2, d_ff=64)
    base["train"].update(tokens_per_batch=2048, max_epochs=2, checkpoint_every_steps=1000,
                         log_every_steps=1000)

    def run(out_dir, max_steps=None, resume=None):
        cfg = copy.deepcopy(base)
        cfg["data"]["pack_dir"] = str(tiny_pack)  # absolute; REPO_ROOT / abs -> abs
        cfg["train"]["out_dir"] = str(out_dir)
        if max_steps:
            cfg["train"]["max_steps"] = max_steps
        return train(cfg, REPO_ROOT, resume=resume, device=torch.device("cpu"))

    straight = run(tmp_path / "straight")
    interrupted = run(tmp_path / "split", max_steps=4)
    resumed = run(tmp_path / "split", resume=interrupted)

    a = torch.load(straight, map_location="cpu", weights_only=False)["model"]
    b = torch.load(resumed, map_location="cpu", weights_only=False)["model"]
    for key in a:
        assert torch.allclose(a[key], b[key], atol=0, rtol=0), f"mismatch at {key}"
