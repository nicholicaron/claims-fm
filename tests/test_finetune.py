"""Fine-tuning stack invariants: pack/sidecar alignment, empty-history
handling, probe freeze, split contract, scratch-init sanity. CPU, tiny data."""

import json

import numpy as np
import polars as pl
import pytest
import torch

from claimsfm.config import REPO_ROOT, data_path, load_config
from claimsfm.finetune.model import build_model, param_groups
from claimsfm.pretrain.data import Pack, collate


@pytest.fixture(scope="module")
def ft_cfg():
    return load_config("configs/finetune_a.yaml")


@pytest.fixture(scope="module")
def pack(cfg, ft_cfg):
    d = REPO_ROOT / ft_cfg["data"]["pack_dir"]
    if not (d / "sidecar.parquet").exists():
        pytest.skip("task A pack not built (scripts/build_task_a_pack.py)")
    return Pack(d), pl.read_parquet(d / "sidecar.parquet")


def test_pack_sidecar_alignment(pack, cfg):
    p, sidecar = pack
    assert len(p.lengths) == len(sidecar) == p.meta["n_members"]
    feats = pl.read_parquet(
        data_path(cfg, "processed") / "task_a_features.parquet",
        columns=["DESYNPUF_ID", "label_ip", "label_cost", "split"],
    ).sort("DESYNPUF_ID")
    assert sidecar["DESYNPUF_ID"].to_list() == feats["DESYNPUF_ID"].to_list()
    assert (sidecar["label_ip"] == feats["label_ip"]).all()
    assert (sidecar["split"] == feats["split"]).all()


def test_split_contract_hash(pack, cfg):
    _, sidecar = pack
    meta = json.loads((REPO_ROOT / "data/processed/task_a_pack/meta.json").read_text())
    task_meta = json.loads((data_path(cfg, "processed") / "task_a_meta.json").read_text())
    assert meta["splits_sha256"] == task_meta["splits_sha256"]
    splits = set(sidecar["split"].unique())
    assert splits == {"train", "val", "test"}


def test_empty_history_members_are_cls_only(pack):
    p, _ = pack
    assert p.meta["n_empty_history"] > 0  # healthy members exist
    empties = np.flatnonzero(p.lengths == 1)
    assert len(empties) == p.meta["n_empty_history"]
    row = p.member(int(empties[0]))
    assert row["input_ids"].tolist() == [3]  # [CLS]
    # survives collate alongside real sequences
    batch = collate(p, np.array([int(empties[0]), 0]))
    assert batch["input_ids"].shape[0] == 2


def _tiny_model(mode, ckpt):
    model_cfg = dict(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.1,
                     max_positions=512, n_claim_types=4, n_ages=112, n_months=37)
    return build_model(mode, model_cfg, vocab_size=100, checkpoint=ckpt, head_dropout=0.1)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    m = _tiny_model("scratch", None)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"model": m.encoder.state_dict()}, path)
    return path


def _fake_batch(n=4, L=8):
    g = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.randint(5, 100, (n, L), generator=g),
        "claim_type_ids": torch.randint(0, 4, (n, L), generator=g),
        "age_years": torch.randint(60, 90, (n, L), generator=g),
        "month_idx": torch.randint(0, 24, (n, L), generator=g),
        "visit_ids": torch.randint(0, 5, (n, L), generator=g),
    }


def test_probe_mode_freezes_encoder(tiny_ckpt):
    model = _tiny_model("probe", tiny_ckpt)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    head_before = model.head.weight.clone()
    opt = torch.optim.AdamW(param_groups(model, 1e-2, 0.0))
    model.train(); model.encoder.eval()
    for _ in range(3):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(_fake_batch()), torch.tensor([1.0, 0.0, 1.0, 0.0])
        )
        opt.zero_grad(); loss.backward(); opt.step()
    after = model.encoder.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"encoder param {k} changed in probe mode"
    assert not torch.equal(head_before, model.head.weight)


def test_scratch_differs_from_pretrained(tiny_ckpt):
    pre = _tiny_model("full", tiny_ckpt)
    scratch = _tiny_model("scratch", None)
    same = sum(
        torch.equal(a, b)
        for (_, a), (_, b) in zip(pre.encoder.state_dict().items(), scratch.encoder.state_dict().items())
    )
    total = len(pre.encoder.state_dict())
    assert same < total, "scratch init identical to pretrained checkpoint"
