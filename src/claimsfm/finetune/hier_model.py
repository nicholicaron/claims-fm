"""Hierarchical fine-tuning head: gated-attention MIL pooling over chunk [CLS].

Each 512-token chunk is encoded independently; its [CLS] state is the chunk
representation. Chunk vectors pool per provider with gated attention (Ilse,
Tomczak & Welling, ICML 2018) — fraud is naturally multiple-instance (a few
bad claims among many), and the attention weights give per-chunk
interpretability. For a single-chunk provider the softmax is over one element,
so the model reduces exactly to the v1.0 flat [CLS]-head path (pinned by
tests). Modes mirror the flat model: probe (frozen encoder; pooler+head still
train), full, scratch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from claimsfm.model.encoder import ClaimsEncoder, EncoderConfig

MODES = ("probe", "full", "scratch")


class GatedAttentionPool(nn.Module):
    def __init__(self, d_model: int, d_att: int = 128):
        super().__init__()
        self.V = nn.Linear(d_model, d_att)
        self.U = nn.Linear(d_model, d_att)
        self.w = nn.Linear(d_att, 1)

    def forward(
        self,
        chunk_cls: torch.Tensor,   # (N, d)
        chunk_row: torch.Tensor,   # (N,) provider row per chunk
        chunk_slot: torch.Tensor,  # (N,) slot within provider
        mask: torch.Tensor,        # (P, K) bool, True where a chunk exists
    ) -> tuple[torch.Tensor, torch.Tensor]:
        P, K = mask.shape
        scores = self.w(torch.tanh(self.V(chunk_cls)) * torch.sigmoid(self.U(chunk_cls))).squeeze(-1)
        # dense scatter into fixed (P, K) grids: advanced-indexing assignment
        # only (MPS-safe; no scatter_reduce). Empty slots score -inf -> weight 0.
        # softmax in fp32 regardless of autocast dtype.
        grid = torch.full((P, K), float("-inf"), device=chunk_cls.device, dtype=torch.float32)
        grid[chunk_row, chunk_slot] = scores.float()
        attn = torch.softmax(grid, dim=1)  # (P, K); rows have >=1 real chunk
        feats = chunk_cls.new_zeros((P, K, chunk_cls.shape[-1]))
        feats[chunk_row, chunk_slot] = chunk_cls
        pooled = (attn.to(feats.dtype).unsqueeze(-1) * feats).sum(dim=1)
        return pooled, attn


class MeanPool(nn.Module):
    """Config-flag sanity ablation (pooling.kind: mean)."""

    def forward(self, chunk_cls, chunk_row, chunk_slot, mask):
        P, K = mask.shape
        feats = chunk_cls.new_zeros((P, K, chunk_cls.shape[-1]))
        feats[chunk_row, chunk_slot] = chunk_cls
        attn = mask.to(chunk_cls.dtype) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (attn.unsqueeze(-1) * feats).sum(dim=1)
        return pooled, attn


class HierClaimsFinetune(nn.Module):
    def __init__(
        self,
        encoder: ClaimsEncoder,
        pool_kind: str = "gated",
        d_att: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.d_model
        self.pool = GatedAttentionPool(d, d_att) if pool_kind == "gated" else MeanPool()
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(d, 1)

    def encode_chunks(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(
            batch["input_ids"], batch["claim_type_ids"], batch["age_years"],
            batch["month_idx"], batch["visit_ids"],
        )
        return hidden[:, 0]

    def head_from_cls(
        self,
        chunk_cls: torch.Tensor,
        chunk_row: torch.Tensor,
        chunk_slot: torch.Tensor,
        mask: torch.Tensor,
        return_details: bool = False,
    ):
        pooled, attn = self.pool(chunk_cls, chunk_row, chunk_slot, mask)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        if return_details:
            return logits, pooled, attn
        return logits

    def forward(self, batch: dict[str, torch.Tensor], return_details: bool = False):
        cls = self.encode_chunks(batch)
        return self.head_from_cls(
            cls, batch["chunk_row"], batch["chunk_slot"], batch["mask"], return_details
        )


def build_hier_model(
    mode: str,
    model_cfg: dict,
    vocab_size: int,
    checkpoint: Path | None,
    head_dropout: float,
    pool_cfg: dict,
) -> HierClaimsFinetune:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    encoder = ClaimsEncoder(EncoderConfig.from_dict(vocab_size, model_cfg))
    if mode in ("probe", "full"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["model"])
    model = HierClaimsFinetune(
        encoder,
        pool_kind=pool_cfg.get("kind", "gated"),
        d_att=pool_cfg.get("d_att", 128),
        head_dropout=head_dropout,
    )
    if mode == "probe":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return model


def hier_param_groups(model: HierClaimsFinetune, head_lr: float, encoder_lr: float) -> list[dict]:
    head_params = list(model.head.parameters()) + list(model.pool.parameters())
    groups = [{"params": head_params, "lr": head_lr}]
    if encoder_lr > 0:
        groups.append({"params": list(model.encoder.parameters()), "lr": encoder_lr})
    return groups
