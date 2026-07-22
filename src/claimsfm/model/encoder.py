"""Claims encoder: a small pre-LN transformer over medical-code sequences.

Input embeddings are summed (BEHRT/Med-BERT style): token + claim-type +
age-at-event + study-month + absolute position + visit parity. The MLM output
head is weight-tied to the token embedding — at a 28k vocab the embedding
table dominates the parameter budget, and tying halves it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1280
    dropout: float = 0.1
    max_positions: int = 512
    n_claim_types: int = 4
    n_ages: int = 112
    n_months: int = 37
    pad_id: int = 0

    @classmethod
    def from_dict(cls, vocab_size: int, d: dict) -> "EncoderConfig":
        return cls(vocab_size=vocab_size, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "vocab_size"})


class Block(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.attn_dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        # bool mask: True = may attend
        attn_mask = key_padding[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class ClaimsEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.claim_type = nn.Embedding(cfg.n_claim_types, cfg.d_model)
        self.age = nn.Embedding(cfg.n_ages, cfg.d_model)
        self.month = nn.Embedding(cfg.n_months, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_positions, cfg.d_model)
        self.visit_parity = nn.Embedding(2, cfg.d_model)
        self.emb_norm = nn.LayerNorm(cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        claim_type_ids: torch.Tensor,
        age_ids: torch.Tensor,
        month_ids: torch.Tensor,
        visit_ids: torch.Tensor,
    ) -> torch.Tensor:
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device)
        x = (
            self.tok(input_ids)
            + self.claim_type(claim_type_ids)
            + self.age(age_ids)
            + self.month(month_ids)
            + self.pos(pos)[None]
            + self.visit_parity(visit_ids.long() % 2)
        )
        x = self.emb_dropout(self.emb_norm(x))
        key_padding = input_ids != self.cfg.pad_id
        for block in self.blocks:
            x = block(x, key_padding)
        return self.final_norm(x)

    def mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.tok.weight.T + self.mlm_bias

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
