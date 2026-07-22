"""Fine-tuning head over the pretrained claims encoder.

Prediction reads the final-layer [CLS] hidden state (design choice, stated in
the report). Modes: `probe` (frozen pretrained encoder, head only), `full`
(pretrained init, discriminative LRs), `scratch` (identical architecture,
fresh init, same budget — the transfer ablation's control arm).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from claimsfm.model.encoder import ClaimsEncoder, EncoderConfig

MODES = ("probe", "full", "scratch")


class ClaimsFinetune(nn.Module):
    def __init__(self, encoder: ClaimsEncoder, head_dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(encoder.cfg.d_model, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(
            batch["input_ids"], batch["claim_type_ids"], batch["age_years"],
            batch["month_idx"], batch["visit_ids"],
        )
        return self.head(self.dropout(hidden[:, 0])).squeeze(-1)


def build_model(
    mode: str, model_cfg: dict, vocab_size: int, checkpoint: Path | None, head_dropout: float
) -> ClaimsFinetune:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    encoder = ClaimsEncoder(EncoderConfig.from_dict(vocab_size, model_cfg))
    if mode in ("probe", "full"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["model"])
    model = ClaimsFinetune(encoder, head_dropout)
    if mode == "probe":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return model


def param_groups(model: ClaimsFinetune, head_lr: float, encoder_lr: float) -> list[dict]:
    groups = [{"params": list(model.head.parameters()), "lr": head_lr}]
    if encoder_lr > 0:
        groups.append({"params": list(model.encoder.parameters()), "lr": encoder_lr})
    return groups
