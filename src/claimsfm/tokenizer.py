"""Tokenizer: member sequence rows -> model-ready id arrays.

Encoding flattens a member's visit structure as
[CLS] v1_tok ... [VISIT] v2_tok ... [VISIT] ... with parallel visit and
claim-type id arrays. Dates are exposed to the caller so M3 can derive
age/time features without touching the ETL. Windowing here mirrors the
sequence-builder's (used when encoding full-range rows for Task A).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

CLAIM_TYPES = {"IP": 1, "OP": 2, "RX": 3}


class ClaimsTokenizer:
    def __init__(self, vocab: dict[str, Any]):
        self.tokens: dict[str, int] = vocab["tokens"]
        self.meta = vocab["meta"]
        self.ids: dict[int, str] = {i: t for t, i in self.tokens.items()}
        self.pad_id = self.tokens["[PAD]"]
        self.unk_id = self.tokens["[UNK]"]
        self.mask_id = self.tokens["[MASK]"]
        self.cls_id = self.tokens["[CLS]"]
        self.visit_id = self.tokens["[VISIT]"]

    @classmethod
    def load(cls, path: str | Path) -> "ClaimsTokenizer":
        with open(path) as f:
            return cls(json.load(f))

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(
        self,
        tokens: list[str],
        dates: list[dt.date],
        claim_types: list[str],
        visit_ids: list[int],
        window: tuple[int, int] | None = None,
        max_len: int | None = None,
    ) -> dict[str, list[int]]:
        input_ids = [self.cls_id]
        out_visits = [0]
        out_types = [0]

        prev_visit = None
        for tok, date, ctype, visit in zip(tokens, dates, claim_types, visit_ids):
            if window and not (window[0] <= date.year <= window[1]):
                continue
            if prev_visit is not None and visit != prev_visit:
                input_ids.append(self.visit_id)
                out_visits.append(visit)
                out_types.append(0)
            input_ids.append(self.tokens.get(tok, self.unk_id))
            out_visits.append(visit)
            out_types.append(CLAIM_TYPES[ctype])
            prev_visit = visit

        if max_len is not None and len(input_ids) > max_len:
            # keep the most recent history; the [CLS] anchor stays
            input_ids = [self.cls_id] + input_ids[-(max_len - 1):]
            out_visits = [0] + out_visits[-(max_len - 1):]
            out_types = [0] + out_types[-(max_len - 1):]

        return {"input_ids": input_ids, "visit_ids": out_visits, "claim_type_ids": out_types}

    def decode(self, input_ids: list[int]) -> list[str]:
        return [self.ids[i] for i in input_ids]
