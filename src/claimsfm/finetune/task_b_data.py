"""Task B provider pack: one sequence per provider from its Kaggle claims.

Design (SPEC §5 gives the pooling implementation as a free choice, documented
here and in the report): a provider is encoded as [CLS] followed by its
claims in date order, each claim a [VISIT]-separated span of that claim's
DX_/PX_ tokens — the same structure the encoder pretrained on, with claim
standing in for visit. The [CLS] state is the provider representation
(self-attention over claim tokens is the attention pooling). Sequences are
truncated keep-most-recent at max_len; truncation coverage is recorded in
meta and reported honestly.

Structural channels: claim_type from IP/OP file of origin; age = beneficiary
age at claim start; month index on the same 2008-01 clock as pretraining
(Kaggle claims run 2008-12..2009-12). Codes map through the pretraining
vocabulary; OOV becomes [UNK] (dx coverage measured at 99.9% in M1).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from claimsfm.etl.kaggle_tables import DX_COLS, PX_COLS
from claimsfm.tasks.provider_b import _find

log = logging.getLogger(__name__)

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def provider_claim_events(kaggle_dir: Path) -> pl.DataFrame:
    """Long-format: one row per (provider, claim) with token list + structure."""
    frames = []
    for kind, is_ip in (("inpatient", True), ("outpatient", False)):
        df = pl.read_parquet(_find(kaggle_dir, f"train_{kind}data*.parquet"))
        frames.append(
            df.select(
                pl.col("Provider"),
                pl.col("ClaimID"),
                pl.col("BeneID"),
                pl.col("ClaimStartDt").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("start"),
                (
                    pl.concat_list([("DX_" + pl.col(c)) for c in DX_COLS]).list.drop_nulls()
                    .list.concat(
                        pl.concat_list([("PX_" + pl.col(c)) for c in PX_COLS]).list.drop_nulls()
                    )
                ).alias("tokens"),
                pl.lit(1 if is_ip else 2, dtype=pl.Int8).alias("claim_type_id"),
            )
        )
    bene = pl.read_parquet(_find(kaggle_dir, "train_beneficiarydata*.parquet")).select(
        "BeneID", pl.col("DOB").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("dob")
    )
    return (
        pl.concat(frames)
        .join(bene, on="BeneID", how="left")
        .filter(pl.col("start").is_not_null())
        .with_columns(
            ((pl.col("start") - pl.col("dob")).dt.total_days() / 365.25)
            .clip(0, 110).fill_null(70).cast(pl.UInt8).alias("age"),
            ((pl.col("start").dt.year() - 2008) * 12 + pl.col("start").dt.month() - 1)
            .clip(0, 35).cast(pl.UInt8).alias("month"),
        )
        .sort("Provider", "start", "ClaimID")
    )


def build_task_b_pack(
    kaggle_dir: Path,
    features_path: Path,
    meta_path: Path,
    vocab_path: Path,
    out_dir: Path,
    max_len: int,
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

    cols = {n: [] for n in ("input_ids", "claim_type_ids", "age_years", "month_idx", "visit_ids")}
    offsets = [0]
    n_truncated = 0
    total_claims = kept_claims = 0

    for prov in sidecar["Provider"]:
        sub = grouped.get((prov,))
        ids, ctypes, ages, months, visits = [cls_id], [0], [0], [0], [0]
        n_claims = 0 if sub is None else len(sub)
        total_claims += n_claims
        if sub is not None:
            spans = []
            for claim_no, row in enumerate(sub.iter_rows(named=True), start=1):
                toks = row["tokens"] or []
                if toks:
                    spans.append({
                        "ids": [tok2id.get(t, unk) for t in toks],
                        "ct": row["claim_type_id"], "age": row["age"],
                        "mo": row["month"], "v": claim_no,
                    })
            # keep most recent claims that fit under max_len
            # (cost per claim = tokens + 1 separator; first kept claim has none)
            budget = max_len - 1
            kept = []
            for span in reversed(spans):
                if budget - (len(span["ids"]) + 1) < 0:
                    n_truncated += 1
                    break
                budget -= len(span["ids"]) + 1
                kept.append(span)
            kept.reverse()
            kept_claims += len(kept)
            for si, span in enumerate(kept):
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
        cols["input_ids"].append(np.asarray(ids, dtype=np.uint16))
        cols["claim_type_ids"].append(np.asarray(ctypes, dtype=np.uint8))
        cols["age_years"].append(np.asarray(ages, dtype=np.uint8))
        cols["month_idx"].append(np.asarray(months, dtype=np.uint8))
        cols["visit_ids"].append(np.asarray(visits, dtype=np.uint16))
        offsets.append(offsets[-1] + len(ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    for n, chunks in cols.items():
        np.save(out_dir / f"{n}.npy", np.concatenate(chunks))
    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(
        out_dir / "split.npy",
        np.array([SPLIT_CODE[s] for s in sidecar["split"]], dtype=np.uint8),
    )
    sidecar.write_parquet(out_dir / "sidecar.parquet", compression="zstd")

    meta = {
        "n_members": len(sidecar),
        "n_positions": offsets[-1],
        "max_len": max_len,
        "n_truncated_providers": n_truncated,
        "claims_total": total_claims,
        "claims_kept": kept_claims,
        "splits_sha256": got,
        "vocab_size": len(tok2id),
        "vocab_counts_hash": vocab["meta"]["counts_hash"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(
        "task B pack: %d providers, %d positions; %d truncated; claims kept %d/%d (%.1f%%)",
        len(sidecar), offsets[-1], n_truncated, kept_claims, total_claims,
        100 * kept_claims / max(1, total_claims),
    )
    return meta
