"""Code vocabulary built exclusively from pretraining-role sequences.

Eval samples (1-2) never touch the vocabulary — the clean-split rule extends
to token statistics, and the build refuses any input whose role isn't in
`vocab_source_roles`. Metadata records provenance so tests can re-verify.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from claimsfm.config import data_path

log = logging.getLogger(__name__)

SPECIALS = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[VISIT]"]


def token_counts(cfg: dict[str, Any]) -> pl.DataFrame:
    """Frequency table of all tokens across pretraining sequences (cached)."""
    out = data_path(cfg, "processed") / "token_counts.parquet"
    if out.exists():
        return pl.read_parquet(out)

    roles = cfg["etl"]["vocab_source_roles"]
    frames = []
    for role in roles:
        seq_path = data_path(cfg, "processed") / f"sequences_{role}.parquet"
        lf = pl.scan_parquet(seq_path)
        bad = lf.filter(~pl.col("role").is_in(roles)).select(pl.len()).collect().item()
        if bad:
            raise RuntimeError(f"{seq_path} contains {bad} rows outside roles {roles}")
        frames.append(lf.select(pl.col("tokens").explode().alias("token")))

    counts = (
        pl.concat(frames)
        .group_by("token")
        .agg(pl.len().alias("count"))
        .sort("count", "token", descending=[True, False])
        .collect(engine="streaming")
    )
    counts.write_parquet(out, compression="zstd")
    return counts


def build_vocab(cfg: dict[str, Any]) -> Path:
    counts = token_counts(cfg)
    min_count = cfg["etl"]["vocab_min_count"]
    kept = counts.filter(pl.col("count") >= min_count)

    # DE-SynPUF synthesizes NDCs with a near-flat frequency profile (~121k
    # distinct NDC-9 after truncation), which would blow the embedding budget
    # for signal that cannot transfer (Kaggle has no drug data). RX is capped
    # at the top-K by frequency; DX/PX are kept in full above the floor.
    rx_top_k = cfg["etl"].get("rx_top_k")
    if rx_top_k:
        rx = kept.filter(pl.col("token").str.starts_with("RX_")).head(rx_top_k)
        kept = pl.concat([kept.filter(~pl.col("token").str.starts_with("RX_")), rx])

    sweep = {
        mc: int(counts.filter(pl.col("count") >= mc).height)
        for mc in cfg["etl"]["vocab_min_count_sweep"]
    }

    tokens = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok in kept["token"]:
        tokens[tok] = len(tokens)

    counts_hash = hashlib.sha256(
        "\n".join(f"{t}:{c}" for t, c in counts.iter_rows()).encode()
    ).hexdigest()[:16]

    vocab = {
        "specials": SPECIALS,
        "tokens": tokens,
        "meta": {
            "min_count": min_count,
            "rx_top_k": rx_top_k,
            "min_count_sweep_sizes": sweep,
            "source_roles": cfg["etl"]["vocab_source_roles"],
            "source_samples": sorted(
                int(s)
                for s, r in cfg["synpuf"]["samples"].items()
                if r in cfg["etl"]["vocab_source_roles"]
            ),
            "n_distinct_tokens_seen": counts.height,
            "vocab_size": len(tokens),
            "counts_hash": counts_hash,
            "built_at": dt.date.today().isoformat(),
        },
    }

    out = data_path(cfg, "processed") / "vocab.json"
    with open(out, "w") as f:
        json.dump(vocab, f, indent=1)
    log.info(
        "vocab: %d tokens (floor %d; %d distinct seen; sweep %s)",
        len(tokens), min_count, counts.height, sweep,
    )
    return out
