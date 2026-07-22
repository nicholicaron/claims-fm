"""Task B provider dataset: features, splits, and the beneficiary-overlap
caveat quantification (SPEC §5 requires reporting it, not pretending it away).

All features are computed strictly within-provider (that provider's claims and
the beneficiary rows linked to them) — no cross-provider signals, so a
provider's feature vector is invariant to the rest of the dataset (tested).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

DX_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PX_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
CHRONIC = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]


def _find(dir: Path, pattern: str) -> Path:
    matches = sorted(dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"{dir}/{pattern}: expected 1 match, got {matches}")
    return matches[0]


def load_claims(kaggle_dir: Path) -> pl.DataFrame:
    """Train IP+OP claims, typed, one frame with is_ip flag."""
    ip = pl.read_parquet(_find(kaggle_dir, "train_inpatientdata*.parquet"))
    op = pl.read_parquet(_find(kaggle_dir, "train_outpatientdata*.parquet"))

    def typed(df: pl.DataFrame, is_ip: bool) -> pl.DataFrame:
        cols = [
            pl.col("Provider"),
            pl.col("BeneID"),
            pl.col("ClaimID"),
            pl.col("ClaimStartDt").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("start"),
            pl.col("ClaimEndDt").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("end"),
            pl.col("InscClaimAmtReimbursed").cast(pl.Float64, strict=False).alias("reimb"),
            pl.col("DeductibleAmtPaid").cast(pl.Float64, strict=False).alias("deductible"),
            pl.col("AttendingPhysician").replace("NA", None).alias("attending"),
            pl.col("OperatingPhysician").replace("NA", None).alias("operating"),
            pl.concat_str([pl.col(c).fill_null("") for c in DX_COLS], separator="|").alias("dx_concat"),
            pl.concat_list([pl.col(c) for c in DX_COLS]).list.drop_nulls().alias("dx_list"),
            pl.concat_list([pl.col(c) for c in PX_COLS]).list.drop_nulls().alias("px_list"),
            pl.lit(is_ip).alias("is_ip"),
        ]
        if is_ip:
            cols += [
                pl.col("AdmissionDt").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("admit"),
                pl.col("DischargeDt").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("discharge"),
                pl.col("DiagnosisGroupCode").replace("NA", None).alias("drg"),
            ]
        else:
            cols += [
                pl.lit(None, dtype=pl.Date).alias("admit"),
                pl.lit(None, dtype=pl.Date).alias("discharge"),
                pl.lit(None, dtype=pl.String).alias("drg"),
            ]
        return df.select(cols)

    return pl.concat([typed(ip, True), typed(op, False)])


def load_bene(kaggle_dir: Path) -> pl.DataFrame:
    df = pl.read_parquet(_find(kaggle_dir, "train_beneficiarydata*.parquet"))
    return df.select(
        pl.col("BeneID"),
        pl.col("DOB").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("dob"),
        pl.col("DOD").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("dod"),
        sum((pl.col(c) == "1").cast(pl.Int8) for c in CHRONIC).alias("chronic_count"),
        (pl.col("RenalDiseaseIndicator") == "Y").cast(pl.Int8).alias("renal"),
        (
            pl.col("IPAnnualReimbursementAmt").cast(pl.Float64, strict=False)
            + pl.col("OPAnnualReimbursementAmt").cast(pl.Float64, strict=False)
        ).alias("annual_reimb"),
    )


def provider_features(claims: pl.DataFrame, bene: pl.DataFrame) -> pl.DataFrame:
    cl = claims.join(bene, on="BeneID", how="left").with_columns(
        ((pl.col("start") - pl.col("dob")).dt.total_days() / 365.25).alias("bene_age"),
        (pl.col("discharge") - pl.col("admit")).dt.total_days().alias("los"),
        pl.col("dx_list").list.len().alias("n_dx"),
        pl.col("start").dt.strftime("%Y-%m").alias("month"),
        pl.col("dx_list").list.sort().list.join("|").alias("dx_key"),
    )

    per_claim_dup = (
        cl.group_by("Provider", "BeneID", "dx_key")
        .agg(pl.len().alias("n"))
        .group_by("Provider")
        .agg(
            ((pl.col("n") - 1).sum() / pl.col("n").sum()).alias("dup_claim_rate"),
        )
    )
    monthly = (
        cl.group_by("Provider", "month")
        .agg(pl.len().alias("n"))
        .group_by("Provider")
        .agg(
            (pl.col("n").max() / pl.col("n").mean()).alias("monthly_burstiness"),
            pl.col("month").n_unique().alias("active_months"),
        )
    )
    top_dx = (
        cl.explode("dx_list")
        .drop_nulls("dx_list")
        .group_by("Provider", "dx_list")
        .agg(pl.len().alias("n"))
        .group_by("Provider")
        .agg((pl.col("n").max() / pl.col("n").sum()).alias("top_dx_concentration"))
    )

    base = cl.group_by("Provider").agg(
        pl.len().alias("n_claims"),
        pl.col("is_ip").sum().alias("n_ip_claims"),
        pl.col("BeneID").n_unique().alias("n_benes"),
        (pl.len() / pl.col("BeneID").n_unique()).alias("claims_per_bene"),
        pl.col("reimb").sum().alias("reimb_sum"),
        pl.col("reimb").mean().alias("reimb_mean"),
        pl.col("reimb").median().alias("reimb_median"),
        pl.col("reimb").quantile(0.9).alias("reimb_p90"),
        pl.col("reimb").max().alias("reimb_max"),
        (pl.col("reimb") == 0).mean().alias("zero_reimb_rate"),
        pl.col("reimb").filter(pl.col("is_ip")).sum().alias("reimb_ip_sum"),
        pl.col("reimb").filter(~pl.col("is_ip")).mean().alias("reimb_op_mean"),
        pl.col("deductible").mean().alias("deductible_mean"),
        pl.col("los").mean().alias("los_mean"),
        pl.col("n_dx").mean().alias("dx_per_claim"),
        pl.col("dx_list").explode().n_unique().alias("n_distinct_dx"),
        pl.col("px_list").explode().n_unique().alias("n_distinct_px"),
        pl.col("drg").n_unique().alias("n_distinct_drg"),
        pl.col("attending").n_unique().alias("n_attending"),
        pl.col("operating").n_unique().alias("n_operating"),
        pl.col("attending").is_null().mean().alias("attending_missing_rate"),
        (pl.col("attending") == pl.col("operating")).mean().alias("attending_eq_operating"),
        pl.col("bene_age").mean().alias("bene_age_mean"),
        pl.col("dod").is_not_null().mean().alias("bene_deceased_share"),
        pl.col("chronic_count").mean().alias("bene_chronic_mean"),
        pl.col("renal").mean().alias("bene_renal_share"),
        pl.col("annual_reimb").mean().alias("bene_annual_reimb_mean"),
    )
    out = (
        base.join(per_claim_dup, on="Provider", how="left")
        .join(monthly, on="Provider", how="left")
        .join(top_dx, on="Provider", how="left")
        .with_columns(
            (pl.col("n_ip_claims") / pl.col("n_claims")).alias("ip_claim_share"),
            (pl.col("reimb_ip_sum") / (pl.col("reimb_sum") + 1e-9)).alias("ip_revenue_share"),
            (pl.col("n_claims") / pl.col("n_attending").clip(1)).alias("claims_per_physician"),
        )
        .sort("Provider")
    )
    return out


def assign_splits(labels: pl.DataFrame, bl: dict[str, Any]) -> pl.DataFrame:
    seed = bl["seed"]
    fr = bl["splits"]
    y = (labels["PotentialFraud"] == "Yes").cast(pl.Int8).to_numpy()
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=1 - fr["train"], stratify=y, random_state=seed)
    val, test = train_test_split(
        rest, test_size=fr["test"] / (fr["val"] + fr["test"]), stratify=y[rest], random_state=seed
    )
    split = np.empty(len(labels), dtype=object)
    split[tr], split[val], split[test] = "train", "val", "test"
    return labels.with_columns(
        pl.Series("split", split, dtype=pl.String),
        pl.Series("label", y, dtype=pl.Int8),
    )


def bene_overlap_stats(claims: pl.DataFrame, labeled: pl.DataFrame) -> dict[str, float]:
    per_bene = claims.group_by("BeneID").agg(pl.col("Provider").n_unique().alias("n_prov"))
    multi = (per_bene["n_prov"] > 1).mean()
    train_prov = set(labeled.filter(pl.col("split") == "train")["Provider"])
    train_benes = set(claims.filter(pl.col("Provider").is_in(train_prov))["BeneID"])
    out = {"benes_with_multiple_providers": float(multi)}
    for split in ("val", "test"):
        prov = set(labeled.filter(pl.col("split") == split)["Provider"])
        sub = claims.filter(pl.col("Provider").is_in(prov))
        out[f"{split}_claims_with_train_bene"] = float(
            sub.select(pl.col("BeneID").is_in(train_benes).mean()).item()
        )
    return out


def splits_hash(df: pl.DataFrame) -> str:
    key = "\n".join(f"{p}:{s}" for p, s in df.select("Provider", "split").iter_rows())
    return hashlib.sha256(key.encode()).hexdigest()
