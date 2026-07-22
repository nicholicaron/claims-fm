"""Post-pretraining probes (SPEC §6): nearest-neighbor anchor tables, a
qualitative 2-D projection, masked-accuracy vs frequency prior, loss curves.

Usage: python scripts/probe_embeddings.py --checkpoint data/checkpoints/pretrain/best.pt
Writes reports/pretrain.md + reports/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from claimsfm.config import REPO_ROOT
from claimsfm.pretrain.data import N_SPECIALS, frequency_prior

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

# Display labels for common ICD-9 codes (readability only; unknowns shown raw).
CODE_LABELS = {
    "DX_25000": "diabetes II w/o complication", "DX_4019": "hypertension NOS",
    "DX_2724": "hyperlipidemia NEC", "DX_42731": "atrial fibrillation",
    "DX_4280": "congestive heart failure", "DX_496": "COPD NOS",
    "DX_5859": "chronic kidney disease NOS", "DX_5856": "ESRD",
    "DX_V5867": "long-term insulin use", "DX_V5861": "long-term anticoagulants",
    "DX_25001": "diabetes I w/o complication", "DX_25002": "diabetes II uncontrolled",
    "DX_3572": "diabetic polyneuropathy", "DX_36201": "diabetic retinopathy",
    "DX_2720": "hypercholesterolemia", "DX_4011": "benign hypertension",
    "DX_40390": "hypertensive CKD", "DX_5853": "CKD stage III",
    "DX_5854": "CKD stage IV", "DX_58881": "sec. hyperparathyroid (renal)",
    "DX_28521": "anemia in CKD", "DX_4254": "cardiomyopathy NEC",
    "DX_4271": "paroxysmal ventricular tach", "DX_42822": "chronic systolic HF",
    "DX_49121": "obstr. chronic bronchitis w/ exac.", "DX_49320": "chronic obstr. asthma",
    "DX_51881": "acute respiratory failure", "DX_7862": "cough", "DX_78605": "SOB",
    "DX_53081": "esophageal reflux", "DX_2859": "anemia NOS",
    "DX_2449": "hypothyroidism NOS", "DX_5939": "kidney disorder NOS",
    "DX_49390": "asthma NOS", "DX_4439": "peripheral vascular disease",
    "DX_25060": "diabetes w/ neuro manifestations", "DX_3051": "tobacco use disorder",
    "DX_41400": "coronary atherosclerosis", "DX_4299": "heart disease NOS",
    "DX_586": "renal failure NOS", "DX_5990": "urinary tract infection",
    "DX_5855": "CKD stage V", "DX_4928": "obstructive chronic bronchitis",
    "DX_5119": "pleural effusion", "DX_42789": "cardiac dysrhythmia NEC",
    "DX_4293": "cardiomegaly", "DX_4292": "cardiovascular disease NOS",
    "DX_60000": "benign prostatic hypertrophy", "DX_4779": "allergic rhinitis",
    "DX_V5866": "long-term aspirin use", "DX_V5863": "long-term antiplatelets",
    "DX_V5865": "long-term steroids", "DX_V5862": "long-term antibiotics",
    "DX_V4581": "aortocoronary bypass status", "DX_V4582": "PTCA status",
    "DX_V4501": "cardiac pacemaker status",
    "PX_3895": "venous cath for renal dialysis", "PX_3491": "thoracentesis",
    "PX_4513": "small bowel endoscopy", "PX_4525": "colonoscopy w/ biopsy",
    "PX_3324": "closed lung biopsy", "PX_3950": "angioplasty NEC",
    "PX_3995": "hemodialysis", "PX_5498": "peritoneal dialysis",
    "PX_9904": "packed cell transfusion", "PX_8154": "total knee replacement",
    "PX_3893": "venous catheterization", "PX_9671": "cont. mech. ventilation <96h",
    "PX_4516": "EGD w/ biopsy", "PX_3722": "left heart cath",
    "PX_8856": "coronary arteriography", "PX_9339": "physical therapy NEC",
}

ANCHORS = ["DX_25000", "DX_4280", "DX_496", "DX_5859", "DX_V5867", "PX_3995"]

NDC_URL = "https://www.accessdata.fda.gov/cder/ndctext.zip"


def load_ndc_names(vocab_tokens: set[str]) -> dict[str, str]:
    """RX_ token -> drug name via the FDA NDC directory (verified URL).

    DE-SynPUF NDCs are partially randomized in synthesis, so coverage is
    expected to be partial — the match rate is reported honestly either way.
    """
    import io
    import zipfile

    import httpx

    try:
        r = httpx.get(NDC_URL, timeout=120, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"NDC directory unavailable ({e}); RX neighbors stay unnamed")
        return {}
    names: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("product.txt") as f:
            header = f.readline().decode("latin-1").rstrip("\n").split("\t")
            idx = {c: i for i, c in enumerate(header)}
            for raw in f:
                parts = raw.decode("latin-1").rstrip("\n").split("\t")
                try:
                    labeler, product = parts[idx["PRODUCTNDC"]].split("-")
                    ndc9 = labeler.zfill(5) + product.zfill(4)
                    tok = f"RX_{ndc9}"
                    if tok in vocab_tokens and tok not in names:
                        name = parts[idx["PROPRIETARYNAME"]] or parts[idx["NONPROPRIETARYNAME"]]
                        names[tok] = name.strip().title()[:40]
                except (ValueError, IndexError):
                    continue
    print(f"NDC names matched for {len(names):,} RX tokens")
    return names

ICD9_CHAPTERS = [
    (1, 139, "infectious"), (140, 239, "neoplasms"), (240, 279, "endocrine/metabolic"),
    (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous/sense"),
    (390, 459, "circulatory"), (460, 519, "respiratory"), (520, 579, "digestive"),
    (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"), (760, 779, "perinatal"),
    (780, 799, "symptoms/ill-defined"), (800, 999, "injury/poisoning"),
]


def load_embeddings(ckpt_path: Path) -> tuple[np.ndarray, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"]
    key = next(k for k in sd if k.endswith("tok.weight"))  # torch.compile prefixes
    return sd[key].float().numpy(), state["config"]


def label(tok: str) -> str:
    return f"{tok} ({CODE_LABELS[tok]})" if tok in CODE_LABELS else tok


def pick_rx_anchor(ndc_names: dict[str, str], counts: pl.DataFrame) -> str | None:
    """Highest-frequency RX token whose FDA name mentions insulin."""
    insulin = {t for t, n in ndc_names.items() if "insulin" in n.lower()}
    if not insulin:
        return None
    ranked = counts.filter(pl.col("token").is_in(list(insulin)))
    return ranked["token"][0] if ranked.height else None


def dx_chapter(tok: str) -> str:
    code = tok.removeprefix("DX_")
    if code.startswith("V"):
        return "V codes"
    if code.startswith("E"):
        return "E codes"
    try:
        n = int(code[:3])
    except ValueError:
        return "other"
    for lo, hi, name in ICD9_CHAPTERS:
        if lo <= n <= hi:
            return name
    return "other"


def neighbor_table(
    emb: np.ndarray,
    id2tok: dict[int, str],
    tok2id: dict[str, int],
    anchors: list[str],
    ndc_names: dict[str, str],
    k: int = 8,
) -> list[str]:
    def name(tok: str) -> str:
        if tok in ndc_names:
            return f"{tok} ({ndc_names[tok]})"
        return label(tok)

    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    lines = []
    for anchor in anchors:
        if anchor not in tok2id:
            lines.append(f"**{anchor}** — not in vocabulary (skipped)\n")
            continue
        sims = normed @ normed[tok2id[anchor]]
        order = np.argsort(-sims)
        lines.append(f"**{name(anchor)}**\n")
        lines.append("| Neighbor | Cosine |")
        lines.append("|---|---|")
        shown = 0
        for j in order:
            if j < N_SPECIALS or j == tok2id[anchor]:
                continue
            lines.append(f"| {name(id2tok[j])} | {sims[j]:.3f} |")
            shown += 1
            if shown == k:
                break
        lines.append("")
    return lines


def pca_figure(emb: np.ndarray, id2tok: dict[int, str], counts: pl.DataFrame, out: Path) -> None:
    top_dx = (
        counts.filter(pl.col("token").str.starts_with("DX_")).head(3000)["token"].to_list()
    )
    tok2id = {t: i for i, t in id2tok.items()}
    ids = [tok2id[t] for t in top_dx if t in tok2id]
    X = emb[ids]
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:2].T
    chapters = [dx_chapter(id2tok[i]) for i in ids]
    uniq = sorted(set(chapters))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, ch in enumerate(uniq):
        mask = np.array([c == ch for c in chapters])
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6, color=cmap(ci % 20), label=ch)
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    ax.set(title="Token-embedding PCA — top 3,000 dx codes by frequency\n(qualitative; 2 of 320 dims)",
           xlabel="PC1", ylabel="PC2")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def loss_curves(metrics_path: Path, out: Path) -> dict:
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*train), lw=1, alpha=0.7, label="train MLM loss")
    if val:
        ax.plot(*zip(*val), "o-", lw=2, label="val MLM loss")
    ax.set(xlabel="step", ylabel="masked-code CE loss", title="Pretraining loss")
    ax.legend()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    last_val = {k: v for r in rows if "val_loss" in r for k, v in r.items()}
    return last_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/checkpoints/pretrain/best.pt")
    parser.add_argument("--metrics", default="data/checkpoints/pretrain/metrics.jsonl")
    args = parser.parse_args()

    emb, cfg = load_embeddings(REPO_ROOT / args.checkpoint)
    with open(REPO_ROOT / "data/processed/vocab.json") as f:
        tok2id = json.load(f)["tokens"]
    id2tok = {i: t for t, i in tok2id.items()}
    counts = pl.read_parquet(REPO_ROOT / "data/processed/token_counts.parquet")

    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    val = loss_curves(REPO_ROOT / args.metrics, fig_dir / "pretrain_loss.png")
    pca_figure(emb, id2tok, counts, fig_dir / "pretrain_dx_pca.png")
    prior = frequency_prior(counts)

    ndc_names = load_ndc_names(set(tok2id))
    anchors = list(ANCHORS)
    rx_anchor = pick_rx_anchor(ndc_names, counts)
    if rx_anchor:
        anchors.append(rx_anchor)

    lines = [
        "# Pretraining report (M3)",
        "",
        f"Encoder: {cfg['model']['n_layers']}L × d{cfg['model']['d_model']}, "
        f"vocab {len(tok2id):,}, masked-code modeling on "
        "samples 3–7 (517,390 members; 2% member-level MLM validation holdout).",
        "",
        "## Masked-code accuracy vs frequency prior",
        "",
        "| Scope | Model top-1 | Frequency-prior top-1 |",
        "|---|---|---|",
        f"| overall | {val.get('val_masked_acc', float('nan')):.1%} | {prior['overall']:.2%} |",
        f"| dx | {val.get('val_masked_acc_dx', float('nan')):.1%} | {prior['dx']:.2%} |",
        f"| px | {val.get('val_masked_acc_px', float('nan')):.1%} | {prior['px']:.2%} |",
        f"| rx | {val.get('val_masked_acc_rx', float('nan')):.1%} | {prior['rx']:.2%} |",
        "",
        f"Final val MLM loss: {val.get('val_loss', float('nan')):.4f}.",
        "The frequency prior is the accuracy of always predicting the modal code",
        "(per scope) — the SPEC acceptance bar is clearing it meaningfully.",
        "",
        "![loss](figures/pretrain_loss.png)",
        "",
        "## Embedding-space anchor probes (cosine nearest neighbors)",
        "",
        "Code descriptions are display labels for common ICD-9 codes; judgment of",
        "clinical coherence is qualitative by design (SPEC §6).",
        "",
        f"RX naming: FDA NDC directory matched {len(ndc_names):,} of the "
        f"{sum(1 for t in tok2id if t.startswith('RX_')):,} RX tokens "
        "(DE-SynPUF randomizes NDCs in synthesis; unmatched neighbors shown raw).",
        "",
        *neighbor_table(emb, id2tok, tok2id, anchors, ndc_names),
        "## 2-D projection (qualitative)",
        "",
        "![pca](figures/pretrain_dx_pca.png)",
        "",
        "PCA of the top 3,000 dx-code embeddings colored by ICD-9 chapter — two of",
        "320 dimensions; treat as a sniff test, not evidence.",
    ]
    out = REPO_ROOT / "reports" / "pretrain.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
