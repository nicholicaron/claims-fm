"""Emit theme-adaptive inline SVG charts for the claims-fm blog post.

Colors are the blog's CSS variables (with light-mode hex fallbacks) so figures
flip with the site's light/dark toggle; `currentColor` (axes/labels) tracks
body text. Every number is read from the repo's committed metrics artifacts —
the same JSONs the frozen reports cite — so charts are ground truth, not
eyeballed. Chart engine shared with the ft-diloco post tooling.

  python analysis/blog_svg.py --fig money       --out fig_money.svg
  python analysis/blog_svg.py --fig task_a      --out fig_task_a.svg
  python analysis/blog_svg.py --fig calibration --out fig_calibration.svg
  python analysis/blog_svg.py --fig pretrain    --out fig_pretrain.svg
  python analysis/blog_svg.py --fig capture     --out fig_capture.svg
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRIMARY = "var(--primary,#94452b)"
ERROR = "var(--error,#a64542)"
PRIMARY_C = "var(--primary-container,#fceee9)"
SURF = "var(--surface-container,#f3f0eb)"
AX = "currentColor"
GREEN = "#2f9e44"  # a hue distinct from the warm primary/error, readable on both themes
BLUE = "#3b7dd8"   # second neutral-safe accent for 4-series charts


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    """Minimal SVG chart canvas: data->pixel mapping + axis/label/shape primitives."""

    def __init__(self, w=720, h=380, ml=64, mr=22, mt=34, mb=52):
        self.w, self.h = w, h
        self.x0, self.x1 = ml, w - mr
        self.y0, self.y1 = mt, h - mb  # y0 top, y1 bottom (pixels)
        self.els = []
        self.xlog = self.ylog = False
        self.xdom = (0, 1)
        self.ydom = (0, 1)

    # --- scales ---
    def setx(self, lo, hi, log=False):
        self.xlog, self.xdom = log, (lo, hi)

    def sety(self, lo, hi, log=False):
        self.ylog, self.ydom = log, (lo, hi)

    def px(self, x):
        lo, hi = self.xdom
        if self.xlog:
            x, lo, hi = math.log10(x), math.log10(lo), math.log10(hi)
        return self.x0 + (x - lo) / (hi - lo) * (self.x1 - self.x0)

    def py(self, y):
        lo, hi = self.ydom
        if self.ylog:
            y, lo, hi = math.log10(max(y, 1e-9)), math.log10(lo), math.log10(hi)
        return self.y1 - (y - lo) / (hi - lo) * (self.y1 - self.y0)

    # --- primitives ---
    def line(self, x1, y1, x2, y2, color=AX, w=1.0, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.els.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color}" stroke-width="{w}" opacity="{op}"{d}/>')

    def rect(self, x, y, w, h, fill, op=1.0, rx=2):
        self.els.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                        f'rx="{rx}" fill="{fill}" opacity="{op}"/>')

    def poly(self, pts, color, w=2.0, fill="none", op=1.0, dash=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.els.append(f'<polyline points="{p}" fill="{fill}" stroke="{color}" '
                        f'stroke-width="{w}" opacity="{op}" stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def dot(self, x, y, color, r=3.2, op=1.0):
        self.els.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" opacity="{op}"/>')

    def text(self, x, y, s, size=12, color=AX, anchor="middle", op=1.0, weight=400, italic=False):
        st_ = ' font-style="italic"' if italic else ""
        self.els.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                        f'text-anchor="{anchor}" opacity="{op}" font-weight="{weight}"{st_}>{esc(s)}</text>')

    # --- axes ---
    def yticks(self, vals, fmt=lambda v: f"{v:g}", grid=True, label=None):
        for v in vals:
            yp = self.py(v)
            if grid:
                self.line(self.x0, yp, self.x1, yp, AX, 0.8, 0.12)
            self.text(self.x0 - 8, yp + 4, fmt(v), 11, AX, "end", 0.65)
        if label:
            self.text(16, (self.y0 + self.y1) / 2, label, 12, AX, "middle", 0.8)
            self.els[-1] = self.els[-1].replace("<text ",
                f'<text transform="rotate(-90 16 {(self.y0+self.y1)/2:.0f})" ', 1)

    def xticklabels(self, xs, labels):
        for x, lab in zip(xs, labels):
            self.text(self.px(x), self.y1 + 18, lab, 11, AX, "middle", 0.7)

    def xlabel(self, s):
        self.text((self.x0 + self.x1) / 2, self.h - 10, s, 12, AX, "middle", 0.8)

    def title(self, s):
        self.text(self.x0, 18, s, 13, AX, "start", 1.0, 600)

    def svg(self):
        body = "\n  ".join(self.els)
        return (f'<svg viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg" '
                f'style="font-family:\'Inter\',sans-serif; max-width:100%; height:auto; '
                f'display:block; margin:0 auto;">\n  '
                f'{body}\n</svg>\n')


# ----------------------------- data loaders -----------------------------

def _j(rel):
    return json.loads((ROOT / rel).read_text())


def mean(xs):
    return sum(xs) / len(xs)


# ----------------------------- figures -----------------------------

def fig_money():
    """Label-efficiency: 4 series (XGB, pretrained, scratch, hybrid) with seed dots."""
    tb = _j("reports/metrics_task_b_transformer.json")
    hy = _j("reports/metrics_hybrid.json")
    bl = _j("baselines/metrics_task_b.json")
    keys = ("0.1", "0.25", "1.0")
    xs = [10, 25, 100]
    series = [
        ("XGBoost, engineered features", bl["label_efficiency"]["test_auprc"], AX, "5 4"),
        ("Transformer, from scratch", tb["label_efficiency_scratch_test_auprc"], BLUE, "7 4"),
        ("Transformer, pretrained", tb["label_efficiency_test_auprc"], PRIMARY, None),
        ("Hybrid: XGB + frozen embeddings", hy["task_b_label_efficiency_test_auprc"], GREEN, "2 3"),
    ]
    c = Chart(760, 430, ml=70, mb=56)
    c.setx(8.2, 122, log=True)
    c.sety(0.52, 0.745)
    c.yticks([0.55, 0.60, 0.65, 0.70], fmt=lambda v: f"{v:.2f}", label="test AUPRC (fraud, prevalence 9.4%)")
    c.xticklabels(xs, ["10%", "25%", "100%"])
    c.xlabel("share of labeled providers used for training")
    c.title("Provider fraud: what pretraining buys when labels are scarce")
    for name, data, color, dash in series:
        pts = []
        for x, k in zip(xs, keys):
            vals = data[k]
            for v in vals:
                c.dot(c.px(x), c.py(v), color, r=2.6, op=0.45)
            pts.append((c.px(x), c.py(mean(vals))))
        c.poly(pts, color, w=2.4, dash=dash)
        for p, k in zip(pts, keys):
            c.dot(p[0], p[1], color, r=4.0)
    # legend, upper-left inside plot
    lx, ly = c.x0 + 14, c.y0 + 12
    for i, (name, data, color, dash) in enumerate(reversed(series)):
        yy = ly + i * 18
        c.line(lx, yy, lx + 26, yy, color, 2.4, dash=dash)
        c.dot(lx + 13, yy, color, r=3.2)
        c.text(lx + 34, yy + 4, name, 11.5, AX, "start", 0.85)
    return c.svg()


def fig_task_a():
    """Member-risk AUROC dot plot with 95% CI whiskers, both labels."""
    tf = _j("reports/metrics_task_a_transformer.json")
    bl = _j("baselines/metrics_task_a.json")
    hy = _j("reports/metrics_hybrid.json")
    rows = []  # (display name, color, {label: (pt, lo, hi)})
    def ci(d):
        m = d["ci"]["auroc"]
        return (m["point"], m["ci_lo"], m["ci_hi"])
    for name, color, get in [
        ("XGBoost (engineered features)", AX, lambda lab: ci(bl["labels"][lab]["models"]["xgb"])),
        ("Hybrid: XGB + embeddings", GREEN, lambda lab: ci(hy[f"task_a/{lab}"])),
        ("Logistic regression", AX, lambda lab: ci(bl["labels"][lab]["models"]["lr"])),
        ("Pretrained, full fine-tune", PRIMARY, lambda lab: ci(tf[lab]["models"]["full"])),
        ("Pretrained, frozen probe", PRIMARY, lambda lab: ci(tf[lab]["models"]["probe"])),
        ("Transformer, from scratch", BLUE, lambda lab: ci(tf[lab]["models"]["scratch"])),
    ]:
        rows.append((name, color, {lab: get(lab) for lab in ("label_ip", "label_cost")}))

    c = Chart(780, 420, ml=228, mr=30, mt=56, mb=48)
    c.setx(0.63, 0.79)
    n = len(rows)
    row_h = (c.y1 - c.y0) / n
    c.title("Member next-year risk: AUROC with 95% CIs (test, scored once)")
    for v in (0.65, 0.70, 0.75):
        xp = c.px(v)
        c.line(xp, c.y0, xp, c.y1, AX, 0.8, 0.12)
        c.text(xp, c.y1 + 18, f"{v:.2f}", 11, AX, "middle", 0.7)
    c.xlabel("AUROC")
    # two labels per row: admission (open marker), cost (filled)
    for i, (name, color, vals) in enumerate(rows):
        yc = c.y0 + (i + 0.5) * row_h
        c.text(c.x0 - 10, yc + 4, name, 11.5, AX, "end", 0.85)
        for lab, dy, filled in (("label_ip", -7, False), ("label_cost", 7, True)):
            pt, lo, hi = vals[lab]
            yy = yc + dy
            c.line(c.px(lo), yy, c.px(hi), yy, color, 1.6, 0.75)
            if filled:
                c.dot(c.px(pt), yy, color, r=4.2)
            else:
                c.els.append(f'<circle cx="{c.px(pt):.1f}" cy="{yy:.1f}" r="4.2" fill="none" '
                             f'stroke="{color}" stroke-width="1.8"/>')
    lx = c.px(0.635)
    c.els.append(f'<circle cx="{lx:.1f}" cy="{c.y0 - 22:.1f}" r="4.2" fill="none" stroke="{AX}" stroke-width="1.8"/>')
    c.text(lx + 10, c.y0 - 18, "inpatient admission (prev 11.6%)", 11, AX, "start", 0.75)
    c.dot(lx + 230, c.y0 - 22, AX, r=4.2)
    c.text(lx + 240, c.y0 - 18, "top-decile cost (prev 10.0%)", 11, AX, "start", 0.75)
    return c.svg()


def fig_calibration():
    """Reliability before/after isotonic recalibration — XGB admission model."""
    bl = _j("baselines/metrics_task_a.json")
    m = bl["labels"]["label_ip"]["models"]["xgb"]
    panels = [("Raw model output", m["reliability_raw"], ERROR),
              ("After isotonic (fit on val)", m["reliability_calibrated"], GREEN)]
    W, H = 760, 400
    half = W // 2
    parts = []
    for pi, (title, rel, color) in enumerate(panels):
        c = Chart(half, H, ml=56 if pi == 0 else 46, mr=14, mt=48, mb=52)
        c.setx(0, 1)
        c.sety(0, 1)
        c.yticks([0, 0.25, 0.5, 0.75, 1.0], fmt=lambda v: f"{v:.2f}",
                 label="observed admission rate" if pi == 0 else None)
        c.xticklabels([0, 0.5, 1.0], ["0", "0.5", "1"])
        c.xlabel("mean predicted probability")
        c.title(title)
        c.poly([(c.px(0), c.py(0)), (c.px(1), c.py(1))], AX, 1.0, op=0.4, dash="4 4")
        pts = [(c.px(x), c.py(y)) for x, y in zip(rel["mean_predicted"], rel["fraction_positive"])]
        c.poly(pts, color, 2.2)
        for p in pts:
            c.dot(p[0], p[1], color, r=3.4)
        g = f'<g transform="translate({pi * half},0)">\n  ' + "\n  ".join(c.els) + "\n</g>"
        parts.append(g)
    body = "\n  ".join(parts)
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family:\'Inter\',sans-serif; max-width:100%; height:auto; '
            f'display:block; margin:0 auto;">\n  {body}\n</svg>\n')


def fig_pretrain():
    """Masked-code top-1 vs frequency-prior baseline, by code kind (log scale)."""
    import polars as pl
    counts = pl.read_parquet(ROOT / "data/processed/token_counts.parquet")
    total = counts["count"].sum()
    prior = {"overall": counts["count"].max() / total}
    for pre, name in (("DX_", "dx"), ("PX_", "px"), ("RX_", "rx")):
        sub = counts.filter(pl.col("token").str.starts_with(pre))
        prior[name] = sub["count"].max() / sub["count"].sum()
    val = {}
    for line in (ROOT / "data/checkpoints/pretrain/metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "val_masked_acc" in r:
            val = r
    model = {"overall": val["val_masked_acc"], "dx": val["val_masked_acc_dx"],
             "px": val["val_masked_acc_px"], "rx": val["val_masked_acc_rx"]}

    c = Chart(720, 400, ml=70, mb=56)
    groups = ["overall", "dx", "px", "rx"]
    c.setx(-0.6, 3.6)
    c.sety(0.0008, 0.6, log=True)
    c.yticks([0.001, 0.01, 0.1, 0.5], fmt=lambda v: f"{v*100:g}%",
             label="masked-code top-1 accuracy (log scale)")
    c.title("Pretraining: model vs always-guess-the-most-common-code")
    c.xlabel("code kind")
    bw = 0.28
    for i, g in enumerate(groups):
        for dx_, v, color in ((-bw / 2 - 0.03, prior[g], AX), (bw / 2 + 0.03, model[g], PRIMARY)):
            x_l = c.px(i + dx_ - bw / 2)
            x_r = c.px(i + dx_ + bw / 2)
            yv = c.py(v)
            op = 0.45 if color == AX else 1.0
            c.rect(x_l, yv, x_r - x_l, c.y1 - yv, color, op=op)
            c.text((x_l + x_r) / 2, yv - 6, f"{v*100:.1f}%" if v > 0.01 else f"{v*100:.2f}%",
                   10.5, color, "middle", 0.9, 600)
        c.text(c.px(i), c.y1 + 18, g, 11, AX, "middle", 0.7)
    lx, ly = c.x0 + 12, c.y0 + 10
    c.rect(lx, ly - 8, 14, 10, AX, op=0.45)
    c.text(lx + 20, ly + 1, "frequency prior (modal code)", 11.5, AX, "start", 0.85)
    c.rect(lx + 230, ly - 8, 14, 10, PRIMARY)
    c.text(lx + 250, ly + 1, "claims-fm encoder (val)", 11.5, AX, "start", 0.85)
    return c.svg()


def fig_capture():
    """Task A capture curves at outreach capacity: XGB vs best transformer."""
    tf = _j("reports/metrics_task_a_transformer.json")
    bl = _j("baselines/metrics_task_a.json")
    lab = "label_cost"
    xgb = bl["labels"][lab]["capture_curve_xgb"]
    best = tf[lab]["capture_curve_best"]
    c = Chart(720, 400, ml=70, mb=56)
    c.setx(0, 30)
    c.sety(0, 62)
    c.yticks([0, 20, 40, 60], fmt=lambda v: f"{v:g}%", label="share of true high-cost members captured")
    c.xticklabels([0, 10, 20, 30], ["0", "10%", "20%", "30%"])
    c.xlabel("share of members outreached, ranked by predicted risk")
    c.title("Top-decile cost: capture at outreach capacity (test)")
    c.poly([(c.px(0), c.py(0)), (c.px(30), c.py(30))], AX, 1.2, op=0.4, dash="4 4")
    c.text(c.px(24.4), c.py(21.5), "random outreach", 11, AX, "start", 0.6, italic=True)
    for name, curve, color, dash in (
        ("XGBoost", xgb, AX, "6 4"),
        ("Pretrained transformer (full fine-tune)", best, PRIMARY, None),
    ):
        pts = [(c.px(f * 100), c.py(v * 100)) for f, v in zip(curve["fracs"], curve["capture"])]
        c.poly(pts, color, 2.4, dash=dash)
    c.text(c.px(13), c.py(44), "XGBoost", 11.5, AX, "start", 0.85, weight=600)
    c.text(c.px(16.2), c.py(33), "pretrained transformer", 11.5, PRIMARY, "start", 1.0, weight=600)
    return c.svg()


FIGS = {"money": fig_money, "task_a": fig_task_a, "calibration": fig_calibration,
        "pretrain": fig_pretrain, "capture": fig_capture}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True, choices=sorted(FIGS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).write_text(FIGS[args.fig]())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
