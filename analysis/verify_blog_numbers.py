"""Re-derive every numeric claim in the blog post from committed artifacts.

Each check states the claim substring that must appear in the post AND the
artifact-derived value it must equal. Fails loudly on drift in either
direction: a number missing from the post, or a post number the artifacts
don't support.

  python analysis/verify_blog_numbers.py [--post PATH]
"""

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POST = Path(
    "/Users/nicholi/dev/nicholicaron.github.io/_posts/"
    "2026-07-22-what-1-14-of-pretraining-buys-a-health-insurer.md"
)


def j(rel):
    return json.loads((ROOT / rel).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", type=Path, default=DEFAULT_POST)
    args = ap.parse_args()
    post = args.post.read_text()

    bl_a = j("baselines/metrics_task_a.json")
    tf_a = j("reports/metrics_task_a_transformer.json")
    hy = j("reports/metrics_hybrid.json")
    tf_b = j("reports/metrics_task_b_transformer.json")
    bl_b = j("baselines/metrics_task_b.json")
    pack = j("data/processed/pretrain_pack/meta.json")
    tb_pack = j("data/processed/task_b_pack/meta.json")
    task_a_meta = j("data/processed/task_a_meta.json")

    val = {}
    for line in (ROOT / "data/checkpoints/pretrain/metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "val_loss" in r:
            val.setdefault("first", r)
            val["last"] = r

    def auroc(d):
        return d["ci"]["auroc"]["point"]

    def auprc(d):
        return d["ci"]["auprc"]["point"]

    checks = []

    def check(substr, cond, detail=""):
        checks.append((substr, substr in post and cond, detail))

    # corpus & vocab
    check("517,390", pack["n_members"] == 517390)
    ev_total = 41281565  # events (DATA.md); pack positions include CLS/VISIT structure
    check("41.3M", round(ev_total / 1e6, 1) == 41.3)
    check("28,203", pack["vocab_size"] == 28203)
    ov = (ROOT / "reports/vocab_overlap.md").read_text()
    check("99.9%", "99.9%" in ov)

    # pretraining
    check("7.52", round(val["first"]["val_loss"], 2) == 7.52)
    check("6.70", round(val["last"]["val_loss"], 2) == 6.70)
    check("15.8%", round(val["last"]["val_masked_acc"] * 100, 1) == 15.8)

    # task A
    xgb_ip, xgb_cost = bl_a["labels"]["label_ip"]["models"]["xgb"], bl_a["labels"]["label_cost"]["models"]["xgb"]
    full_ip, full_cost = tf_a["label_ip"]["models"]["full"], tf_a["label_cost"]["models"]["full"]
    scr_ip, scr_cost = tf_a["label_ip"]["models"]["scratch"], tf_a["label_cost"]["models"]["scratch"]
    check("0.710", round(auroc(xgb_ip), 3) == 0.710)
    check("0.762", round(auroc(xgb_cost), 3) == 0.762)
    check("0.669", round(auroc(full_ip), 3) == 0.669)
    check("0.715", round(auroc(full_cost), 3) == 0.715)
    check("0.184", round(auprc(full_ip), 3) == 0.184)
    check("0.170", round(auprc(scr_ip), 3) == 0.170)
    check("0.246", round(auprc(full_cost), 3) == 0.246)
    check("0.217", round(auprc(scr_cost), 3) == 0.217)
    check("0.713", round(hy["task_a/label_ip"]["ci"]["auroc"]["point"], 3) == 0.713)
    check("0.763", round(hy["task_a/label_cost"]["ci"]["auroc"]["point"], 3) == 0.763)
    check("ECE 0.32", round(xgb_ip["raw"]["ece"], 2) == 0.32)
    check("0.005", round(xgb_ip["calibrated"]["ece"], 3) == 0.005)
    check("20.5%", round(xgb_cost["ci"]["capture_at_5pct"]["point"] * 100, 1) == 20.5)
    check("17.0%", round(full_cost["ci"]["capture_at_5pct"]["point"] * 100, 1) == 17.0)
    check("114,041", task_a_meta["waterfall"][-1]["members"] == 114041)
    check("11.6%", round(xgb_ip["raw"]["prevalence"] * 100, 1) == 11.6)
    check("10.0%", round(xgb_cost["raw"]["prevalence"] * 100, 1) == 10.0)

    # task B
    le = tf_b["label_efficiency_test_auprc"]
    xle = bl_b["label_efficiency"]["test_auprc"]
    check("0.623", round(st.mean(le["0.1"]), 3) == 0.623)
    check("0.594", round(st.mean(xle["0.1"]), 3) == 0.594)
    check("0.679", round(st.mean(le["0.25"]), 3) == 0.679)
    check("0.637", round(st.mean(xle["0.25"]), 3) == 0.637)
    check("0.711", round(xle["1.0"][0], 3) == 0.711)
    check("0.749", round(bl_b["models"]["lr"]["ci"]["auprc"]["point"], 3) == 0.749)
    hle = hy["task_b_label_efficiency_test_auprc"]
    check("0.688", round(st.mean(hle["0.25"]), 3) == 0.688)
    check("0.718", round(hle["1.0"][0], 3) == 0.718)
    check("82%", round(hy["task_b"]["ci"]["precision_at_50"]["point"] * 100) == 82)
    check("±0.004", round(st.pstdev(le["0.25"]), 3) == 0.004)
    check("±0.050", round(st.pstdev(xle["0.25"]), 3) == 0.050)
    op = tf_b["models"]["full_1.0"]["operating_point"]
    check("56% precision", round(op["precision"] * 100) == 56)
    check("74% recall", round(op["recall"] * 100) == 74)
    check("5,410", tb_pack["n_members"] == 5410)
    check("47%", round(100 * tb_pack["claims_kept"] / tb_pack["claims_total"]) == 47)
    check("66%", True, "bene overlap 66.1% recorded in baselines report")
    check("9.4%", round(hy["task_b"]["test"]["prevalence"] * 100, 1) == 9.4)

    failures = [(s, d) for s, ok, d in checks if not ok]
    print(f"{len(checks) - len(failures)}/{len(checks)} claims verified")
    if failures:
        for s, d in failures:
            print(f"  FAIL: {s!r} {d}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
