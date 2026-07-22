"""Manifest-driven DE-SynPUF downloader.

For each (sample, file type): resolve a working URL from the candidate list
(HEAD), stream-download the zip, sha256 it, extract the CSV, count rows and
distinct beneficiary IDs, and record everything in configs/data.lock.yaml.
Re-runs skip files whose zip is present with a matching checksum.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from claimsfm.config import data_path, load_lock, save_lock

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
BENE_ID = "DESYNPUF_ID"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def resolve_url(candidates: list[str], client: httpx.Client) -> str | None:
    for url in candidates:
        try:
            r = client.head(url, follow_redirects=True, timeout=30)
            if r.status_code == 200:
                return url
        except httpx.HTTPError:
            continue
    return None


def download(url: str, dest: Path, client: httpx.Client, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with client.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return
        except httpx.HTTPError as e:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def extract_csv(zip_path: Path, dest_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"{zip_path.name}: expected exactly one CSV, got {csvs}")
        z.extract(csvs[0], dest_dir)
    return dest_dir / csvs[0]


def csv_stats(csv_path: Path) -> dict[str, Any]:
    lf = pl.scan_csv(csv_path, infer_schema_length=0)
    cols = lf.collect_schema().names()
    stats: dict[str, Any] = {
        "rows": lf.select(pl.len()).collect().item(),
        "columns": len(cols),
    }
    if BENE_ID in cols:
        stats["distinct_members"] = lf.select(pl.col(BENE_ID).n_unique()).collect().item()
    return stats


def candidates_for(cfg: dict[str, Any], sample: str, file_type: str) -> list[str]:
    sp = cfg["synpuf"]
    override = sp.get("overrides", {}).get(f"{sample}:{file_type}")
    if override:
        return list(override)
    hosts = sp["hosts"]
    return [t.format(n=sample, **hosts) for t in sp["url_candidates"][file_type]]


def download_synpuf(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_root = data_path(cfg, "raw") / "synpuf"
    lock = load_lock()
    lock.setdefault("synpuf", {})
    failures: list[str] = []

    with httpx.Client(headers=UA) as client:
        for sample, role in cfg["synpuf"]["samples"].items():
            sample_dir = raw_root / f"sample_{int(sample):02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for file_type in cfg["synpuf"]["file_types"]:
                key = f"sample_{int(sample):02d}:{file_type}"
                entry = lock["synpuf"].get(key, {})
                zip_path = sample_dir / f"{file_type}.zip"

                if zip_path.exists() and entry.get("sha256") == sha256_file(zip_path):
                    log.info("skip %s (already downloaded, checksum ok)", key)
                    continue

                cands = candidates_for(cfg, sample, file_type)
                url = resolve_url(cands, client)
                if url is None:
                    failures.append(f"{key}: no candidate URL responded 200: {cands}")
                    continue
                if url != cands[0]:
                    log.warning("%s: primary URL failed, using fallback %s", key, url)

                log.info("downloading %s from %s", key, url)
                download(url, zip_path, client)
                csv_path = extract_csv(zip_path, sample_dir)
                stats = csv_stats(csv_path)

                lock["synpuf"][key] = {
                    "role": role,
                    "url": url,
                    "fallback_used": url != cands[0],
                    "zip_bytes": zip_path.stat().st_size,
                    "sha256": sha256_file(zip_path),
                    "csv_file": csv_path.name,
                    "downloaded_at": dt.date.today().isoformat(),
                    **stats,
                }
                save_lock(lock)

    if failures:
        raise RuntimeError("download failures:\n" + "\n".join(failures))
    return lock
