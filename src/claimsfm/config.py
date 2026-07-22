from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def data_path(cfg: dict[str, Any], tier: str) -> Path:
    return REPO_ROOT / cfg["paths"][tier]


def lock_path() -> Path:
    return REPO_ROOT / "configs" / "data.lock.yaml"


def load_lock() -> dict[str, Any]:
    p = lock_path()
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_lock(lock: dict[str, Any]) -> None:
    with open(lock_path(), "w") as f:
        yaml.safe_dump(lock, f, sort_keys=True, default_flow_style=False)
