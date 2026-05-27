"""Shared utilities for config and paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root() / config_file
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abs_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

