"""Load and merge YAML configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml from the aegis package directory unless path is given."""
    if path is None:
        base = Path(__file__).resolve().parent.parent
        path = str(base / "config.yaml")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return deepcopy(cfg)


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge nested dicts (one level deep)."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
