from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: str):
    with open(Path(path).expanduser(), "r") as f:
        return yaml.safe_load(f) or {}


def get_cfg(cfg: dict, keys: tuple[str, ...], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
