"""Shared helpers: config loading, seeding, IO."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_latex_row(values: list, path: str | os.PathLike) -> None:
    """Append a '&'-joined LaTeX table row to a file (handy for filling [XX] cells)."""
    ensure_dir(Path(path).parent)
    with open(path, "a") as fh:
        fh.write(" & ".join(str(v) for v in values) + r" \\" + "\n")
