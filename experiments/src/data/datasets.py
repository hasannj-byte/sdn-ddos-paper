"""Dataset loading and the cross-dataset feature mapping.

The proposed model uses eight canonical features (config: `features`). CICDDoS2019
already uses these column names (after whitespace cleaning). InSDN and the live
testbed capture use different names, so we map them onto the same eight columns
via INSDN_FEATURE_MAP / TESTBED_FEATURE_MAP. This is what makes the cross-dataset
evaluation an apples-to-apples comparison (Section: Cross-Dataset Evaluation).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_csv_dir(path: str, keep_cols: set[str] | None = None) -> pd.DataFrame:
    """Concatenate every CSV in a directory (or load a single CSV file).

    If `keep_cols` is given (a set of post-strip column names), only those columns
    are read from disk. This is essential for full-size runs: CICDDoS2019 has 88
    columns but we need ~10, so reading just those cuts memory ~9x.
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found at {path}")
    usecols = (lambda c: c.strip() in keep_cols) if keep_cols else None
    frames = [_clean_columns(pd.read_csv(f, low_memory=False, usecols=usecols))
              for f in files]
    return pd.concat(frames, ignore_index=True)


def basic_clean(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop inf/NaN in the selected feature columns and coerce to numeric."""
    cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(cols)
    if missing:
        raise KeyError(f"Features missing from dataframe: {sorted(missing)}")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return df


def make_labels(df: pd.DataFrame, label_col: str, benign_label: str) -> np.ndarray:
    """Map benign -> 0, everything else -> 1."""
    raw = df[label_col].astype(str).str.strip().str.upper()
    return (raw != str(benign_label).strip().upper()).astype(int).to_numpy()


def _subsample_keep_benign(df: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int):
    """Cap total rows at max_rows while KEEPING every benign (minority) row.

    Benign is rare in CICDDoS2019 (attack captures), so a uniform subsample would
    throw away the class that matters. We keep all benign and randomly sample the
    attack rows to fill the remaining budget.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    n_attack = max(0, max_rows - len(benign_idx))
    keep_attack = rng.choice(attack_idx, size=min(n_attack, len(attack_idx)), replace=False)
    keep = np.sort(np.concatenate([benign_idx, keep_attack]))
    return df.iloc[keep].reset_index(drop=True)


def load_cicddos2019(cfg: dict, with_family: bool = False):
    """Return (X, y[, family]) for CICDDoS2019.

    `family` is the original (string) attack label, used by the drift protocol to
    slice the stream by attack type. Set `data.max_rows` in config to subsample
    (benign-preserving) for a tractable run; omit/null to use everything.
    """
    keep = {f.strip() for f in cfg["features"]} | {cfg["label_column"].strip()}
    df = load_csv_dir(cfg["data"]["cicddos2019_dir"], keep_cols=keep)
    df = basic_clean(df, cfg["features"])
    y = make_labels(df, cfg["label_column"], cfg["benign_label"])
    df = _subsample_keep_benign(df, y, cfg["data"].get("max_rows"), cfg["seed"])
    y = make_labels(df, cfg["label_column"], cfg["benign_label"])
    X = df[cfg["features"]].to_numpy(dtype=np.float32)
    if with_family:
        fam = df[cfg["label_column"]].astype(str).str.strip().to_numpy()
        return X, y, fam
    return X, y


# --- Cross-dataset feature mapping -----------------------------------------
# Keys are the canonical feature names (CICDDoS2019 verbose naming); values are the
# source-dataset column names. InSDN uses the newer ABBREVIATED CICFlowMeter naming,
# verified against the downloaded InSDN header.
INSDN_FEATURE_MAP = {
    "Flow Duration": "Flow Duration",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Flow Packets/s": "Flow Pkts/s",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Fwd Header Length": "Fwd Header Len",
    "ACK Flag Count": "ACK Flag Cnt",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

# The testbed extractor (sdn/feature_extractor.py) already emits canonical names.
TESTBED_FEATURE_MAP = {f: f for f in INSDN_FEATURE_MAP}


def load_mapped(path: str, cfg: dict, feature_map: dict, label_col: str | None = None,
                benign_label: str | None = None):
    """Load a foreign dataset and rename its columns to the canonical eight."""
    lc = label_col or cfg["label_column"]
    keep = {src.strip() for src in feature_map.values()} | {lc.strip()}
    df = load_csv_dir(path, keep_cols=keep)
    rename = {src: canon for canon, src in feature_map.items() if src in df.columns}
    df = df.rename(columns=rename)
    df = basic_clean(df, cfg["features"])
    X = df[cfg["features"]].to_numpy(dtype=np.float32)
    y = None
    lc = label_col or cfg["label_column"]
    if lc in df.columns:
        y = make_labels(df, lc, benign_label or cfg["benign_label"])
    return X, y


def load_insdn(cfg: dict):
    # InSDN's binary label is the numeric `target` column (0 = Normal, 1 = attack).
    return load_mapped(cfg["data"]["insdn_dir"], cfg, INSDN_FEATURE_MAP,
                       label_col=cfg["data"].get("insdn_label_column", "target"),
                       benign_label=cfg["data"].get("insdn_benign_label", "0"))


def load_testbed(cfg: dict):
    return load_mapped(cfg["data"]["testbed_capture"], cfg, TESTBED_FEATURE_MAP)
