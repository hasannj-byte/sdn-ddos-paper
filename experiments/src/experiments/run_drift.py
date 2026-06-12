"""Phase 2 -- Robustness to concept drift. Fills Table tab:drift_results.

Protocol:
  1. Build a stream where attack families enter one after another (config:
     drift.stream_families), each mixed with benign traffic, so the input
     distribution shifts at known points.
  2. Pre-train the base CNN-LSTM on the first segment only.
  3. Replay the remaining stream through three settings and compare prequentially:
        - static (no updates)
        - adaptive, no replay buffer
        - adaptive + replay (proposed)
  4. Report prequential accuracy/F1, forgetting, recovery (updates), update cost.

Run:
    python -m src.experiments.run_drift --config config.yaml
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import tensorflow as tf

from ..data import datasets, preprocess
from ..drift.online_adapt import OnlineAdapter
from ..models import cnn_lstm
from ..utils import metrics
from ..utils.common import ensure_dir, load_config, save_json, set_seed


def build_stream(X, y, fam, cfg, rng):
    """Yield ordered segments [(family, X_seg, y_seg), ...].

    Each segment = benign flows + the flows of one attack family, shuffled.
    """
    fam_up = np.array([f.upper() for f in fam])
    benign_mask = y == 0
    segments = []
    for name in cfg["drift"]["stream_families"]:
        m = fam_up == name.upper()
        if not m.any():
            print(f"  [warn] family '{name}' not found in data; skipping")
            continue
        # pair each family with a benign chunk of similar size
        idx_attack = np.where(m)[0]
        idx_benign = rng.choice(np.where(benign_mask)[0],
                                size=min(len(idx_attack), benign_mask.sum()), replace=False)
        idx = rng.permutation(np.concatenate([idx_attack, idx_benign]))
        segments.append((name, X[idx], y[idx]))
    return segments


def prequential(adapter, segments, batch=256):
    """Stream segments through an adapter; collect per-family final accuracy + peak."""
    preds, trues = [], []
    final_acc, peak_acc = {}, {}
    for name, Xseg, yseg in segments:
        seg_pred, seg_true = [], []
        for i in range(0, len(Xseg), batch):
            xb, yb = Xseg[i:i + batch], yseg[i:i + batch]
            prob = adapter.step(xb, yb)
            seg_pred.extend((prob >= 0.5).astype(int))
            seg_true.extend(yb)
        preds.extend(seg_pred); trues.extend(seg_true)
        acc = float(np.mean(np.array(seg_pred) == np.array(seg_true)))
        final_acc[name] = acc
        peak_acc[name] = max(peak_acc.get(name, 0.0), acc)
    m = metrics.compute_metrics(trues, preds)
    return m, final_acc, peak_acc


def run_setting(base_weights, segments, cfg, adaptive, use_replay, label):
    model = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    model.set_weights(base_weights)
    adapter = OnlineAdapter(model, cfg, adaptive=adaptive, use_replay=use_replay)
    m, final_acc, peak_acc = prequential(adapter, segments)
    return {
        "label": label,
        "preq_accuracy": m.accuracy,
        "preq_f1": m.f1,
        "forgetting": metrics.forgetting_measure(final_acc, peak_acc),
        "n_updates": adapter.n_updates,
        "mean_update_ms": float(np.mean(adapter.update_time_ms)) if adapter.update_time_ms else None,
        "per_family_final_acc": final_acc,
    }


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = ensure_dir(cfg["output_dir"])

    print("Loading CICDDoS2019 (with attack family labels) ...")
    X, y, fam = datasets.load_cicddos2019(cfg, with_family=True)

    # fit scaler on the whole stream's first segment only would be ideal; for the
    # streaming study we scale with a scaler fit on a held-out base sample.
    splits = preprocess.prepare(X, y, cfg)
    X = splits.scaler.transform(X)  # reuse the train-fitted scaler

    segments = build_stream(X, y, fam, cfg, rng)
    if len(segments) < 2:
        raise RuntimeError("Need >=2 attack families present to study drift.")

    # 2. pre-train base model on the first segment
    base_name, X0, y0 = segments[0]
    print(f"Pre-training base model on first segment: {base_name} ({len(y0)} flows)")
    base = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    base.fit(preprocess.to_sequences(X0, cfg), y0,
             epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5,
                                                         restore_best_weights=True)],
             verbose=2)
    base_weights = base.get_weights()
    stream = segments[1:]  # the drift portion

    results = []
    for adaptive, use_replay, label in [
        (False, False, "static"),
        (True, False, "adaptive_no_replay"),
        (True, True, "adaptive_replay_proposed"),
    ]:
        print(f"\n=== {label} ===")
        results.append(run_setting(base_weights, stream, cfg, adaptive, use_replay, label))
        print(results[-1])

    save_json({"base_family": base_name, "settings": results}, out / "drift_results.json")
    print(f"\nSaved -> {out / 'drift_results.json'}  (fill tab:drift_results)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
