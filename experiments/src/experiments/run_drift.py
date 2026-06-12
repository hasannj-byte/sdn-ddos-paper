"""Phase 2 -- Robustness to concept drift. Fills Table tab:drift_results.

Protocol (v2, addresses the cold-start / forgetting / realism issues found in v1):
  1. Split each attack family into a STREAM part and a held-out EVAL part.
  2. Pre-train the base CNN-LSTM on a MIX of the first `base_n_families` families
     (more realistic than train-on-one); record its accuracy on every family's
     eval set (the "peak" reference).
  3. Stream the REMAINING families one after another through three settings:
        - static (no updates)
        - adaptive, no replay buffer
        - adaptive + replay (proposed; buffer seeded with base-family data)
  4. After streaming, RE-TEST the final model on every family's eval set. This
     measures (a) recovery on the drift families and (b) forgetting on the base
     families -- which is what gives the replay buffer something to prove.

Run:
    python -m src.experiments.run_drift --config config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from ..data import datasets, preprocess
from ..drift.online_adapt import OnlineAdapter
from ..models import cnn_lstm
from ..utils import metrics
from ..utils.common import ensure_dir, load_config, save_json, set_seed


def build_family_data(X, y, fam, cfg, rng, eval_frac=0.1):
    """Return ordered {family: dict(X_stream,y_stream,X_eval,y_eval)}.

    Each family is paired with a benign chunk, shuffled, then split into a stream
    part (seen online) and an eval part (held out for end-of-run re-testing).
    """
    fam_up = np.array([f.upper() for f in fam])
    benign_idx_all = np.where(y == 0)[0]
    data = {}
    for name in cfg["drift"]["stream_families"]:
        m = fam_up == name.upper()
        if not m.any():
            print(f"  [warn] family '{name}' not found; skipping")
            continue
        idx_attack = np.where(m)[0]
        idx_benign = rng.choice(benign_idx_all,
                                size=min(len(idx_attack), len(benign_idx_all)), replace=False)
        idx = rng.permutation(np.concatenate([idx_attack, idx_benign]))
        n_eval = max(1, int(len(idx) * eval_frac))
        ev, st = idx[:n_eval], idx[n_eval:]
        data[name] = dict(X_stream=X[st], y_stream=y[st], X_eval=X[ev], y_eval=y[ev])
    return data


def eval_acc(model, X_flat, y, cfg):
    prob = model.predict(preprocess.to_sequences(X_flat, cfg), batch_size=4096, verbose=0)
    return float(np.mean((prob.ravel() >= 0.5).astype(int) == np.asarray(y).ravel()))


def run_setting(base_weights, base_names, drift_names, fdata, base_eval_acc,
                cfg, adaptive, use_replay, label, batch=256):
    model = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    model.set_weights(base_weights)
    adapter = OnlineAdapter(model, cfg, adaptive=adaptive, use_replay=use_replay)

    # seed the replay buffer with base-family data so it can defend those families
    if use_replay:
        for nm in base_names:
            adapter.seed_buffer(fdata[nm]["X_stream"], fdata[nm]["y_stream"])

    # stream the drift families prequentially
    preds, trues, seg_acc = [], [], {}
    for nm in drift_names:
        Xs, ys = fdata[nm]["X_stream"], fdata[nm]["y_stream"]
        sp, stv = [], []
        for i in range(0, len(Xs), batch):
            prob = adapter.step(Xs[i:i + batch], ys[i:i + batch])
            sp.extend((prob >= 0.5).astype(int)); stv.extend(ys[i:i + batch])
        preds.extend(sp); trues.extend(stv)
        seg_acc[nm] = float(np.mean(np.array(sp) == np.array(stv)))

    m = metrics.compute_metrics(trues, preds)

    # end-of-run re-test on EVERY family's held-out eval set
    final_eval = {nm: eval_acc(model, fdata[nm]["X_eval"], fdata[nm]["y_eval"], cfg)
                  for nm in fdata}
    # forgetting = how much accuracy on the base families dropped vs right after pretrain
    forgetting = float(np.mean([base_eval_acc[nm] - final_eval[nm] for nm in base_names]))

    return {
        "label": label,
        "preq_accuracy": m.accuracy,
        "preq_f1": m.f1,
        "forgetting_base": forgetting,
        "n_updates": adapter.n_updates,
        "mean_update_ms": float(np.mean(adapter.update_time_ms)) if adapter.update_time_ms else None,
        "drift_stream_seg_acc": seg_acc,
        "final_eval_acc": final_eval,
    }


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = ensure_dir(cfg["output_dir"])

    print("Loading CICDDoS2019 (with attack family labels) ...")
    X, y, fam = datasets.load_cicddos2019(cfg, with_family=True)
    splits = preprocess.prepare(X, y, cfg)
    X = splits.scaler.transform(X)  # reuse the train-fitted scaler

    fdata = build_family_data(X, y, fam, cfg, rng)
    names = list(fdata.keys())
    base_n = min(cfg["drift"].get("base_n_families", 2), len(names) - 1)
    if base_n < 1 or len(names) - base_n < 1:
        raise RuntimeError("Need enough families for a base set and a drift set.")
    base_names, drift_names = names[:base_n], names[base_n:]
    print(f"base families: {base_names}  |  drift families: {drift_names}")

    # 2. pre-train base on a MIX of the base families
    Xb = np.concatenate([fdata[n]["X_stream"] for n in base_names])
    yb = np.concatenate([fdata[n]["y_stream"] for n in base_names])
    perm = rng.permutation(len(yb))
    base = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    base.fit(preprocess.to_sequences(Xb[perm], cfg), yb[perm],
             epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5,
                                                         restore_best_weights=True)],
             verbose=2)
    base_weights = base.get_weights()
    base_eval_acc = {n: eval_acc(base, fdata[n]["X_eval"], fdata[n]["y_eval"], cfg) for n in names}
    print("post-pretrain eval acc:", {k: round(v, 3) for k, v in base_eval_acc.items()})

    results = []
    for adaptive, use_replay, label in [
        (False, False, "static"),
        (True, False, "adaptive_no_replay"),
        (True, True, "adaptive_replay_proposed"),
    ]:
        print(f"\n=== {label} ===")
        results.append(run_setting(base_weights, base_names, drift_names, fdata,
                                   base_eval_acc, cfg, adaptive, use_replay, label))
        print(results[-1])

    save_json({"base_families": base_names, "drift_families": drift_names,
               "post_pretrain_eval_acc": base_eval_acc, "settings": results},
              out / "drift_results.json")
    print(f"\nSaved -> {out / 'drift_results.json'}  (fill tab:drift_results)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
