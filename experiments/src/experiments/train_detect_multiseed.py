"""Phase 0 statistical rigor pass -- REVISION_PLAN.md mandatory fix #4.

Re-uses the exact same leakage-free split as train_detect.py (same seed, same
config) so the test set is identical to the one behind the already-committed
tab:detection_results numbers. Only the training-time randomness (weight
init, batch shuffling, tree random_state) varies across seeds -- this is
standard practice and far cheaper than re-splitting/re-SMOTE-ing the 2M-row
set 5 times, which is not what this fix is meant to test.

Produces:
  results/detection_multiseed.json  -- per-model mean/std across 5 seeds for
    every metric, plus a paired significance test (t-test and Wilcoxon) of
    the proposed model vs. XGBoost (the strongest baseline in Table 1) on MCC.
  results/optimizer_sweep.json      -- the proposed model under
    Adam/SGD x batch 64/128, on the same fixed split (fills Figure p4;
    Adam+batch64 reuses the 5-seed mean/std already computed above instead
    of a 6th training run).

Run:
    python -m src.experiments.train_detect_multiseed --config config.yaml
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import tensorflow as tf
from scipy import stats

from ..data import datasets, preprocess
from ..models import baselines, cnn_lstm
from ..utils import metrics
from ..utils.common import ensure_dir, load_config, save_json, set_seed

SEEDS = [42, 43, 44, 45, 46]


def train_keras_once(builder, splits, cfg, seed):
    set_seed(seed)
    Xtr = preprocess.to_sequences(splits.X_train, cfg)
    Xva = preprocess.to_sequences(splits.X_val, cfg)
    Xte = preprocess.to_sequences(splits.X_test, cfg)

    model = cnn_lstm.compile_model(builder(cfg), cfg)
    model.fit(
        Xtr, splits.y_train, validation_data=(Xva, splits.y_val),
        epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                patience=cfg["train"]["early_stopping_patience"], restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                factor=cfg["train"]["reduce_lr_factor"], patience=cfg["train"]["reduce_lr_patience"]),
        ],
        verbose=0,
    )
    y_prob = model.predict(Xte, batch_size=4096, verbose=0)
    return metrics.compute_metrics(splits.y_test, y_prob)


def train_tree_once(estimator_builder, splits, cfg, seed):
    cfg_seeded = copy.deepcopy(cfg)
    cfg_seeded["seed"] = seed
    estimators = estimator_builder(cfg_seeded)
    results = {}
    for name, est in estimators.items():
        est.fit(splits.X_train, splits.y_train)
        y_prob = est.predict_proba(splits.X_test)[:, 1]
        results[name] = metrics.compute_metrics(splits.y_test, y_prob)
    return results


def aggregate(per_seed_metrics: list) -> dict:
    """per_seed_metrics: list of DetectionMetrics (one per seed) -> mean/std dict."""
    keys = ["accuracy", "precision", "recall", "f1", "pr_auc", "mcc", "fpr"]
    out = {}
    for k in keys:
        vals = [getattr(m, k) for m in per_seed_metrics]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                   "values": vals}
    return out


def main(config_path: str):
    cfg = load_config(config_path)
    out = ensure_dir(cfg["output_dir"])

    print(f"Building the fixed split once (seed={cfg['seed']}, matches tab:detection_results)...")
    set_seed(cfg["seed"])
    X, y = datasets.load_cicddos2019(cfg)
    splits = preprocess.prepare(X, y, cfg)
    print(f"train={len(splits.y_train)} val={len(splits.y_val)} test={len(splits.y_test)}")

    per_model_per_seed = {name: [] for name in
                           ["proposed", *baselines.KERAS_BUILDERS.keys(), "random_forest", "xgboost"]}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        for name, builder in {"proposed": cnn_lstm.build_cnn_lstm, **baselines.KERAS_BUILDERS}.items():
            print(f"  {name} ...")
            m = train_keras_once(builder, splits, cfg, seed)
            per_model_per_seed[name].append(m)
            print(f"    acc={m.accuracy:.4f} mcc={m.mcc:.4f}")

        print("  tree baselines ...")
        tree_results = train_tree_once(baselines.build_tree_baselines, splits, cfg, seed)
        for name, m in tree_results.items():
            per_model_per_seed[name].append(m)
            print(f"    {name}: acc={m.accuracy:.4f} mcc={m.mcc:.4f}")

    aggregated = {name: aggregate(ms) for name, ms in per_model_per_seed.items() if ms}

    # Paired significance test: proposed vs. xgboost (Table 1's strongest
    # baseline) on MCC, paired by seed (same test set every time).
    sig = {}
    if "xgboost" in aggregated:
        prop_mcc = aggregated["proposed"]["mcc"]["values"]
        xgb_mcc = aggregated["xgboost"]["mcc"]["values"]
        t_stat, t_p = stats.ttest_rel(prop_mcc, xgb_mcc)
        try:
            w_stat, w_p = stats.wilcoxon(prop_mcc, xgb_mcc)
        except ValueError as e:  # e.g. all differences identical
            w_stat, w_p = None, None
            print(f"[warn] wilcoxon failed: {e}")
        sig = {
            "metric": "mcc", "n_seeds": len(SEEDS),
            "proposed_mean": float(np.mean(prop_mcc)), "xgboost_mean": float(np.mean(xgb_mcc)),
            "paired_ttest": {"t": float(t_stat), "p": float(t_p)},
            "wilcoxon": {"stat": w_stat, "p": w_p},
        }
        print(f"\nProposed vs XGBoost on MCC: paired t-test p={t_p:.4g}, "
              f"wilcoxon p={w_p if w_p is None else round(w_p, 4)}")

    save_json({"seeds": SEEDS, "per_model": aggregated, "significance": sig},
               out / "detection_multiseed.json")
    print(f"\nSaved -> {out / 'detection_multiseed.json'}")

    # --- Optimizer/batch sweep for the proposed model (Figure p4) ---
    print("\n=== optimizer/batch sweep (proposed model, seed=42) ===")
    sweep = {
        "adam_64": {"mean": aggregated["proposed"]["accuracy"]["mean"],
                    "std": aggregated["proposed"]["accuracy"]["std"],
                    "f1_mean": aggregated["proposed"]["f1"]["mean"],
                    "f1_std": aggregated["proposed"]["f1"]["std"],
                    "note": "reused from the 5-seed run above, not a separate training run"},
    }
    for opt_name, batch in [("adam", 128), ("sgd", 64), ("sgd", 128)]:
        cfg_sweep = copy.deepcopy(cfg)
        cfg_sweep["train"]["optimizer"] = opt_name
        cfg_sweep["train"]["batch_size"] = batch
        print(f"  optimizer={opt_name} batch={batch} ...")
        m = train_keras_once(cnn_lstm.build_cnn_lstm, splits, cfg_sweep, seed=42)
        sweep[f"{opt_name}_{batch}"] = {"mean": m.accuracy, "std": 0.0,
                                         "f1_mean": m.f1, "f1_std": 0.0,
                                         "note": "single run, seed=42"}
        print(f"    acc={m.accuracy:.4f} f1={m.f1:.4f}")

    save_json(sweep, out / "optimizer_sweep.json")
    print(f"Saved -> {out / 'optimizer_sweep.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
