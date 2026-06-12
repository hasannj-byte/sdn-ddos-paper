"""Phase 0 -- Detection performance on CICDDoS2019 (leakage-free).

Trains the proposed CNN-LSTM and every baseline on the identical split, then
evaluates on the natural-distribution test set. Fills Table tab:detection_results.

Run:
    python -m src.experiments.train_detect --config config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from ..data import datasets, preprocess
from ..models import baselines, cnn_lstm
from ..utils import metrics
from ..utils.common import ensure_dir, load_config, save_json, set_seed


def _callbacks(cfg):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["train"]["reduce_lr_factor"],
            patience=cfg["train"]["reduce_lr_patience"]),
    ]


def train_keras(name, builder, splits, cfg):
    Xtr = preprocess.to_sequences(splits.X_train, cfg)
    Xva = preprocess.to_sequences(splits.X_val, cfg)
    Xte = preprocess.to_sequences(splits.X_test, cfg)

    model = cnn_lstm.compile_model(builder(cfg), cfg)
    model.fit(
        Xtr, splits.y_train,
        validation_data=(Xva, splits.y_val),
        epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
        callbacks=_callbacks(cfg), verbose=2,
    )
    y_prob = model.predict(Xte, batch_size=4096, verbose=0)
    res = metrics.compute_metrics(splits.y_test, y_prob)
    res.n_params = int(model.count_params())
    res.latency_ms = metrics.measure_latency_ms(
        lambda x: model(x, training=False).numpy(), Xte)
    return res, model


def train_tree(name, estimator, splits, cfg):
    estimator.fit(splits.X_train, splits.y_train)
    y_prob = estimator.predict_proba(splits.X_test)[:, 1]
    res = metrics.compute_metrics(splits.y_test, y_prob)
    # tree "params" -> report node/tree count where available, else None
    res.n_params = getattr(estimator, "n_estimators", None)
    res.latency_ms = metrics.measure_latency_ms(
        lambda x: estimator.predict_proba(x)[:, 1], splits.X_test)
    return res, estimator


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out = ensure_dir(cfg["output_dir"])

    print("Loading CICDDoS2019 ...")
    X, y = datasets.load_cicddos2019(cfg)
    splits = preprocess.prepare(X, y, cfg)
    print(f"train={len(splits.y_train)} val={len(splits.y_val)} test={len(splits.y_test)} "
          f"(test positive rate={splits.y_test.mean():.4f})")

    results = {}

    # proposed + keras baselines
    for name, builder in {"proposed": cnn_lstm.build_cnn_lstm, **baselines.KERAS_BUILDERS}.items():
        print(f"\n=== {name} ===")
        res, model = train_keras(name, builder, splits, cfg)
        results[name] = res.as_dict()
        if name == "proposed":
            model.save(out / "proposed_cnn_lstm.keras")
        print(results[name])

    # tree baselines
    for name, est in baselines.build_tree_baselines(cfg).items():
        print(f"\n=== {name} ===")
        res, _ = train_tree(name, est, splits, cfg)
        results[name] = res.as_dict()
        print(results[name])

    save_json(results, out / "detection_results.json")
    print(f"\nSaved -> {out / 'detection_results.json'}")
    print("Fill the [XX] cells of tab:detection_results from this file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
