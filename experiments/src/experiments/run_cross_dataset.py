"""Phase 1 -- Cross-dataset generalization. Fills Table tab:cross_dataset_results.

Train on CICDDoS2019, then:
  - zero-shot test on InSDN and on the live testbed capture (no retraining)
  - brief adaptation on a small labeled fraction of the target, then re-test
The drop from same-dataset to zero-shot is the gap single-dataset studies hide;
the adapted column shows how much a short, label-light update recovers.

Run:
    python -m src.experiments.run_cross_dataset --config config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from ..data import datasets, preprocess
from ..models import cnn_lstm
from ..utils import metrics
from ..utils.common import ensure_dir, load_config, save_json, set_seed


def evaluate(model, X_seq, y):
    prob = model.predict(X_seq, batch_size=4096, verbose=0)
    m = metrics.compute_metrics(y, prob)
    return {"accuracy": m.accuracy, "f1": m.f1, "pr_auc": m.pr_auc, "fpr": m.fpr}


def brief_adapt(model, X_seq, y, cfg):
    """Fine-tune the upper layers on a small labeled fraction of the target."""
    cnn_lstm.set_trainable_for_adaptation(model)
    model.compile(optimizer=cnn_lstm.make_optimizer(cfg),
                  loss="binary_crossentropy", metrics=["accuracy"])
    n = int(len(y) * cfg["cross_dataset"]["adapt_fraction"])
    idx = np.random.default_rng(cfg["seed"]).choice(len(y), size=max(1, n), replace=False)
    model.fit(X_seq[idx], y[idx],
              epochs=cfg["cross_dataset"]["adapt_steps"],
              batch_size=cfg["train"]["batch_size"], verbose=0)
    return idx


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out = ensure_dir(cfg["output_dir"])

    # --- train (or load) the source model on CICDDoS2019 ---
    print("Preparing CICDDoS2019 source model ...")
    X, y = datasets.load_cicddos2019(cfg)
    splits = preprocess.prepare(X, y, cfg)
    model = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    model.fit(preprocess.to_sequences(splits.X_train, cfg), splits.y_train,
              validation_data=(preprocess.to_sequences(splits.X_val, cfg), splits.y_val),
              epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                          restore_best_weights=True)],
              verbose=2)
    source_weights = model.get_weights()

    results = {
        "CICDDoS2019->CICDDoS2019": {
            "zero_shot": evaluate(model, preprocess.to_sequences(splits.X_test, cfg), splits.y_test),
            "adapted": None,
        }
    }

    targets = {
        "CICDDoS2019->InSDN": datasets.load_insdn,
        "CICDDoS2019->Testbed": datasets.load_testbed,
    }
    for name, loader in targets.items():
        print(f"\n=== {name} ===")
        try:
            Xt, yt = loader(cfg)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [skip] {e}")
            continue
        if yt is None:
            print("  [skip] target has no usable label column")
            continue
        Xt_seq = preprocess.apply_fitted(Xt, splits.scaler, cfg)

        zero = evaluate(model, Xt_seq, yt)

        # reset to source weights, then briefly adapt
        model.set_weights(source_weights)
        idx = brief_adapt(model, Xt_seq, yt, cfg)
        mask = np.ones(len(yt), dtype=bool); mask[idx] = False  # eval on held-out target
        adapted = evaluate(model, Xt_seq[mask], yt[mask])
        model.set_weights(source_weights)  # restore for the next target

        results[name] = {"zero_shot": zero, "adapted": adapted}
        print(results[name])

    save_json(results, out / "cross_dataset_results.json")
    print(f"\nSaved -> {out / 'cross_dataset_results.json'}  (fill tab:cross_dataset_results)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
