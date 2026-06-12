"""Baselines re-implemented on the SAME split as the proposed model.

The point (paper, contribution 4) is a fair, like-for-like comparison: do NOT quote
other papers' headline numbers. Every model here is trained and evaluated on the
identical leakage-free split, and every one reports its parameter count and latency.

Deep baselines (CNN, LSTM, full CNN-LSTM, small Transformer) consume the 3D
sequence input; tree baselines (RF, XGBoost) consume the flat 2D input.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from .cnn_lstm import build_cnn_lstm, compile_model  # noqa: F401  (re-export)


def build_cnn(cfg: dict) -> tf.keras.Model:
    m = cfg["model"]
    inp = layers.Input(shape=(m["timesteps"], m["channels"]))
    x = layers.Conv1D(m["conv_filters"], m["conv_kernel"], activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(m["dense_units"], activation="relu")(x)
    x = layers.Dropout(m["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="cnn")


def build_lstm(cfg: dict) -> tf.keras.Model:
    m = cfg["model"]
    inp = layers.Input(shape=(m["timesteps"], m["channels"]))
    x = layers.LSTM(m["lstm_units"], return_sequences=True)(inp)
    x = layers.LSTM(m["lstm_units"])(x)
    x = layers.Dense(m["dense_units"], activation="relu")(x)
    x = layers.Dropout(m["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="lstm")


def build_transformer(cfg: dict, num_heads: int = 2, ff_dim: int = 64) -> tf.keras.Model:
    """A deliberately small Transformer encoder block, for a modern comparison."""
    m = cfg["model"]
    inp = layers.Input(shape=(m["timesteps"], m["channels"]))
    x = layers.Dense(32)(inp)  # project channels up so attention has width
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(32)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(m["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="transformer")


# full CNN-LSTM == the proposed architecture (kept as a named baseline for clarity)
def build_cnn_lstm_full(cfg: dict) -> tf.keras.Model:
    model = build_cnn_lstm(cfg)
    model._name = "cnn_lstm_full"
    return model


KERAS_BUILDERS = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "transformer": build_transformer,
    "cnn_lstm_full": build_cnn_lstm_full,
}


def build_tree_baselines(cfg: dict) -> dict:
    """Return untrained sklearn/xgboost estimators (operate on flat 2D features)."""
    from sklearn.ensemble import RandomForestClassifier

    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=cfg["seed"]
        ),
    }
    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, eval_metric="logloss", random_state=cfg["seed"], n_jobs=-1,
        )
    except Exception as e:  # ImportError or dlopen/libomp failure
        print(f"[warn] xgboost unavailable, skipping ({e.__class__.__name__}). "
              f"On macOS: brew install libomp")
    return estimators
