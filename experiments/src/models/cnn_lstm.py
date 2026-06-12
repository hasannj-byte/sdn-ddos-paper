"""The proposed lightweight CNN-LSTM model (Section: CNN-LSTM Model Architecture).

Architecture, exactly as described in the paper:
    Input (timesteps, channels)
    -> Conv1D(64, kernel=2, ReLU) -> BatchNorm
    -> LSTM(50, return_sequences=True) -> LSTM(50)
    -> Dense(64, ReLU) -> Dropout(0.5)
    -> Dense(1, sigmoid)

Call `build_cnn_lstm(cfg).count_params()` to get the exact parameter count for the
"~80,000 parameters" claim and the Params column of Table tab:detection_results.
The two LSTM layers in the FC head are tagged so the online-adaptation step can
fine-tune only the upper layers while freezing the conv feature extractor.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cnn_lstm(cfg: dict) -> tf.keras.Model:
    m = cfg["model"]
    inp = layers.Input(shape=(m["timesteps"], m["channels"]), name="input")

    # --- CNN feature extractor (frozen during online adaptation) ---
    x = layers.Conv1D(m["conv_filters"], m["conv_kernel"], activation="relu",
                      padding="same", name="conv1d")(inp)
    x = layers.BatchNormalization(name="batchnorm")(x)

    # --- LSTM temporal block (upper layers; fine-tuned online) ---
    x = layers.LSTM(m["lstm_units"], return_sequences=True, name="lstm_1")(x)
    x = layers.LSTM(m["lstm_units"], return_sequences=False, name="lstm_2")(x)

    # --- Fully connected head ---
    x = layers.Dense(m["dense_units"], activation="relu", name="dense")(x)
    x = layers.Dropout(m["dropout"], name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return models.Model(inp, out, name="cnn_lstm")


def make_optimizer(cfg: dict) -> tf.keras.optimizers.Optimizer:
    name = cfg["train"]["optimizer"].lower()
    lr = cfg["train"]["learning_rate"]
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def compile_model(model: tf.keras.Model, cfg: dict) -> tf.keras.Model:
    model.compile(
        optimizer=make_optimizer(cfg),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Layers that the online-adaptation step is allowed to update. The conv feature
# extractor (conv1d, batchnorm) stays frozen so each update is cheap, but both LSTM
# layers and the dense head are tunable so the model has enough capacity to learn a
# genuinely new attack family during recovery.
UPPER_LAYER_NAMES = ("lstm_1", "lstm_2", "dense", "output")


def set_trainable_for_adaptation(model: tf.keras.Model) -> None:
    """Freeze everything except the upper layers, for incremental updates."""
    for layer in model.layers:
        layer.trainable = layer.name in UPPER_LAYER_NAMES
