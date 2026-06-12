"""Drift-triggered incremental update (paper, Algorithm 1).

Implements the test-then-train streaming loop:
    for each incoming labeled batch:
        predict and score (this is what the prequential metric uses)
        feed the errors to the DDM detector
        if DRIFT: fine-tune the UPPER layers on (recent window + replay sample)
                  for K steps, freezing the conv feature extractor
        update the reservoir buffer
"""
from __future__ import annotations

from collections import deque

import numpy as np
import tensorflow as tf

from ..data import preprocess
from ..models.cnn_lstm import make_optimizer, set_trainable_for_adaptation
from .drift_detector import DDM, State
from .replay_buffer import ReservoirBuffer


class OnlineAdapter:
    def __init__(self, model: tf.keras.Model, cfg: dict, adaptive: bool = True,
                 use_replay: bool = True):
        self.model = model
        self.cfg = cfg
        self.adaptive = adaptive
        self.use_replay = use_replay
        d = cfg["drift"]
        self.ddm = DDM(d["warning_level"], d["drift_level"])
        self.buffer = ReservoirBuffer(d["replay_capacity"], X_FEATURES(cfg))
        self.window_X = deque(maxlen=d["window"])
        self.window_y = deque(maxlen=d["window"])
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.opt = make_optimizer(cfg)
        self.recovering = False
        self.err_ema = None
        self.n_updates = 0
        self.update_time_ms = []

    def seed_buffer(self, X_flat: np.ndarray, y: np.ndarray) -> None:
        """Pre-fill the replay buffer with earlier (base-family) data, so the buffer
        can defend those families against forgetting once drift adaptation starts."""
        if self.use_replay:
            self.buffer.add(np.asarray(X_flat, dtype=np.float32), np.asarray(y).ravel())

    # -- one streaming step over a mini-batch of flows --
    def step(self, X_flat: np.ndarray, y: np.ndarray) -> np.ndarray:
        Xseq = preprocess.to_sequences(X_flat, self.cfg)
        prob = self.model(Xseq, training=False).numpy().ravel()
        pred = (prob >= 0.5).astype(int)
        y = np.asarray(y).ravel()

        drift = False
        for xi, yi, pi in zip(X_flat, y, pred):
            self.window_X.append(xi)
            self.window_y.append(int(yi))
            if self.ddm.update(int(pi != yi)) is State.DRIFT:
                drift = True

        batch_err = float(np.mean(pred != y))
        d = self.cfg["drift"]
        decay = d.get("err_ema_decay", 0.7)
        self.err_ema = batch_err if self.err_ema is None else \
            decay * self.err_ema + (1 - decay) * batch_err

        if self.adaptive:
            # Recovery is driven by the error level itself, with hysteresis, NOT only
            # by a DDM transition. This fixes the cold-start failure: a family that is
            # hard from the very first batch (no preceding low-error baseline, so DDM
            # stays silent) still trips the high-error trigger and gets adapted to.
            # DDM is retained only as an auxiliary drift-event signal.
            if drift or self.err_ema >= d.get("high_error_trigger", 0.30):
                self.recovering = True
            if self.recovering:
                self._incremental_update(X_flat, y)
                if self.err_ema <= d["recovery_exit_error"]:
                    self.recovering = False

        if self.use_replay:
            self.buffer.add(X_flat, y)
        return prob

    def _incremental_update(self, X_cur=None, y_cur=None) -> None:
        import time

        set_trainable_for_adaptation(self.model)
        Xw = np.asarray(self.window_X, dtype=np.float32)
        yw = np.asarray(self.window_y, dtype=np.int64)
        if X_cur is not None:  # always include the freshest batch
            Xw = np.concatenate([Xw, np.asarray(X_cur, dtype=np.float32)], axis=0)
            yw = np.concatenate([yw, np.asarray(y_cur, dtype=np.int64)], axis=0)
        if self.use_replay and len(self.buffer) > 0:
            Xr, yr = self.buffer.sample(self.cfg["drift"]["replay_batch"])
            Xw = np.concatenate([Xw, Xr], axis=0)
            yw = np.concatenate([yw, yr], axis=0)
        Xw_seq = preprocess.to_sequences(Xw, self.cfg)
        yw_t = tf.convert_to_tensor(yw.reshape(-1, 1), dtype=tf.float32)

        t0 = time.perf_counter()
        for _ in range(self.cfg["drift"]["update_steps"]):
            with tf.GradientTape() as tape:
                out = self.model(Xw_seq, training=True)
                loss = self.loss_fn(yw_t, out)
            trainables = [v for v in self.model.trainable_variables]
            grads = tape.gradient(loss, trainables)
            self.opt.apply_gradients(zip(grads, trainables))
        self.update_time_ms.append((time.perf_counter() - t0) * 1000.0)
        self.n_updates += 1


def X_FEATURES(cfg: dict) -> int:
    return len(cfg["features"])
