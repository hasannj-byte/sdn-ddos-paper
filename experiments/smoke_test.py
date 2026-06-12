"""End-to-end SMOKE TEST on synthetic data.

PURPOSE: prove the pipeline executes and emits correctly-shaped output. The numbers
it produces are MEANINGLESS (random synthetic flows) and must NEVER be used in the
paper. It exercises: leakage-free preprocessing -> CNN-LSTM build/train -> metrics,
the exact parameter count, and a short drift-adaptation loop.

Run:
    .venv/bin/python smoke_test.py
"""
from __future__ import annotations

import numpy as np

from src.data import preprocess
from src.drift.online_adapt import OnlineAdapter
from src.models import cnn_lstm
from src.utils import metrics
from src.utils.common import load_config, set_seed


def synthetic_dataset(n=20000, n_features=8, pos_rate=0.15, seed=42):
    """Two Gaussian blobs with a mean shift, class-imbalanced like real traffic."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos
    X_neg = rng.normal(0.0, 1.0, size=(n_neg, n_features))
    X_pos = rng.normal(0.8, 1.0, size=(n_pos, n_features))  # shifted -> learnable
    X = np.vstack([X_neg, X_pos]).astype(np.float32)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(np.int64)
    order = rng.permutation(n)
    return X[order], y[order]


def main():
    cfg = load_config("config.yaml")
    cfg["train"]["epochs"] = 3            # tiny, just to prove training runs
    set_seed(cfg["seed"])

    print("== generating synthetic flows ==")
    X, y = synthetic_dataset(seed=cfg["seed"])
    print(f"   X={X.shape}  positive rate={y.mean():.3f}")

    print("== leakage-free preprocessing ==")
    s = preprocess.prepare(X, y, cfg)
    print(f"   train={len(s.y_train)} (after resample, pos={s.y_train.mean():.3f}) "
          f"test={len(s.y_test)} (natural pos={s.y_test.mean():.3f})")

    print("== build + train proposed CNN-LSTM ==")
    model = cnn_lstm.compile_model(cnn_lstm.build_cnn_lstm(cfg), cfg)
    n_params = model.count_params()
    print(f"   >>> parameter count = {n_params:,}  (the paper's '~80,000' claim) <<<")
    model.fit(preprocess.to_sequences(s.X_train, cfg), s.y_train,
              validation_data=(preprocess.to_sequences(s.X_val, cfg), s.y_val),
              epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"], verbose=2)

    print("== evaluate on natural-distribution test set ==")
    prob = model.predict(preprocess.to_sequences(s.X_test, cfg), verbose=0)
    m = metrics.compute_metrics(s.y_test, prob)
    m.n_params = int(n_params)
    m.latency_ms = metrics.measure_latency_ms(
        lambda x: model(x, training=False).numpy(), preprocess.to_sequences(s.X_test, cfg))
    print("   metrics:", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in m.as_dict().items()})

    print("== drift adaptation smoke (synthetic stream with a distribution shift) ==")
    adapter = OnlineAdapter(model, cfg, adaptive=True, use_replay=True)
    rng = np.random.default_rng(0)
    for block in range(20):
        shift = 0.0 if block < 10 else 2.5         # drift halfway through
        Xb = rng.normal(shift, 1.0, size=(256, 8)).astype(np.float32)
        yb = (rng.random(256) < 0.3).astype(np.int64)
        adapter.step(Xb, yb)
    print(f"   incremental updates triggered = {adapter.n_updates}, "
          f"mean update time = "
          f"{np.mean(adapter.update_time_ms):.1f} ms" if adapter.update_time_ms else
          "   no drift update triggered")

    print("\nSMOKE TEST PASSED -- pipeline executes end to end.")
    print("NOTE: these numbers are from random synthetic data and are NOT paper results.")


if __name__ == "__main__":
    main()
