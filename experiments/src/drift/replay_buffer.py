"""Class-balanced reservoir replay buffer (guards against catastrophic forgetting).

Reservoir sampling keeps a uniform random sample of the whole stream in bounded
memory. We keep a separate reservoir per class so that, when the model is fine-tuned
after drift, the replayed batch still contains both benign and attack examples --
which is what stops the model from forgetting earlier attack families while it
adapts to the newest one.
"""
from __future__ import annotations

import numpy as np


class ReservoirBuffer:
    def __init__(self, capacity: int, n_features: int, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.per_class = max(1, capacity // 2)
        self.rng = rng or np.random.default_rng()
        self._X = {0: [], 1: []}
        self._seen = {0: 0, 1: 0}
        self.n_features = n_features

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        for xi, yi in zip(X, np.asarray(y).ravel()):
            yi = int(yi)
            store = self._X[yi]
            self._seen[yi] += 1
            if len(store) < self.per_class:
                store.append(xi)
            else:
                j = self.rng.integers(0, self._seen[yi])
                if j < self.per_class:
                    store[j] = xi

    def sample(self, batch_size: int):
        """Return up to `batch_size` examples, balanced across classes when possible."""
        per = max(1, batch_size // 2)
        xs, ys = [], []
        for cls in (0, 1):
            store = self._X[cls]
            if not store:
                continue
            idx = self.rng.integers(0, len(store), size=min(per, len(store)))
            xs.extend([store[i] for i in idx])
            ys.extend([cls] * len(idx))
        if not xs:
            return np.empty((0, self.n_features), dtype=np.float32), np.empty((0,), dtype=np.int64)
        order = self.rng.permutation(len(xs))
        return np.asarray(xs, dtype=np.float32)[order], np.asarray(ys, dtype=np.int64)[order]

    def __len__(self) -> int:
        return len(self._X[0]) + len(self._X[1])
