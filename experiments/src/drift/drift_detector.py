"""DDM-style drift detector (paper, Section: Online Adaptation under Concept Drift).

Tracks the running error rate p_t and its standard deviation s_t = sqrt(p(1-p)/n).
It records the minimum of (p + s) seen so far and signals:
    WARNING  when  p_t + s_t >= p_min + warning_level * s_min
    DRIFT    when  p_t + s_t >= p_min + drift_level   * s_min
On drift the statistics are reset so the next regime is tracked from scratch.
"""
from __future__ import annotations

import math
from enum import Enum


class State(Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"


class DDM:
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0,
                 min_samples: int = 30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.p = 1.0          # running error rate
        self.s = 0.0
        self.p_min = math.inf
        self.s_min = math.inf

    def update(self, error: int) -> State:
        """Feed one prediction outcome (1 = misclassified, 0 = correct)."""
        self.n += 1
        # incremental mean of the error indicator
        self.p += (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1.0 - self.p) / self.n)

        if self.n < self.min_samples:
            return State.STABLE

        if self.p + self.s <= self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        level = self.p + self.s
        if level >= self.p_min + self.drift_level * self.s_min:
            self.reset()
            return State.DRIFT
        if level >= self.p_min + self.warning_level * self.s_min:
            return State.WARNING
        return State.STABLE
