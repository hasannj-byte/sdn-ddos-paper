"""Turn sFlow records into the eight canonical flow features, in real time.

Two roles:
  1. Online: aggregate sFlow samples per (src, dst, proto) flow over a short window
     and emit the 8-feature vector for the controller to classify.
  2. Offline: write the same vectors to a CSV (config: data.testbed_capture) so the
     live testbed traffic can be used as a target dataset in run_cross_dataset.py.

sFlow is sampled, so several CICFlowMeter features can only be approximated from
the available fields. The TODOs mark where the mapping must be calibrated against
your sFlow agent output (sflowtool / sFlow-RT).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field

CANONICAL_FEATURES = [
    "Flow Duration", "Fwd Packet Length Mean", "Flow Packets/s",
    "Bwd Packet Length Mean", "Fwd Header Length", "ACK Flag Count",
    "Init_Win_bytes_forward", "min_seg_size_forward",
]


@dataclass
class FlowAccumulator:
    first_ts: float
    last_ts: float
    fwd_lengths: list = field(default_factory=list)
    bwd_lengths: list = field(default_factory=list)
    fwd_header_bytes: int = 0
    ack_count: int = 0
    init_win_fwd: int = 0
    min_seg_fwd: float = float("inf")
    n_packets: int = 0

    def to_features(self) -> dict:
        dur = max(self.last_ts - self.first_ts, 1e-6)
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        return {
            "Flow Duration": dur * 1e6,                       # microseconds, as in CIC
            "Fwd Packet Length Mean": mean(self.fwd_lengths),
            "Flow Packets/s": self.n_packets / dur,
            "Bwd Packet Length Mean": mean(self.bwd_lengths),
            "Fwd Header Length": self.fwd_header_bytes,
            "ACK Flag Count": self.ack_count,
            "Init_Win_bytes_forward": self.init_win_fwd,
            "min_seg_size_forward": 0.0 if self.min_seg_fwd == float("inf") else self.min_seg_fwd,
        }


class SFlowAggregator:
    """Maintains per-flow accumulators and flushes feature vectors per window."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self.flows: dict[tuple, FlowAccumulator] = {}

    def update(self, sample: dict) -> None:
        """Fold one parsed sFlow sample into its flow accumulator.

        Expected keys in `sample` (adapt to your parser): src, dst, proto, length,
        direction ('fwd'/'bwd'), tcp_flags, tcp_window, tcp_hdr_len, ts.
        """
        key = (sample["src"], sample["dst"], sample.get("proto", "ip"))
        acc = self.flows.get(key)
        if acc is None:
            acc = FlowAccumulator(first_ts=sample["ts"], last_ts=sample["ts"],
                                  init_win_fwd=sample.get("tcp_window", 0))
            self.flows[key] = acc
        acc.last_ts = sample["ts"]
        acc.n_packets += 1
        if sample.get("direction", "fwd") == "fwd":
            acc.fwd_lengths.append(sample.get("length", 0))
            acc.fwd_header_bytes += sample.get("tcp_hdr_len", 0)
            acc.min_seg_fwd = min(acc.min_seg_fwd, sample.get("length", acc.min_seg_fwd))
        else:
            acc.bwd_lengths.append(sample.get("length", 0))
        if sample.get("tcp_flags", 0) & 0x10:  # ACK bit
            acc.ack_count += 1

    def flush(self):
        """Return [(src, feature_dict), ...] and reset the accumulators."""
        rows = [(key[0], acc.to_features()) for key, acc in self.flows.items()]
        self.flows.clear()
        return rows


def parse_sflowtool_line(line: str) -> dict | None:
    """Parse one CSV line from `sflowtool -l`.

    TODO: align the field indices with your sflowtool version's output format.
    The line layout below is the common `-l` (line) format.
    """
    parts = line.strip().split(",")
    if len(parts) < 20 or parts[0] != "FLOW":
        return None
    try:
        return {
            "ts": time.time(),
            "src": parts[9],            # srcIP
            "dst": parts[10],           # dstIP
            "proto": parts[11],         # IP protocol
            "length": int(parts[6]),    # frame length
            "direction": "fwd",         # TODO: infer from port/MAC
            "tcp_flags": int(parts[16]) if parts[16].isdigit() else 0,
            "tcp_window": 0,            # TODO: not in -l output; use sFlow-RT if needed
            "tcp_hdr_len": 20,          # TODO: approximate / parse if available
        }
    except (ValueError, IndexError):
        return None


def run_capture(out_csv: str, duration: int = 120, window_s: float = 1.0):
    """Capture live flows to CSV for offline use as a target dataset."""
    agg = SFlowAggregator(window_s)
    proc = subprocess.Popen(["sflowtool", "-l"], stdout=subprocess.PIPE, text=True)
    end = time.time() + duration
    last_flush = time.time()
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["src"] + CANONICAL_FEATURES + ["Label"])
        writer.writeheader()
        for line in proc.stdout:  # type: ignore[union-attr]
            sample = parse_sflowtool_line(line)
            if sample:
                agg.update(sample)
            if time.time() - last_flush >= window_s:
                for src, feats in agg.flush():
                    feats["src"] = src
                    feats["Label"] = "UNKNOWN"  # TODO: label by attacker host IP set
                    writer.writerow(feats)
                last_flush = time.time()
            if time.time() >= end:
                break
    proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--duration", type=int, default=120)
    args = ap.parse_args()
    from ..utils.common import load_config

    cfg = load_config(args.config)
    run_capture(cfg["data"]["testbed_capture"], duration=args.duration)
