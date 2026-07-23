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

    def pop_for_source(self, src: str) -> dict | None:
        """Merge and remove only this source's accumulator entries (there can
        be several (src,dst,proto) keys per source), leaving every other
        source's in-progress window untouched. Returns None if src has no
        accumulated data yet."""
        keys = [k for k in self.flows if k[0] == src]
        if not keys:
            return None
        merged = FlowAccumulator(first_ts=min(self.flows[k].first_ts for k in keys),
                                  last_ts=max(self.flows[k].last_ts for k in keys))
        for k in keys:
            acc = self.flows.pop(k)
            merged.fwd_lengths.extend(acc.fwd_lengths)
            merged.bwd_lengths.extend(acc.bwd_lengths)
            merged.fwd_header_bytes += acc.fwd_header_bytes
            merged.ack_count += acc.ack_count
            merged.init_win_fwd = merged.init_win_fwd or acc.init_win_fwd
            merged.min_seg_fwd = min(merged.min_seg_fwd, acc.min_seg_fwd)
            merged.n_packets += acc.n_packets
        return merged.to_features()


def parse_sflowtool_line(line: str) -> dict | None:
    """Parse one CSV line from `sflowtool -l`.

    Field indices verified against a real capture from this project's testbed
    (sflowtool 5.24, built from source -- see feature_extractor smoke test):
    a bare TCP SYN (20B IP + 20B TCP, no options/payload) measured length=40
    at index 18, and its 0x02 SYN flag appeared at index 16 in sflowtool's own
    hex notation; a 200B UDP payload (+20B IP +8B UDP = 228B) measured
    length=228 at the same index 18. Index 6 (previously used for "length")
    is actually the ethertype (e.g. "0x0800"), not a length -- that was a bug.
    """
    parts = line.strip().split(",")
    if len(parts) < 20 or parts[0] != "FLOW":
        return None
    try:
        flags_field = parts[16]
        tcp_flags = int(flags_field, 16) if flags_field.startswith("0x") else int(flags_field)
        return {
            "ts": time.time(),
            "src": parts[9],            # srcIP
            "dst": parts[10],           # dstIP
            "proto": parts[11],         # IP protocol
            "length": int(parts[18]),   # IP packet length (verified, see docstring)
            "direction": "fwd",         # approximation: no bidirectional flow tracking
            "tcp_flags": tcp_flags,
            "tcp_window": 0,            # not in -l output; disclosed approximation
            "tcp_hdr_len": 20,          # not in -l output; disclosed approximation
                                         # (correct for the no-options case, which is
                                         # what this project's synthetic traffic uses)
        }
    except (ValueError, IndexError):
        return None


def load_hosts(hosts_file: str) -> dict:
    """Read the {"benign": [...ips], "attack": [...ips], "attack_kind": ...}
    file topology.py writes at net.start() time."""
    import json

    with open(hosts_file) as fh:
        return json.load(fh)


def label_for(src: str, hosts: dict) -> str | None:
    """BENIGN / the uppercased attack kind / None if src isn't a known host
    (an unrecognized source is dropped rather than guessed at)."""
    if src in hosts.get("benign", []):
        return "BENIGN"
    if src in hosts.get("attack", []):
        return str(hosts.get("attack_kind", "ATTACK")).upper()
    return None


def run_capture(out_csv: str, hosts_file: str, duration: int = 120, window_s: float = 1.0):
    """Capture live flows to CSV for offline use as a target dataset."""
    hosts = load_hosts(hosts_file)
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
                    label = label_for(src, hosts)
                    if label is None:
                        continue  # unrecognized source -- don't guess a label
                    feats["src"] = src
                    feats["Label"] = label
                    writer.writerow(feats)
                last_flush = time.time()
            if time.time() >= end:
                break
    proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--hosts-file", default="results/topology_hosts.json")
    args = ap.parse_args()
    from ..utils.common import load_config

    cfg = load_config(args.config)
    run_capture(cfg["data"]["testbed_capture"], args.hosts_file, duration=args.duration)
