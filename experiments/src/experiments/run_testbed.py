"""Phase 3 -- Closed-loop mitigation & controller overhead. Fills tab:mitigation_results.

This orchestrator measures the control plane while an attack runs, with the
mitigation engine disabled vs enabled, and derives:
  time-to-detect, time-to-mitigate, peak Packet-In/s, controller CPU, legit goodput.

It does NOT itself start Mininet/POX (those need root inside the guest). The
intended workflow per condition:

  Terminal 1:  ./pox.py src.sdn.pox_mitigation --model=... --config=...   # (omit module for "defense OFF")
  Terminal 2:  sudo python3 -m src.sdn.topology --attack syn --duration 60
  Terminal 3:  python3 -m src.experiments.run_testbed --controller-pid <POX_PID> \
                   --label defense_on --duration 60

Run it once per condition (defense_off, defense_on); it appends a row each time.
"""
from __future__ import annotations

import argparse
import time

from ..utils.common import ensure_dir, load_config, save_json


def monitor_controller(pid: int, duration: int, interval: float = 0.5) -> dict:
    """Sample the controller process CPU% and RSS over `duration` seconds."""
    import psutil

    proc = psutil.Process(pid)
    proc.cpu_percent(None)  # prime
    cpu, rss = [], []
    end = time.time() + duration
    while time.time() < end:
        cpu.append(proc.cpu_percent(None))
        rss.append(proc.memory_info().rss / (1024 * 1024))
        time.sleep(interval)
    return {
        "cpu_mean": sum(cpu) / len(cpu) if cpu else None,
        "cpu_peak": max(cpu) if cpu else None,
        "rss_mb_mean": sum(rss) / len(rss) if rss else None,
    }


def derive_timing(runtime_json: str) -> dict:
    """From pox_mitigation's event log, compute detect/mitigate timings."""
    import json
    from pathlib import Path

    if not Path(runtime_json).exists():
        return {"time_to_detect_ms": None, "time_to_mitigate_ms": None,
                "total_packetins": None}
    data = json.loads(Path(runtime_json).read_text())
    events = data.get("events", [])
    first = events[0][0] if events else None
    detect = next((e[0] for e in events if e[1] == "detect"), None)
    mitigate = next((e[0] for e in events if e[1] == "mitigate"), None)
    ms = lambda a, b: round((b - a) * 1000.0, 2) if (a and b) else None
    return {
        "time_to_detect_ms": ms(first, detect),
        "time_to_mitigate_ms": ms(first, mitigate),
        "total_packetins": data.get("total_packetins"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--controller-pid", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. defense_off | defense_on")
    ap.add_argument("--duration", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(cfg["output_dir"])
    runtime_json = str(out / "mitigation_runtime.json")

    print(f"Monitoring controller PID {args.controller_pid} for {args.duration}s ...")
    overhead = monitor_controller(args.controller_pid, args.duration)
    timing = derive_timing(runtime_json)

    row = {"label": args.label, **overhead, **timing,
           "note": "fill legit goodput/RTT from iperf logs in topology run"}

    results_file = out / "mitigation_results.json"
    existing = []
    if results_file.exists():
        import json
        existing = json.loads(results_file.read_text())
    existing.append(row)
    save_json(existing, results_file)
    print(row)
    print(f"\nAppended -> {results_file}  (fill tab:mitigation_results)")


if __name__ == "__main__":
    main()
