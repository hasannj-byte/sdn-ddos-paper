"""Phase 3 -- Closed-loop mitigation & controller overhead. Fills tab:mitigation_results.

This orchestrator measures the control plane while an attack runs, with the
mitigation engine disabled vs enabled, and derives:
  time-to-detect, time-to-mitigate, peak Packet-In/s, controller CPU, legit goodput.

It does NOT itself start Mininet/POX (those need root inside the guest). Intended
per-condition sequence (driven by an external script -- see PLAN.md / README):

  1. rm -f results/mitigation_runtime.json results/legit_traffic_<label>.json
  2. Terminal 1:  ./pox.py src.sdn.pox_mitigation --engine=<off|on> ...
  3. Terminal 2:  sudo python3 -m src.sdn.topology --attack syn --duration 60 \
                     --out-dir results --label <label>
  4. Terminal 3:  python3 -m src.experiments.run_testbed --controller-pid <POX_PID> \
                     --label <label> --duration 60

run_testbed.py itself: monitors the controller for `duration` seconds, then
signals POX to stop, waits for it to flush mitigation_runtime.json AND for
topology.py to finish writing legit_traffic_<label>.json (topology.py sleeps
duration+5 before its own teardown, so this waits a little past `duration`),
then appends one row to results/mitigation_results.json and archives the
runtime file per-label so the next condition starts from a clean slate.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

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


def stop_controller_and_wait_for_flush(pid: int, runtime_json: str,
                                        before_mtime: float | None,
                                        timeout: float = 20.0) -> bool:
    """SIGTERM the controller, then poll for runtime_json to appear/update.

    Returns True if a freshly-flushed file was observed, False on timeout --
    callers must treat False as "no data", never silently read a stale file.
    """
    import psutil

    try:
        psutil.Process(pid).terminate()
    except psutil.NoSuchProcess:
        pass  # already gone

    end = time.time() + timeout
    path = Path(runtime_json)
    while time.time() < end:
        if path.exists():
            mtime = path.stat().st_mtime
            if before_mtime is None or mtime > before_mtime:
                return True
        time.sleep(0.5)
    return False


def wait_for_file(path: str, timeout: float = 20.0) -> bool:
    p = Path(path)
    end = time.time() + timeout
    while time.time() < end:
        if p.exists():
            return True
        time.sleep(0.5)
    return p.exists()


def derive_timing(runtime_json: str) -> dict:
    """From pox_mitigation's event log, compute detect/mitigate timings and
    the peak Packet-In rate. Missing file -> all-None row (never a stale/
    fabricated value)."""
    if not Path(runtime_json).exists():
        return {"time_to_detect_ms": None, "time_to_mitigate_ms": None,
                "total_packetins": None, "peak_packetins_per_s": None}
    data = json.loads(Path(runtime_json).read_text())
    events = data.get("events", [])
    first = events[0][0] if events else None
    detect = next((e[0] for e in events if e[1] == "detect"), None)
    mitigate = next((e[0] for e in events if e[1] == "mitigate"), None)
    ms = lambda a, b: round((b - a) * 1000.0, 2) if (a and b) else None

    samples = data.get("packetin_rate_samples", [])
    peak_rate = max((delta for _, delta in samples), default=None)

    return {
        "time_to_detect_ms": ms(first, detect),
        "time_to_mitigate_ms": ms(first, mitigate),
        "total_packetins": data.get("total_packetins"),
        "peak_packetins_per_s": peak_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--controller-pid", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. syn_off | syn_on")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--out-dir", default=None, help="defaults to cfg.output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(args.out_dir or cfg["output_dir"])
    runtime_json = str(out / "mitigation_runtime.json")
    legit_json = str(out / f"legit_traffic_{args.label}.json")

    before_mtime = Path(runtime_json).stat().st_mtime if Path(runtime_json).exists() else None

    print(f"Monitoring controller PID {args.controller_pid} for {args.duration}s ...")
    overhead = monitor_controller(args.controller_pid, args.duration)

    print("Attack window done; stopping controller and waiting for flush ...")
    flushed = stop_controller_and_wait_for_flush(args.controller_pid, runtime_json, before_mtime)
    if not flushed:
        print(f"WARNING: {runtime_json} was not (re)written after stopping the "
              f"controller -- timing fields will be None, not guessed.")

    # topology.py sleeps duration+5 before net.stop() writes this file.
    got_legit = wait_for_file(legit_json, timeout=20.0)
    legit = {"goodput_mbps": None, "rtt_avg_ms": None}
    if got_legit:
        legit = json.loads(Path(legit_json).read_text())
    else:
        print(f"WARNING: {legit_json} never appeared -- goodput/RTT will be None.")

    timing = derive_timing(runtime_json) if flushed else derive_timing("__missing__")

    row = {"label": args.label, **overhead, **timing, **legit}

    results_file = out / "mitigation_results.json"
    existing = []
    if results_file.exists():
        existing = json.loads(results_file.read_text())
    existing.append(row)
    save_json(existing, results_file)

    # Archive the runtime file per-label so the next condition starts clean
    # even if the external driver forgets to rm it.
    if Path(runtime_json).exists():
        os.replace(runtime_json, str(out / f"mitigation_runtime_{args.label}.json"))

    print(row)
    print(f"\nAppended -> {results_file}  (fill tab:mitigation_results)")


if __name__ == "__main__":
    main()
