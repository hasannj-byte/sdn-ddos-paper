"""POX controller component: closed-loop DDoS detection + mitigation.

Load it with POX (inside the guest, with the trained model available):

    ./pox.py log.level --DEBUG src.sdn.pox_mitigation \
        --model=/path/results/proposed_cnn_lstm.keras \
        --scaler=/path/results/scaler.joblib \
        --config=/path/config.yaml --engine=on

Behaviour (paper, Section: Closed-Loop Detection and Mitigation):
  * flood every packet from a non-mitigated source (OFPP_FLOOD, no cached
    forwarding flow) so attack traffic keeps generating Packet-Ins until it is
    mitigated -- this is what makes "Packet-In/s collapses after mitigation" a
    real measurement rather than an artifact of some other forwarding component.
  * count Packet-In messages per source and fold each into a per-source
    flow-feature aggregator (SFlowAggregator); a source's window closes after
    sdn.detect_window_seconds of wall-clock time (not a fixed packet count --
    see the note in SFlowAggregator/pop_for_source about why: a fixed count
    closes almost instantly for any source under Mininet's virtual switching,
    regardless of its real send rate, which made Flow Duration/Flow Packets/s
    uninformative for classification)
  * once a source's window closes (and it has at least sdn.detect_window_min_packets),
    build its 8-feature vector, scale it with the SAME StandardScaler fit at
    training time, and classify
  * if mean attack-probability >= sdn.detect_threshold, install an OpenFlow
    drop rule for that source, with a hard timeout so it can recover
  * record time-to-detect, time-to-mitigate, and a 1s Packet-In-rate series, to
    results/mitigation_runtime.json

`--engine=off` runs the same flood-forwarding + Packet-In counting/sampling
with NO model loaded and no classify/mitigate events -- this is the "defense
OFF" condition; it never imports TensorFlow.

This module is import-safe without POX (the heavy imports happen in launch()).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict


class MitigationEngine:
    def __init__(self, cfg: dict, results_path: str,
                 model_path: str | None = None, scaler_path: str | None = None):
        from .feature_extractor import SFlowAggregator

        self.cfg = cfg
        self.results_path = results_path
        self.agg = SFlowAggregator(window_s=1.0)

        self.enabled = model_path is not None
        self.model = None
        self.scaler = None
        if self.enabled:
            # A single-sample inference gets nothing from a GPU except extra
            # CUDA-context latency/noise -- keep the controller's timing
            # numbers CPU-only and deterministic.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            import numpy as np
            import joblib
            import tensorflow as tf

            self.np = np
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            # Warm up graph tracing now, not during the first real detection.
            t = cfg["model"]["timesteps"]
            c = cfg["model"]["channels"]
            self.model(np.zeros((1, t, c), dtype="float32"), training=False)

        self.packetin_count = 0
        self.packetin_per_src = defaultdict(int)
        self.window_start = {}              # src -> wall-clock time of its window's first packet
        self.mitigated = {}                 # src -> install time
        self.events = []                    # timeline of (ts, kind, src, detail)
        self.packetin_samples = []          # [(ts, count_since_last_sample), ...]

    def window_ready(self, src: str, now: float) -> bool:
        """True once src's current window has run for detect_window_seconds
        and accumulated at least detect_window_min_packets -- a time-based
        close, not a packet-count-based one (see module docstring)."""
        start = self.window_start.get(src)
        if start is None:
            return False
        window_s = self.cfg["sdn"].get("detect_window_seconds", 1.0)
        min_pkts = self.cfg["sdn"].get("detect_window_min_packets", 2)
        return (now - start) >= window_s and self.packetin_per_src[src] >= min_pkts

    # ---- detection on a per-source feature vector ----
    def classify_source(self, src: str, feats: dict) -> float:
        t = self.cfg["model"]["timesteps"]
        c = self.cfg["model"]["channels"]
        raw = self.np.array([[feats[k] for k in self.cfg["features"]]], dtype="float32")
        x = self.scaler.transform(raw)
        if os.environ.get("POX_MITIGATION_DEBUG_FEATS"):
            import sys
            print(f"DEBUGFEATS src={src} raw={dict(zip(self.cfg['features'], raw[0].tolist()))} "
                  f"scaled={x[0].tolist()}", file=sys.stderr, flush=True)
        x = x.reshape((1, t, c))
        return float(self.model(x, training=False).numpy().ravel()[0])

    def should_mitigate(self, prob: float) -> bool:
        return prob >= self.cfg["sdn"]["detect_threshold"]

    def record(self, kind: str, src: str, detail=None) -> None:
        self.events.append((time.time(), kind, src, detail))

    def sample_packetin_rate(self) -> None:
        """Called every ~1s by a recoco Timer; records the Packet-In delta."""
        now = time.time()
        last_total = self.packetin_samples[-1][1] if self.packetin_samples else 0
        self.packetin_samples.append((now, self.packetin_count - last_total))

    def flush_metrics(self) -> None:
        from ..utils.common import save_json

        save_json({
            "events": self.events,
            "total_packetins": self.packetin_count,
            "packetin_rate_samples": self.packetin_samples,
        }, self.results_path)


def launch(config="config.yaml", results="results/mitigation_runtime.json",
           model=None, scaler=None, engine="off"):
    """POX entry point.

    engine: "off" -> flood-forward + count/sample only, no model, no mitigation.
            "on"  -> full detection + mitigation (requires model/scaler).
    """
    from pox.core import core
    import pox.openflow.libopenflow_01 as of
    from pox.lib.addresses import IPAddr
    from pox.lib.recoco import Timer

    from ..utils.common import load_config

    cfg = load_config(config)
    use_model = str(engine).lower() == "on"
    if use_model and not (model and scaler):
        raise ValueError("--engine=on requires --model and --scaler")
    engine_obj = MitigationEngine(cfg, results,
                                   model_path=model if use_model else None,
                                   scaler_path=scaler if use_model else None)
    log = core.getLogger()

    def install_block(event, src_ip: str):
        """Push a hard drop rule for src_ip (rate-limiting via OVS meters is a
        documented follow-up; config.yaml's sdn.mitigation is treated as
        informational here -- every mitigation is a drop)."""
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x0800, nw_src=IPAddr(src_ip))
        msg.hard_timeout = cfg["sdn"]["rule_hard_timeout"]
        msg.priority = 65000
        # empty action list == drop
        event.connection.send(msg)
        engine_obj.mitigated[src_ip] = time.time()
        engine_obj.record("mitigate", src_ip,
                           {"mode": "drop", "ts": time.time()})
        log.warning("Mitigation installed for %s", src_ip)

    def flood(event):
        """No learned/cached forwarding flow -- every packet from a
        non-mitigated source is flooded, so attack sources keep generating
        Packet-Ins until they're mitigated."""
        msg = of.ofp_packet_out()
        msg.data = event.ofp
        msg.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        msg.in_port = event.port
        event.connection.send(msg)

    def _handle_PacketIn(event):
        engine_obj.packetin_count += 1
        packet = event.parsed
        ip = packet.find("ipv4")
        if ip is None:
            flood(event)  # let ARP etc. through so hosts can resolve each other
            return
        src = str(ip.srcip)
        now = time.time()
        if src not in engine_obj.window_start:
            engine_obj.window_start[src] = now
        engine_obj.packetin_per_src[src] += 1

        if src in engine_obj.mitigated:
            return  # matched by the installed drop rule in practice; no-op here

        tcp = packet.find("tcp")
        sample = {
            "src": src, "dst": str(ip.dstip), "proto": ip.protocol,
            "ts": now, "length": ip.iplen, "direction": "fwd",
            "tcp_flags": int(tcp.flags) if tcp else 0,
            "tcp_window": int(tcp.win) if tcp else 0,
            "tcp_hdr_len": int(tcp.hdr_len) if tcp else 0,
        }
        engine_obj.agg.update(sample)

        if use_model and engine_obj.window_ready(src, now):
            feats = engine_obj.agg.pop_for_source(src)
            if feats is not None:
                prob = engine_obj.classify_source(src, feats)
                engine_obj.record("detect", src, {"prob": prob})
                if engine_obj.should_mitigate(prob):
                    install_block(event, src)
            engine_obj.packetin_per_src[src] = 0
            engine_obj.window_start[src] = now  # start the next window

        flood(event)

    def _handle_ConnectionUp(event):
        log.info("Switch %s connected; mitigation engine active (engine=%s)",
                  event.dpid, "on" if use_model else "off")

    core.openflow.addListenerByName("PacketIn", _handle_PacketIn)
    core.openflow.addListenerByName("ConnectionUp", _handle_ConnectionUp)
    core.addListenerByName("GoingDownEvent", lambda e: engine_obj.flush_metrics())
    Timer(1, engine_obj.sample_packetin_rate, recurring=True)

    # POX itself only installs a SIGHUP handler (config reread, not shutdown).
    # run_testbed.py needs a clean, deterministic way to make it flush
    # mitigation_runtime.json before exiting, so handle SIGTERM/SIGINT here.
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_: core.quit())
    _signal.signal(_signal.SIGINT, lambda *_: core.quit())

    log.info("pox_mitigation loaded (engine=%s, model=%s)", "on" if use_model else "off", model)
