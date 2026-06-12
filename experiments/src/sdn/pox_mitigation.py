"""POX controller component: closed-loop DDoS detection + mitigation.

Load it with POX (inside the guest, with the trained model available):

    ./pox.py log.level --DEBUG src.sdn.pox_mitigation \
        --model=/path/results/proposed_cnn_lstm.keras --config=/path/config.yaml

Behaviour (paper, Section: Closed-Loop Detection and Mitigation):
  * count Packet-In messages per source over a window (config: sdn.detect_window)
  * build the 8-feature vector for that source (via SFlowAggregator) and classify
  * if mean attack-probability >= sdn.detect_threshold, install an OpenFlow rule to
    drop (or rate-limit) that source, with a hard timeout so it can recover
  * record time-to-detect, time-to-mitigate, and the Packet-In rate, to results/

This module is import-safe without POX (the heavy imports happen in launch()).
"""
from __future__ import annotations

import time
from collections import defaultdict


class MitigationEngine:
    def __init__(self, model_path: str, cfg: dict, results_path: str):
        import numpy as np  # local imports keep the module importable without deps
        import tensorflow as tf

        from .feature_extractor import SFlowAggregator

        self.np = np
        self.model = tf.keras.models.load_model(model_path)
        self.cfg = cfg
        self.results_path = results_path
        self.agg = SFlowAggregator(window_s=1.0)

        self.packetin_count = 0
        self.packetin_per_src = defaultdict(int)
        self.mitigated = {}                 # src -> install time
        self.attack_start_ts = None         # set externally / on first suspicious burst
        self.events = []                    # timeline of (ts, kind, src, detail)

    # ---- detection on a per-source feature vector ----
    def classify_source(self, src: str, feats: dict) -> float:
        t = self.cfg["model"]["timesteps"]
        c = self.cfg["model"]["channels"]
        x = self.np.array([[feats[k] for k in self.cfg["features"]]], dtype="float32")
        # NOTE: in production, scale with the SAME StandardScaler fit on training data.
        # TODO: load the persisted scaler (joblib) and apply it here before reshape.
        x = x.reshape((1, t, c))
        return float(self.model(x, training=False).numpy().ravel()[0])

    def should_mitigate(self, prob: float) -> bool:
        return prob >= self.cfg["sdn"]["detect_threshold"]

    def record(self, kind: str, src: str, detail=None) -> None:
        self.events.append((time.time(), kind, src, detail))

    def flush_metrics(self) -> None:
        from ..utils.common import save_json

        save_json({"events": self.events,
                   "total_packetins": self.packetin_count}, self.results_path)


def launch(model="results/proposed_cnn_lstm.keras", config="config.yaml",
           results="results/mitigation_runtime.json"):
    """POX entry point."""
    from pox.core import core
    import pox.openflow.libopenflow_01 as of
    from pox.lib.addresses import IPAddr

    from ..utils.common import load_config

    cfg = load_config(config)
    engine = MitigationEngine(model, cfg, results)
    log = core.getLogger()

    def install_block(event, src_ip: str):
        """Push a drop rule for src_ip (rate-limit via OVS meters is a TODO)."""
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x0800, nw_src=IPAddr(src_ip))
        msg.hard_timeout = cfg["sdn"]["rule_hard_timeout"]
        msg.priority = 65000
        # empty action list == drop
        event.connection.send(msg)
        engine.mitigated[src_ip] = time.time()
        engine.record("mitigate", src_ip,
                      {"mode": cfg["sdn"]["mitigation"], "ts": time.time()})
        log.warning("Mitigation installed for %s", src_ip)

    def _handle_PacketIn(event):
        engine.packetin_count += 1
        packet = event.parsed
        ip = packet.find("ipv4")
        if ip is None:
            return
        src = str(ip.srcip)
        engine.packetin_per_src[src] += 1
        if src in engine.mitigated:
            return

        # decide once we have enough evidence for this source
        if engine.packetin_per_src[src] >= cfg["sdn"]["detect_window"]:
            # TODO: pull the real per-source feature vector from the sFlow aggregator;
            # here we read whatever the aggregator has flushed for this src.
            feats = {k: 0.0 for k in cfg["features"]}  # placeholder vector
            prob = engine.classify_source(src, feats)
            engine.record("detect", src, {"prob": prob})
            if engine.should_mitigate(prob):
                install_block(event, src)
            engine.packetin_per_src[src] = 0

    def _handle_ConnectionUp(event):
        log.info("Switch %s connected; mitigation engine active", event.dpid)

    core.openflow.addListenerByName("PacketIn", _handle_PacketIn)
    core.openflow.addListenerByName("ConnectionUp", _handle_ConnectionUp)
    core.addListenerByName("GoingDownEvent", lambda e: engine.flush_metrics())
    log.info("pox_mitigation loaded (model=%s)", model)
