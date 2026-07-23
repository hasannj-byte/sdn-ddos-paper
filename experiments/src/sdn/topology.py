"""Mininet topology and traffic generation for the closed-loop testbed.

Run inside the Ubuntu guest, as root, with the POX controller already listening:

    sudo python3 -m src.sdn.topology --config config.yaml --attack syn \
        --out-dir results --label syn_on

Topology: one OVS switch, a remote POX controller, N benign hosts + M attacker
hosts. Benign traffic is ICMP/TCP; attacks are SYN flood, UDP flood, and a
low-rate (Shrew-style) pulse. sFlow is enabled on the switch so the feature
extractor can observe traffic (see feature_extractor.py).

Writes, under --out-dir:
  topology_hosts.json      -- {"benign": [ips], "attack": [ips], "attack_kind": ...}
                               (feature_extractor.run_capture uses this to label rows)
  legit_traffic_<label>.json -- parsed iperf goodput + ping RTT for this run
"""
from __future__ import annotations

import argparse
import re
import time

try:
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSSwitch
    from mininet.topo import Topo
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.cli import CLI
except ImportError:  # allow import on a machine without Mininet
    Mininet = None
    Topo = object


class SingleSwitchTopo(Topo):
    def build(self, n_benign=4, n_attack=2):
        # POX's default openflow.of_01 module speaks OpenFlow 1.0 only; we
        # don't need 1.3 since mitigation is drop-only (no OVS meters).
        switch = self.addSwitch("s1", protocols="OpenFlow10")
        for i in range(n_benign):
            h = self.addHost(f"h{i+1}")
            self.addLink(h, switch, cls=TCLink, bw=100)
        for j in range(n_attack):
            a = self.addHost(f"a{j+1}")
            self.addLink(a, switch, cls=TCLink, bw=100)


def enable_sflow(switch_name: str, collector: str = "127.0.0.1:6343",
                 sampling: int = 64, polling: int = 5) -> None:
    """Configure sFlow export on the OVS bridge (mirrors the paper's sFlow setup)."""
    import subprocess

    cmd = (
        f"ovs-vsctl -- --id=@sflow create sflow agent=eth0 target=\\\"{collector}\\\" "
        f"sampling={sampling} polling={polling} "
        f"-- set bridge {switch_name} sflow=@sflow"
    )
    subprocess.run(cmd, shell=True, check=False)


def write_hosts_file(net, attack_kind: str, out_dir: str) -> str:
    import json
    from pathlib import Path

    benign = [h.IP() for h in net.hosts if h.name.startswith("h")]
    attack = [h.IP() for h in net.hosts if h.name.startswith("a")]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/topology_hosts.json"
    with open(path, "w") as fh:
        json.dump({"benign": benign, "attack": attack, "attack_kind": attack_kind}, fh, indent=2)
    return path


def generate_benign(net, duration: int, out_dir: str, label: str,
                     rate_mbps: float = 1.0, packet_len: int = 200) -> None:
    """Light ICMP + bandwidth-capped, small-packet UDP iperf between benign
    hosts, output captured to files so goodput/RTT can be parsed after the
    run (previously discarded). Deliberately modest (rather than unbounded
    TCP, which saturates the link and floods the controller at the same
    order of magnitude as an actual attack under our flood-only forwarding)
    AND small-packet (rather than iperf's default MTU-sized ~1470B UDP
    datagrams, which -- diagnosed via live feature dumps against the
    training scaler's fitted stats -- read as statistically closer to
    CICDDoS2019's large-payload reflection-attack families than to its own
    benign class) so this represents light background traffic rather than
    an accidentally attack-shaped stream."""
    hosts = [h for h in net.hosts if h.name.startswith("h")]
    if len(hosts) >= 2:
        server, client = hosts[0], hosts[1]
        server.cmd(f"iperf -s -u > {out_dir}/iperf_server_{label}.log &")
        client.cmd(f"ping -c {duration} {server.IP()} > {out_dir}/ping_{label}.log &")
        client.cmd(f"iperf -c {server.IP()} -u -b {rate_mbps}M -l {packet_len} -t {duration} "
                    f"> {out_dir}/iperf_{label}.log &")


def launch_attack(net, kind: str, target_ip: str, duration: int) -> None:
    """Start an attack from the attacker hosts. Requires hping3 in the guest."""
    attackers = [h for h in net.hosts if h.name.startswith("a")]
    for a in attackers:
        if kind == "syn":
            a.cmd(f"timeout {duration} hping3 -S --flood -p 80 {target_ip} &")
        elif kind == "udp":
            a.cmd(f"timeout {duration} hping3 --udp --flood -p 53 {target_ip} &")
        elif kind == "lowrate":
            # Shrew-style: short high-rate bursts separated by silence
            a.cmd(f"timeout {duration} sh -c 'while true; do "
                  f"hping3 -S -i u100 -c 500 -p 80 {target_ip}; sleep 1; done' &")
        else:
            raise ValueError(f"unknown attack kind: {kind}")


def parse_legit_traffic(out_dir: str, label: str) -> dict:
    """Parse the iperf/ping logs generate_benign() wrote into a goodput/RTT
    summary; None fields mean the log was missing or didn't match (e.g. no
    benign traffic completed before the network was torn down)."""
    result = {"goodput_mbps": None, "rtt_avg_ms": None}
    try:
        with open(f"{out_dir}/iperf_{label}.log") as fh:
            text = fh.read()
        m = re.search(r"([\d.]+)\s*Mbits/sec", text)
        if m:
            result["goodput_mbps"] = float(m.group(1))
    except FileNotFoundError:
        pass
    try:
        with open(f"{out_dir}/ping_{label}.log") as fh:
            text = fh.read()
        m = re.search(r"= [\d.]+/([\d.]+)/[\d.]+/[\d.]+", text)
        if m:
            result["rtt_avg_ms"] = float(m.group(1))
    except FileNotFoundError:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--attack", choices=["none", "syn", "udp", "lowrate"], default="syn")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--label", default="run")
    ap.add_argument("--cli", action="store_true", help="drop into Mininet CLI at the end")
    args = ap.parse_args()

    if Mininet is None:
        raise SystemExit("Mininet not installed -- run this inside the Ubuntu guest.")

    from ..utils.common import load_config, save_json
    cfg = load_config(args.config)
    setLogLevel("info")

    topo = SingleSwitchTopo()
    net = Mininet(topo=topo, switch=OVSSwitch,
                  controller=lambda name: RemoteController(
                      name, ip=cfg["sdn"]["controller_ip"], port=cfg["sdn"]["controller_port"]),
                  link=TCLink, autoSetMacs=True)
    net.start()
    enable_sflow("s1")
    write_hosts_file(net, args.attack, args.out_dir)

    target = net.get("h1")
    generate_benign(net, args.duration, args.out_dir, args.label)
    if args.attack != "none":
        time.sleep(5)  # let benign baseline establish
        launch_attack(net, args.attack, target.IP(), args.duration)

    time.sleep(args.duration + 5)
    if args.cli:
        CLI(net)
    net.stop()

    legit = parse_legit_traffic(args.out_dir, args.label)
    save_json(legit, f"{args.out_dir}/legit_traffic_{args.label}.json")
    print(f"Legit traffic summary -> {args.out_dir}/legit_traffic_{args.label}.json: {legit}")


if __name__ == "__main__":
    main()
