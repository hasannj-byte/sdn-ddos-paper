"""Mininet topology and traffic generation for the closed-loop testbed.

Run inside the Ubuntu guest, as root, with the POX controller already listening:

    sudo python3 -m src.sdn.topology --config config.yaml --attack syn

Topology: one OVS switch, a remote POX controller, N benign hosts + M attacker
hosts. Benign traffic is ICMP/TCP; attacks are SYN flood, UDP flood, and a
low-rate (Shrew-style) pulse. sFlow is enabled on the switch so the feature
extractor can observe traffic (see feature_extractor.py).
"""
from __future__ import annotations

import argparse
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


class SingleSwitchTopo(Topo):
    def build(self, n_benign=4, n_attack=2):
        switch = self.addSwitch("s1", protocols="OpenFlow13")
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


def generate_benign(net, duration: int) -> None:
    """Light ICMP + iperf TCP between benign hosts."""
    hosts = [h for h in net.hosts if h.name.startswith("h")]
    if len(hosts) >= 2:
        server, client = hosts[0], hosts[1]
        server.cmd("iperf -s &")
        client.cmd(f"ping -c {duration} {server.IP()} &")
        client.cmd(f"iperf -c {server.IP()} -t {duration} &")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--attack", choices=["none", "syn", "udp", "lowrate"], default="syn")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--cli", action="store_true", help="drop into Mininet CLI at the end")
    args = ap.parse_args()

    if Mininet is None:
        raise SystemExit("Mininet not installed -- run this inside the Ubuntu guest.")

    from ..utils.common import load_config
    cfg = load_config(args.config)
    setLogLevel("info")

    topo = SingleSwitchTopo()
    net = Mininet(topo=topo, switch=OVSSwitch,
                  controller=lambda name: RemoteController(
                      name, ip=cfg["sdn"]["controller_ip"], port=cfg["sdn"]["controller_port"]),
                  link=TCLink, autoSetMacs=True)
    net.start()
    enable_sflow("s1")

    target = net.get("h1")
    generate_benign(net, args.duration)
    if args.attack != "none":
        time.sleep(5)  # let benign baseline establish
        launch_attack(net, args.attack, target.IP(), args.duration)

    time.sleep(args.duration + 5)
    if args.cli:
        CLI(net)
    net.stop()


if __name__ == "__main__":
    main()
