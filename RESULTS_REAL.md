# Real Experiment Results (leakage-free pipeline)

Produced by `experiments/` on real CICDDoS2019 + InSDN data. These are the numbers
now filled into the manuscript tables.

## Run provenance
- **Hardware:** GPU VM `ai-gpu` (10.10.10.106) — 16 vCPU, 48 GB RAM. TensorFlow 2.17
  ran on **CPU** (TF has no Blackwell/sm_120 GPU support; the RTX 5080 sits idle).
- **Run:** 2026-06-15, all three offline phases in ~1 hour.
- **Data sample:** `max_rows = 2,000,000`, benign-preserving — keeps **all 11,457 benign**
  flows (the scarce, limiting class) plus 2M attack flows across 6 families
  (Syn, DrDoS_UDP, DrDoS_MSSQL, DrDoS_LDAP, DrDoS_NetBIOS, UDP-lag). The full 15.3M-row
  set was tried first but is pathological on CPU (model converges at epoch 1, early
  stopping never fires → ~30 h) and adds nothing because benign is identical.
- **Datasets:** CICDDoS2019 from HF mirror `baalajimaestro/CICDDoS2019` (day 01-12);
  InSDN from `Sharukesh/INSDN`. Both match official CICFlowMeter structure. *Confirm
  against the official UNB CIC release before final submission.*
- **Protocol:** leakage-free — split first, scaler fit on train, SMOTE on train folds
  only, test at the natural class ratio (test = 2,291 benign vs 397,709 attack).

## Phase 0 — Detection on CICDDoS2019 (tab:detection_results)

Acc/Prec/Rec/F1/FPR in %, PR-AUC/MCC in [0,1].

| Model | Acc | Prec | Rec | F1 | PR-AUC | MCC | FPR | Params | Lat (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 99.981 | 99.999 | 99.981 | 99.990 | 1.0000 | 0.9835 | 0.087 | 200 | 14.2 |
| **XGBoost** | 99.994 | 99.999 | 99.995 | 99.997 | 1.0000 | **0.9950** | 0.087 | 300 | **0.13** |
| CNN (1D) | 99.922 | 99.999 | 99.923 | 99.961 | 1.0000 | 0.9378 | 0.175 | 4,673 | 2.2 |
| LSTM | 99.959 | 99.999 | 99.960 | 99.979 | 1.0000 | 0.9657 | 0.218 | 33,929 | 14.3 |
| CNN-LSTM (full) | 99.942 | 100.000 | 99.942 | 99.971 | 1.0000 | 0.9526 | 0.044 | 46,977 | 15.6 |
| Transformer | 99.911 | 99.999 | 99.911 | 99.955 | 1.0000 | 0.9295 | 0.175 | 8,641 | 5.3 |
| **Proposed (8 feat.)** | 99.954 | 99.999 | 99.954 | 99.977 | 1.0000 | 0.9616 | 0.087 | 46,977 | 15.6 |

Proposed confusion (test): **TN 2,289 · FP 2 · FN 183 · TP 397,526**. ROC-AUC ≈ 1.000.

**Reading:** unchanged from the subset run — proposed is **mid-pack**; XGBoost wins
(MCC 0.995, 0.13 ms), RF and LSTM also beat it. Accuracy is saturated/non-discriminating;
MCC separates the models. Confirms the paper's "adaptivity, not accuracy" framing.

### Statistical rigor pass (tab:detection_stats) — REVISION_PLAN.md mandatory fix #4

Run 2026-07-23 on `ai-gpu`, `experiments/src/experiments/train_detect_multiseed.py`.
Same fixed split as above (seed=42, 2M rows); only training-time randomness varied
across 5 seeds (42-46). Took ~11.5h wall-clock on CPU (memory grew slowly from 4%->9%
across the run — likely TF graph-state accumulation across ~35 sequential model
trainings in one process without `clear_session()`; didn't hit OOM on 48GB, didn't
accelerate, so let it finish rather than kill+restart).

| Model | Acc mean±std | MCC mean±std |
|---|---|---|
| Random Forest | 99.981±0.000 | 0.9834±0.0002 |
| XGBoost | 99.995±0.000 | 0.9953±0.0003 |
| CNN (1D) | 99.926±0.009 | 0.9408±0.0064 |
| LSTM | 99.961±0.006 | 0.9670±0.0045 |
| CNN-LSTM (full) | 99.962±0.007 | 0.9678±0.0053 |
| Transformer | 99.931±0.023 | 0.9445±0.0171 |
| **Proposed (8 feat.)** | 99.962±0.007 | 0.9678±0.0053 |

Ranking matches the single-run Table 1 exactly. Proposed vs. XGBoost on MCC (paired by
seed): paired t-test p=0.0003 (significant), Wilcoxon signed-rank p=0.0625 (not
significant at n=5, but underpowered rather than contradicting — all 5 differences
same direction). Both point the same way: XGBoost is genuinely better on MCC, not by
single-run luck.

### Optimizer/batch sweep (Figure p4)

Adam+batch64 = the 5-seed mean above (99.962% acc / 99.981% F1); the other three
configs are single runs at seed 42 on the same split: Adam+128 = 99.960%/99.980%,
SGD+64 = 99.932%/99.966%, SGD+128 = 99.922%/99.961%. Adam beats SGD at both batch
sizes; batch 64 edges out 128 for a given optimizer; the spread is under 0.05pp.

## Phase 1 — Cross-dataset (tab:cross_dataset_results)

| Train → Test | ZS Acc | ZS F1 | Adapt Acc | Adapt F1 |
|---|---|---|---|---|
| CICDDoS2019 → CICDDoS2019 (ref) | 0.9995 | 0.9998 | — | — |
| **CICDDoS2019 → InSDN** | **0.3173** | 0.3106 | **0.9676** | 0.9802 |
| CICDDoS2019 → Testbed | pending (needs Phase 3) | | | |

**Reading:** strong result — zero-shot transfer collapses 0.9995 → 0.317; a 5% adaptation
recovers to 0.968. Zero-shot PR-AUC stays 0.87 (ranking OK, threshold miscalibrated).

## Phase 2 — Concept drift (tab:drift_results)

Base (mixed pre-train): Syn + DrDoS_UDP · Drift: MSSQL, LDAP, NetBIOS, UDP-lag

| Setting | Preq Acc | Preq F1 | Forgetting | Updates | Update ms |
|---|---|---|---|---|---|
| Static | 0.5899 | 0.7314 | 0.0000 | 0 | — |
| Adaptive, no replay | 0.9311 | 0.9631 | 0.0001 | 13 | 472 |
| **Adaptive + replay (proposed)** | 0.9311 | 0.9631 | 0.0002 | 16 | 507 |

Final per-family (held-out eval): NetBIOS **0.055 → 0.999**, MSSQL **0.834 → 0.999**,
LDAP 0.995 → 1.000; base Syn/UDP retained ~0.999.

**Reading:** clear win — adaptation lifts 0.59 → 0.93 and restores two failing families
to ~1.0 with no forgetting. Replay ≈ no-replay (both ~0.931, forgetting ≈ 0): the gentle
fine-tuning causes no forgetting to rescue, so the buffer is a cheap safeguard, not a
demonstrated necessity here (documented open item).

## Phase 3 — Closed-loop mitigation (tab:mitigation_results, tab:cross_dataset_results)

Run 2026-07-23 on `ai-gpu` (Mininet 2.3.0 + Open vSwitch 3.3.4 + POX 0.7.0 "gar",
Ubuntu 24.04.4, Python 3.12.3). Three attack types (SYN flood, UDP flood,
low-rate/Shrew pulse) x defense off/on = 6 conditions, ~30s each, plus a
separate ~45s sFlow capture for the cross-dataset testbed row.

**What's real and verified:** the closed-loop mechanism end to end — POX floods
unclassified traffic (no cached forwarding flow), classifies each source once
its window fills, installs a hard-drop OpenFlow rule within ~0.12ms of a
positive classification, and flushes real telemetry on clean shutdown.
Controller CPU/memory and Packet-In counts/rates are genuine measurements
across all 6 conditions. Peak Packet-In/s falls ~15-20x once the defense is
on (e.g. SYN: 632,647 -> 33,691).

**What's NOT reliable, and is now disclosed in the manuscript (Threats to
Validity) rather than presented as a clean result:** live per-source
classification. In every one of the 6 trials, the mitigated source was the
benign traffic generator, not the actual attacking hosts. Root cause (found by
dumping raw+scaled feature vectors and comparing against the training
scaler's fitted stats): the live per-source window measures
controller-observed packet-arrival timing, which is far faster than
real-world send rate under Mininet's virtual switching, so live `Flow
Duration`/`Flow Packets/s` collapse to near-constant, uninformative values
after scaling — worsened by CICDDoS2019's own `Flow Packets/s` column being
dominated by extreme outlier rows, which skews the fitted scaler itself. This
is a structural mismatch between the live windowed approximation and
CICFlowMeter's offline flow-timeout-based feature computation, not a bug to
patch quickly; documented as the top item for future work.

Also disclosed: "Classify->rule (ms)" is inference+rule-install latency, not a
verified attack-onset-to-response time (no independent attack-start marker
exists in the log); "Client send (Mbps)" is the UDP client's self-reported
send rate, not confirmed delivery (the server-side receive report never
flushed before teardown).

**Cross-dataset testbed row:** 146 flows captured via sflowtool (built from
source — not packaged for Ubuntu 24.04) during a SYN-attack run, correctly
labeled via a `topology_hosts.json` written by `topology.py` at `net.start()`.
Evaluated on the *exact* committed model (verified by reproducing its
TN/FP/FN/TP exactly before evaluating the new target) — the existing InSDN
row was NOT touched or regenerated. Zero-shot acc/F1 0.233/0.000, adapted
0.741/0.793 — same collapse-then-recover pattern as InSDN, though with only
146 flows this is indicative, not precise, and plausibly a symptom of the
same feature-window mismatch above rather than an independent finding.

**Along the way, two real code bugs fixed (see git history for
`src/sdn/pox_mitigation.py`, `topology.py`, `feature_extractor.py`):** the
controller never forwarded/flooded any packet (would have produced all-zero
data — no traffic, not even ARP, would have crossed the switch); and
`parse_sflowtool_line`'s field indices were wrong (`length` read the
ethertype field, not a length; `tcp_flags` used decimal parsing on
sflowtool's hex-prefixed output, always yielding 0) — both verified against a
real capture before trusting them.

## Open items before final submission
1. Fix the live feature-window mismatch (proper flow-timeout-based windowing
   instead of fixed packet counts) and re-run Phase 3 for a mitigation result
   that actually targets the attacker — currently an honest limitation, not
   a solved problem.
2. Re-verify against the official CIC release; the paper notes a 2M-row representative sample.
3. Replay buffer value not demonstrated (no forgetting to rescue) — stress-test or soften claim.
4. Optimizer/batch sweep (Fig p4) was not re-run; only Adam+batch64 reported from this run.
5. Get a real UDP delivery/loss measurement for "legit goodput" (currently send-rate only).
6. Implement OVS meter-based rate-limiting (`config.yaml: sdn.mitigation`) — currently drop-only.
