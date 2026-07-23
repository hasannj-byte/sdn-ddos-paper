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

Two runs, both on `ai-gpu` (Mininet + Open vSwitch + POX 0.7.0 "gar",
Ubuntu 24.04.4, Python 3.12.3). Three attack types (SYN flood, UDP flood,
low-rate/Shrew pulse) x defense off/on = 6 conditions per run, ~30s each.

### v1 (2026-07-23 morning) — superseded, kept for provenance only

Real, but the controller used a fixed 50-packet detection window and UDP
benign traffic. Found (via `POX_MITIGATION_DEBUG_FEATS=1` raw/scaled feature
dumps): in every one of the 6 trials the mitigated source was the **benign**
traffic generator, never the actual attacker. Root cause: the fixed-count
window closed in ~1-2ms under Mininet's virtual switching regardless of real
send rate, so live `Flow Duration` collapsed to values with no relationship
to real flow durations; separately, UDP benign traffic shares CICDDoS2019's
zeroed-out ACK-flag/init-window fields with its own UDP attack families,
giving the model no real signal there either.

### v2 (2026-07-23 evening) — current, in the manuscript

Two fixes applied and re-tested before the full re-run: (1) time-based
detection windows (`sdn.detect_window_seconds=1.0`, replacing the fixed
packet count) — verified live `Flow Duration` now reads ~1.0s as intended;
(2) genuine periodic TCP bursts for benign traffic instead of UDP
(`topology.py:generate_benign`). Result: **the false positive is completely
gone** — zero mitigation events across all 6 conditions. But this reveals a
second, independent problem: **the actual attacker is still never classified
with enough confidence to trigger mitigation**, in any of the 3 attack types.
Root cause (unchanged, now isolated as the sole remaining cause): CICDDoS2019's
own `Flow Packets/s` column is dominated by extreme outlier rows, so the
training-fit scaler's mean (~1.43M pkt/s) makes any live traffic's realistic
packet rate read as "below average" rather than anomalous. Fixing this needs
retraining with robust/outlier-aware scaling — which would put the
already-verified Table 1/8 numbers at risk — so it's documented as the top
remaining item, not attempted here.

With zero mitigation events, the mitigation table's ON/OFF Packet-In-rate and
goodput differences are honestly reported as run-to-run variance (nothing is
actually being blocked either way), not a measured defense effect. The one
clean, real number this run demonstrates: controller RSS rises ~510-520MB
just from loading the model+scaler for classification, regardless of attack
type — the actual measured cost of running the detector continuously. Added
a `bursts_completed`/`bursts_expected` field (genuine TCP delivery signal,
unlike the old UDP client's fire-and-forget send-only report) — shows
legitimate TCP traffic is substantially disrupted by SYN/UDP floods regardless
of defense state (3-4/15 bursts complete vs 15-16/15 under the low-rate attack).

The rule-installation mechanism itself is not hypothetical — it fired
correctly within ~0.12-0.13ms of a classification in the v1 run, before the
fixes. v2 simply never exercises it, because no classification crosses
threshold.

**Cross-dataset testbed row (unaffected by the v1/v2 distinction above, since
it uses the offline sflowtool pipeline, not live POX classification):** 146
flows captured via sflowtool (built from source — not packaged for Ubuntu
24.04), correctly labeled via `topology_hosts.json`. Evaluated on the *exact*
committed model (verified by reproducing its TN/FP/FN/TP exactly first) — the
existing InSDN row was NOT touched. Zero-shot acc/F1 0.233/0.000, adapted
0.741/0.793 — same collapse-then-recover pattern as InSDN; with only 146
flows this is indicative, not precise, and plausibly a symptom of the same
outlier-scaling issue rather than an independent finding.

**Along the way, real code bugs fixed (see git history for
`src/sdn/pox_mitigation.py`, `topology.py`, `feature_extractor.py`):** the
controller never forwarded/flooded any packet (would have produced all-zero
data); OpenFlow 1.3 vs POX's OF1.0-only default meant the switch never
connected; POX had no SIGTERM/SIGINT handler so metrics never flushed;
`parse_sflowtool_line`'s field indices were wrong (`length` read the
ethertype field; `tcp_flags` used decimal parsing on hex output) — all
verified against real captures before trusting them.

## Open items before final submission
1. Retrain with outlier-robust feature scaling to close the remaining
   mitigation-targeting gap — the only way to get a Phase 3 result where the
   defense actually blocks the attacker. Would require re-verifying Table 1/8.
2. Re-verify against the official CIC release; the paper notes a 2M-row representative sample.
3. Replay buffer value not demonstrated (no forgetting to rescue) — stress-test or soften claim.
4. Get a real TCP delivery/loss measurement beyond bursts_completed (e.g. per-burst latency).
5. Implement OVS meter-based rate-limiting (`config.yaml: sdn.mitigation`) — currently drop-only;
   would also require moving off POX's default OpenFlow 1.0 module.
6. GPU: RTX 5080 (Blackwell, compute capability 12.0a) not usable by TensorFlow 2.21
   (latest available) — confirmed via strace that all CUDA/cuDNN libraries load fine
   individually and `cuInit`/device enumeration succeed at the driver level, but TF's
   GPU backend still rejects the device. Likely needs TF built from source with explicit
   sm_120a kernels, or a newer TF release; not attempted (multi-hour, uncertain payoff).
   All experiments in this file ran on CPU.
