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

## Phase 3 — Closed-loop mitigation (tab:mitigation_results)
**Pending** — needs the Mininet/POX/Open vSwitch testbed on the VM. Marked `pending` in
the paper (mitigation table + the live-testbed cross-dataset row).

## Open items before final submission
1. Run Phase 3 (mitigation) on the VM testbed to fill the last table + testbed row.
2. Re-verify against the official CIC release; the paper notes a 2M-row representative sample.
3. Replay buffer value not demonstrated (no forgetting to rescue) — stress-test or soften claim.
4. Optimizer/batch sweep (Fig p4) was not re-run; only Adam+batch64 reported from this run.
