# Real Experiment Results (leakage-free pipeline)

Produced by `experiments/` on real CICDDoS2019 data. These replace the `[XX]`
placeholders in the manuscript once finalized.

## Run provenance (read before using)
- **Data:** CICDDoS2019, day `01-12`, from a non-gated Hugging Face mirror
  (`baalajimaestro/CICDDoS2019`) whose files match the official CICFlowMeter
  structure, sizes, and columns. The official UNB CIC download is registration-gated.
  *For the final submission, confirm against the official release.*
- **Subset:** 6 attack families downloaded (Syn, DrDoS_UDP, DrDoS_MSSQL, DrDoS_LDAP,
  DrDoS_NetBIOS, UDP-lag). Benign is rare in CICDDoS2019 (~11.5k rows across these
  files vs ~16M attack).
- **Tractability cap:** `data.max_rows = 600000`, benign-preserving (keeps all benign,
  subsamples attack). Set to `null` for a full run on the final machine.
- **Protocol:** leakage-free — split first, scaler fit on train, SMOTE on train folds
  only, **test set kept at natural imbalance** (~99% attack). Hardware: macOS, 24 GB
  RAM, CPU-only TensorFlow 2.16.2. `epochs=30` with early stopping.

## Phase 0 — Detection on CICDDoS2019 (Table tab:detection_results)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | MCC | FPR | Params | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.99956 | — | — | 0.99978 | 1.00000 | 0.9771 | 0.088% | 200 trees | 13.54 |
| XGBoost | **0.99992** | — | — | **0.99996** | 1.00000 | **0.9960** | 0.088% | 300 trees | **0.29** |
| CNN (1D) | 0.99944 | — | — | 0.99972 | 1.00000 | 0.9713 | 0.177% | 4,673 | 1.31 |
| LSTM | 0.99952 | — | — | 0.99976 | 0.99999 | 0.9750 | 0.088% | 33,929 | 11.04 |
| Transformer (small) | 0.99913 | — | — | 0.99956 | 1.00000 | 0.9563 | 0.354% | 8,641 | 2.99 |
| CNN-LSTM (full) | 0.99918 | — | — | 0.99958 | 1.00000 | 0.9584 | 0.088% | 46,977 | 11.82 |
| **Proposed (8 feat.)** | 0.99925 | — | — | 0.99962 | 0.99999 | 0.9620 | 0.088% | 46,977 | 11.91 |

Proposed-model confusion matrix (natural-imbalance test set, 120,000 flows):
**TN = 1,130 · FP = 1 · FN = 89 · TP = 118,780** (only ~1,131 benign flows in test).

### Honest reading of Phase 0
- All models exceed 99.9% accuracy because CICDDoS2019 is attack-heavy; **accuracy is
  not discriminating**. MCC is.
- On raw detection the **proposed model is mid-pack and does not win**: XGBoost
  (MCC 0.996, 0.29 ms) and Random Forest (MCC 0.977) clearly beat it, and plain LSTM
  and CNN also score higher. This is expected on a saturated benchmark and is exactly
  why the paper's contribution is framed as adaptivity + mitigation + cross-dataset,
  **not** "best detector."
- FPR is coarse here: with only ~1,131 benign test flows, one false positive = 0.088%,
  so several models tie. A larger benign sample (full dataset) would refine this.

### Parameter count
Confirmed: the proposed architecture has **46,977 trainable parameters** (not the
~80,000 originally stated). The paper has been corrected accordingly.

## Phase 1 — Cross-dataset (Table tab:cross_dataset_results)

Train on CICDDoS2019 (leakage-free), test zero-shot on InSDN (the CICDDoS2019-fitted
scaler applied unchanged), then briefly adapt on 5% of InSDN and re-test on held-out InSDN.
InSDN from HF mirror `Sharukesh/INSDN` (343,889 flows; columns mapped from its
abbreviated CICFlowMeter naming; binary `target` label).

| Train -> Test | Zero-shot Acc | ZS F1 | ZS PR-AUC | Adapted Acc | Adapted F1 |
|---|---|---|---|---|---|
| CICDDoS2019 -> CICDDoS2019 (ref) | 0.9996 | 0.9998 | 1.0000 | - | - |
| **CICDDoS2019 -> InSDN** | **0.3659** | 0.3973 | 0.8784 | **0.9649** | 0.9785 |
| CICDDoS2019 -> Testbed | pending (needs Phase 3 capture) | | | | |

### Honest reading of Phase 1
- **Zero-shot transfer collapses**: 0.9996 (same-dataset) -> 0.366 accuracy on InSDN.
  This is the generalization gap that same-dataset evaluations hide, shown on real data
  and directly supporting the paper's cross-dataset critique.
- **Brief adaptation recovers it**: fine-tuning on 5% of InSDN lifts accuracy to 0.965
  (F1 0.979). The same adaptation mechanism used for drift closes the domain gap.
- **Nuance**: zero-shot PR-AUC stays 0.878 while accuracy is 0.366 -- the model still
  ranks attack vs benign well, but the 0.5 threshold is miscalibrated for InSDN. Part of
  the failure is calibration; adaptation fixes it cleanly. (A threshold recalibration
  baseline would be a fair thing to add.)

## Phase 2 — Concept drift (Table tab:drift_results)

Protocol (v2): split each family into stream/eval parts; pre-train the base model on a
MIX of the first 2 families (Syn, DrDoS_UDP); stream the remaining families
(DrDoS_MSSQL, DrDoS_LDAP, DrDoS_NetBIOS, UDP-lag) prequentially; re-test the final
model on every family's held-out eval set to measure recovery and forgetting. Recovery
is triggered by a smoothed high-error signal with hysteresis (cold-start fix), and the
replay buffer is seeded with base-family data.

| Setting | Preq. Acc | Preq. F1 | Forgetting (base) | Updates | Update ms |
|---|---|---|---|---|---|
| Static (no update) | 0.698 | 0.799 | 0.000 | 0 | - |
| Adaptive, no replay | 0.931 | 0.960 | 0.000 | 2 | 386 |
| **Adaptive + replay (proposed)** | 0.931 | 0.960 | 0.000 | 2 | 392 |

Final per-family accuracy (held-out eval, re-tested at stream end):

| Family | Post-pretrain | Static | Adaptive (+/- replay) | Type |
|---|---|---|---|---|
| Syn | 0.999 | 0.999 | 0.999 | base |
| DrDoS_UDP | 1.000 | 1.000 | 0.999 | base |
| DrDoS_MSSQL | 0.885 | 0.885 | 0.928 | drift |
| DrDoS_LDAP | 0.999 | 0.999 | 0.999 | drift |
| **DrDoS_NetBIOS** | **0.272** | **0.272** | **0.909** | drift |
| UDP-lag | 1.000 | 1.000 | 0.999 | drift |

### Honest reading of Phase 2 (v2)
- **Adaptation works.** Prequential accuracy rises 0.70 -> 0.93, and NetBIOS -- which the
  static model fails (0.27) -- is recovered to 0.91. The cold-start fix resolved the
  earlier failure where hard-from-the-start families never triggered recovery.
- **No forgetting.** Base families (Syn, DrDoS_UDP) stay at ~0.999 after adaptation; the
  lightweight partial-fine-tuning adapts without erasing prior knowledge.
- **Replay vs no-replay is indistinguishable** (0.9307 vs 0.9310), and this is now
  explained rather than mysterious: the gentle adaptation causes no forgetting even
  without replay, so the buffer has nothing to rescue. Replay is a harmless safeguard
  here, not a demonstrated necessity. Showing replay's value would require a more
  aggressive adaptation regime (more steps / unfrozen conv) that induces forgetting --
  a deliberate stress test, noted as optional future work.

## Phase 3 — Closed-loop mitigation (Table tab:mitigation_results)
Pending: requires the Mininet/POX/OVS testbed (Ubuntu guest), not this machine.
