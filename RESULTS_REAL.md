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
Pending: requires the InSDN dataset (not yet downloaded).

## Phase 2 — Concept drift (Table tab:drift_results)

Protocol: pre-train on the first family (Syn), then stream the remaining families
(DrDoS_UDP, DrDoS_MSSQL, DrDoS_LDAP, DrDoS_NetBIOS, UDP-lag) prequentially.

| Setting | Preq. Acc | Preq. F1 | Updates | Update ms |
|---|---|---|---|---|
| Static (no update) | 0.2554 | 0.2989 | 0 | — |
| Adaptive, no replay | 0.4998 | 0.6223 | 8 | 387 |
| **Adaptive + replay (proposed)** | 0.5006 | 0.6227 | 8 | 438 |

Per-family final accuracy (stream order):

| Family | Static | Adaptive | Adaptive+replay |
|---|---|---|---|
| DrDoS_UDP | 0.089 | 0.089 | 0.089 |
| DrDoS_MSSQL | 0.127 | 0.127 | 0.127 |
| DrDoS_LDAP | 0.882 | 0.882 | 0.882 |
| **DrDoS_NetBIOS** | **0.071** | **0.954** | **0.957** |
| UDP-lag | 0.931 | 0.930 | 0.933 |

### Honest reading of Phase 2
- **The adaptation works, partially.** Overall prequential accuracy nearly doubles
  (0.255 -> 0.50) and the clearest case is NetBIOS, which the static model fails
  (0.07) and the adaptive model recovers (0.95). That is a real demonstration of the
  mechanism.
- **It is not a clean sweep.** DrDoS_UDP and DrDoS_MSSQL never recover. Root cause: the
  DDM detector needs a low->high error transition to fire; UDP/MSSQL arrive first and
  are hard from the start, so no "spike" registers and recovery never triggers (a
  cold-start weakness). NetBIOS fires because it follows the easy LDAP segment.
- **Replay vs no-replay is indistinguishable here** (0.954 vs 0.957). The protocol
  never re-tests earlier families at the end, so catastrophic forgetting is not
  actually exercised and the replay buffer's value cannot be shown. The forgetting
  metric reads 0 for all settings for the same reason -- it is measured within each
  family's own segment, not by re-evaluation at the end.

### Open items before this table is paper-ready
1. Fix the DDM cold-start (e.g., trigger recovery whenever recent error is high, not
   only on a transition) so UDP/MSSQL also recover.
2. Re-test all earlier families at the END of the stream to actually measure
   forgetting and give the replay buffer something to prove.
3. Consider pre-training on a mix of families (more realistic than train-on-one).

## Phase 3 — Closed-loop mitigation (Table tab:mitigation_results)
Pending: requires the Mininet/POX/OVS testbed (Ubuntu guest), not this machine.
