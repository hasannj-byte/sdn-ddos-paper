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
Running. Results appended here when complete.

## Phase 3 — Closed-loop mitigation (Table tab:mitigation_results)
Pending: requires the Mininet/POX/OVS testbed (Ubuntu guest), not this machine.
