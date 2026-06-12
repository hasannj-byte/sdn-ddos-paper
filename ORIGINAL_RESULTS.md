# Original Results (as in the IJIS-submitted version)

Recovered from the initial commit (`616add2`), before the revision replaced numbers
with `[XX]` placeholders. These are the authors' actual reported results from the
first submission.

> **Caveat:** these come from the *original* pipeline — SMOTE applied before the
> train/test split and evaluation on a **rebalanced** test set (note the ~50/50
> confusion matrix). They are detection-only; there are no original results for
> drift, mitigation, or cross-dataset because those experiments did not exist in
> the first submission. See "How to use these" at the bottom.

## Proposed CNN-LSTM on CICDDoS2019
| Metric | Value |
|---|---|
| Accuracy | 99.5% |
| Precision | 99.3% |
| Recall | 99.5% |
| F1-score | 99.4% |
| FPR | 0.076% |
| ROC-AUC | 0.99 |
| Parameters | ~80,000 |

(An inconsistency existed in the original: a commented-out comparison table listed
the proposed model as Accuracy 99.95% / Precision 99.93% / Recall 99.97% / FPR 0.076%,
while the body text used 99.5% / 99.3% / 99.5%. The body values are the ones above.)

## Confusion matrix (original, balanced test set)
| | Predicted benign | Predicted attack |
|---|---|---|
| **Actual benign** | TN = 4,281,867 | FP = 2,884 |
| **Actual attack** | FN = 1,110 | TP = 4,283,641 |

## Optimizer / batch-size analysis
| Optimizer | Batch | Accuracy | F1 |
|---|---|---|---|
| Adam | 64 | 99.5% | 99.4% |
| Adam | 128 | 99.40% | 99.25% |
| SGD | 64 | 99.10% | 98.85% |
| SGD | 128 | 98.70% | 98.40% |

Best: Adam + batch size 64.

## Baseline comparison (F1-score, from the cited baseline papers)
| Model | F1 |
|---|---|
| NB | 71.1% |
| RF (P1) | 91.1% |
| SVM | 92.7% |
| DT (P1) | 94.9% |
| KNN | 96.6% |
| CNN-AD | 97.4% |
| DT (P2) | 97.9% |
| RF (P2) | 99.1% |
| **Proposed CNN-LSTM** | **99.4%** |

(Baselines were quoted from the source papers — Narmadha et al. and Saluja et al. —
not re-implemented on the same split.)

## How to use these
- **As a sanity check / reference:** keep these to confirm a re-run of Phase 0 lands
  in the same ballpark for the proposed model.
- **Do NOT paste them back into the revised tables as-is.** They were produced with
  the leakage-before-split protocol the revision explicitly fixes, and on a
  rebalanced test set, so re-using them would reintroduce the exact flaw a reviewer
  would reject. Re-run Phase 0 (`src.experiments.train_detect`) under the
  leakage-free pipeline to get the values that belong in `tab:detection_results`.
- **Drift / mitigation / cross-dataset tables** have no original counterpart and can
  only be filled by running Phases 1–3.
