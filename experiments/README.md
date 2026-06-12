# Experiments — Adaptive, Closed-Loop CNN-LSTM for SDN DDoS

Code that produces the numbers behind the four result tables in the paper. Each
phase writes a JSON file to `results/`; use those values to replace the `[XX]`
placeholders in `Optimal-Design-layout.tex`.

| Phase | Script | Fills table |
|-------|--------|-------------|
| 0 — Detection (leakage-free) | `src.experiments.train_detect` | `tab:detection_results` |
| 1 — Cross-dataset generalization | `src.experiments.run_cross_dataset` | `tab:cross_dataset_results` |
| 2 — Concept drift / continual learning | `src.experiments.run_drift` | `tab:drift_results` |
| 3 — Closed-loop mitigation & overhead | `src.sdn.*` + `src.experiments.run_testbed` | `tab:mitigation_results` |

## Layout

```
experiments/
  config.yaml              # all paths/hyperparameters (edit dataset paths here)
  requirements.txt
  src/
    data/        preprocess.py (leakage-free split), datasets.py (loaders + feature map)
    models/      cnn_lstm.py (proposed ~80k model), baselines.py (fair re-implementations)
    drift/       drift_detector.py (DDM), replay_buffer.py (reservoir), online_adapt.py (Algorithm 1)
    sdn/         topology.py (Mininet), feature_extractor.py (sFlow->8 feats), pox_mitigation.py (POX engine)
    experiments/ train_detect.py, run_cross_dataset.py, run_drift.py, run_testbed.py
    utils/       common.py, metrics.py
```

## Setup (analysis machine)

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then edit `config.yaml`:
- `data.cicddos2019_dir` — folder of CICDDoS2019 CSVs
- `data.insdn_dir` — folder of InSDN CSVs
- `data.testbed_capture` — where `feature_extractor.py` writes live flows

## Running the offline phases

```bash
python -m src.experiments.train_detect        --config config.yaml   # Phase 0
python -m src.experiments.run_cross_dataset    --config config.yaml   # Phase 1
python -m src.experiments.run_drift            --config config.yaml   # Phase 2
```

Phase 0 also saves the trained model to `results/proposed_cnn_lstm.keras`, which
the SDN controller loads in Phase 3.

## Running the testbed (Phase 3, inside the Ubuntu guest)

Requires Mininet, Open vSwitch, POX 0.7.0, sflowtool, and `hping3`, plus TensorFlow
available to POX's Python. Run each condition once (defense off, then on):

```bash
# Terminal 1 — controller (omit the module for the "defense OFF" baseline)
./pox.py log.level --DEBUG src.sdn.pox_mitigation \
    --model=results/proposed_cnn_lstm.keras --config=config.yaml

# Terminal 2 — topology + attack
sudo python3 -m src.sdn.topology --attack syn --duration 60

# Terminal 3 — overhead monitor (PID of the POX process)
python3 -m src.experiments.run_testbed --controller-pid <POX_PID> \
    --label defense_on --duration 60
```

To use the live testbed traffic as a cross-dataset target, capture it first:

```bash
python3 -m src.sdn.feature_extractor --config config.yaml --duration 120
```

## Important: what to verify before trusting results

These are the spots where the scaffold makes assumptions that must be checked:

1. **`config.yaml: features`** — the eight column names must match your CICDDoS2019
   export exactly (whitespace is stripped automatically).
2. **`data/datasets.py: INSDN_FEATURE_MAP`** — confirm the InSDN column names and fix
   the mapping; cross-dataset results are meaningless if the mapping is wrong.
3. **`sdn/feature_extractor.py: parse_sflowtool_line`** — align field indices with your
   sflowtool version; several features are approximated from sampled sFlow.
4. **`sdn/pox_mitigation.py: classify_source`** — load and apply the persisted
   `StandardScaler` from Phase 0 before inference (a `TODO` marks the line); without
   it the live features are unscaled and predictions will be wrong.
5. **Parameter count** — `build_cnn_lstm(cfg).count_params()` prints the exact figure
   for the "~80,000 parameters" claim; report whatever it actually is.

## Reproducibility

`config.yaml: seed` seeds Python/NumPy/TensorFlow. For the multi-seed runs the paper
promises (mean ± std, significance test), wrap Phase 0/1/2 in a loop over seeds and
aggregate the JSON outputs.
