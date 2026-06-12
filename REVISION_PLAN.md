# Revision Plan — SDN DDoS Paper (Resubmission to IJIS)

**Goal:** Convert a saturated "lightweight CNN-LSTM on CICDDoS2019" paper into a novel,
defensible contribution for a Q1 venue (International Journal of Intelligent Systems).

**New framing / working title:**
*"An Adaptive, Closed-Loop CNN-LSTM Framework for Real-Time DoS/DDoS Detection and
Mitigation in Software-Defined Networks under Concept Drift."*

The novelty is no longer the architecture — it is (A) online adaptivity to drift,
(B) a measured closed-loop detect-and-mitigate system, and (C) cross-dataset generalization.

---

## 1. The three contributions (state these verbatim in the Introduction)

**C1 — Adaptive / continual-learning detector (Direction A).**
A drift-aware detector that incrementally updates *on the SDN controller* as traffic
distribution shifts, instead of the offline train-once paradigm used by all prior work.

- Drift detector (ADWIN or DDM) monitors prediction error / confidence on the live stream.
- On drift, trigger incremental fine-tuning on a small buffered batch of recently-labeled
  flows, using **rehearsal/replay** (a reservoir buffer of past samples) to prevent
  catastrophic forgetting. Keep the CNN-LSTM backbone, update lightweight head + partial layers.
- Claim: first online, drift-triggered, continually-updating DDoS detector at the SDN controller.

**C2 — Closed-loop detection + automated mitigation (Direction B).**
Move from classification-only to an end-to-end system: on detection, push OpenFlow
drop/rate-limit rules to OVS via POX, and **measure** what no cited paper measures.

**C3 — Cross-dataset / zero-day generalization (Direction C).**
Train on CICDDoS2019, evaluate zero-shot on a *second SDN-specific dataset* (InSDN) and on
the live testbed captures, with and without short adaptation. Proves generalization, which
same-dataset evaluations cannot.

---

## 2. Mandatory methodology fixes (independent reject triggers — must fix regardless)

1. **SMOTE leakage.** Apply SMOTE/undersampling **only inside the training fold, after the
   train/test split.** Never resample the test set. (Current text resamples then splits.)
2. **Realistic test distribution.** Evaluate on the natural imbalanced distribution. Report
   **PR-AUC, MCC, balanced accuracy, per-class recall** — not just accuracy/F1 on a 50/50 set.
3. **Fair baselines.** Re-implement the strong competitors on *your identical split*:
   plain CNN, plain LSTM, Elubeyd-style CNN-LSTM, a small Transformer, and 1-2 ML models.
   Do NOT quote other papers' headline numbers as the comparison.
4. **Statistical rigor.** ≥5 seeds (or k-fold); report mean ± std and a significance test
   (e.g., paired t-test / Wilcoxon) vs the closest baseline.
5. **Substantiate "lightweight".** Report parameter count, model size (MB), FLOPs, and
   **measured inference latency + throughput** for your model AND every baseline.
6. **Remove the autoencoder contradiction** (related work claims an AE that the model lacks).
7. **Fix all leftover template content** (see Section 5).
8. **Open-source claim:** publish an actual repo (GitHub) with code + preprocessing, or
   delete the "open-source" wording. Reconcile with the data-availability statement.

---

## 3. Experiment plan (full re-run)

### Phase 0 — Clean baseline (redo current results, correctly)
- Re-split CICDDoS2019 with leakage-free pipeline; retrain CNN-LSTM.
- Report on imbalanced test set with the expanded metric set.
- Build the fair baseline table (CNN, LSTM, CNN-LSTM(Elubeyd), Transformer, RF/XGBoost).
- Latency/param/FLOPs table for all models.

### Phase 1 — C3: Cross-dataset generalization
- Acquire **InSDN** (SDN-specific). Map common flow features to your 8 (or a shared subset).
- Zero-shot: train CICDDoS2019 → test InSDN, and train InSDN → test CICDDoS2019.
- Report degradation; this becomes the motivation for C1 (adaptation closes the gap).

### Phase 2 — C1: Drift / continual learning
- Build a streaming protocol: order attack families into a temporal stream (e.g., introduce
  LDAP → NetBIOS → MSSQL → SYN → UDP-Lag sequentially; or CICDDoS2019 → InSDN domain shift).
- Compare **static model vs adaptive model** over the stream:
  - prequential (test-then-train) accuracy / F1 over time,
  - **forgetting measure** (drop on earlier classes),
  - **recovery time** after each drift point,
  - update cost (time per incremental update, samples needed).
- Ablate: replay buffer size, with/without drift detector, frozen vs partial backbone.

### Phase 3 — C2: Closed-loop mitigation on the testbed
On Mininet/POX/OVS/sFlow, with attacks beyond just SYN flood (add UDP flood, low-rate/Shrew):
- **Detection→mitigation pipeline:** flag malicious source → install OpenFlow drop/meter rule.
- Measure, attack ON, defense ON vs OFF:
  - controller CPU & memory,
  - **Packet-In rate** to controller (should collapse after mitigation),
  - **time-to-detect** and **time-to-mitigate** (ms),
  - legitimate-traffic throughput / RTT during attack (availability preserved?),
  - inference latency per flow, flows/sec sustained.
- This is the section that turns "real-time/lightweight" from a claim into evidence.

### Phase 4 — Robustness (optional stretch, strengthens novelty)
- Low-rate/Shrew attacks specifically (CNN-LSTM temporal modeling should shine here; you
  mention Shrew in the intro but never evaluate it).

---

## 4. Paper restructure

- **Abstract / Intro:** rewrite around C1-C3. Drop "we stack CNN+LSTM" as the selling point.
- **Related work:** add a gap-matrix table with columns **Online/Adaptive? | Closed-loop
  mitigation? | Cross-dataset eval? | Latency/overhead reported? | SDN-specific dataset?**
  Show every prior work = mostly "No", yours = "Yes". This is how you *visually prove* novelty.
- **Methodology:** new subsections — (i) drift detection + incremental update with replay;
  (ii) controller-side mitigation logic + OpenFlow rule installation.
- **Experiments:** new sections for drift evaluation, mitigation/overhead, cross-dataset.
- **Add a "Threats to Validity" subsection** (dataset bias, testbed scale, label availability
  for online updates) — reviewers reward this.
- **Conclusion:** reframe around the adaptive closed-loop system.

---

## 5. Template / presentation cleanup (fast, do early)

Remove all leftover AIChE/dental/oncology placeholder content from `Optimal-Design-layout.tex`:
- `\specialissue{...Tooth Extraction...}` → correct or remove
- `\journal{AIChE Journal}`, `\subarticletype{Particle Technology and Fluidization}`
- `\authormark{TAYLOR et al.}` → correct authors; `\titlemark{PLEASE INSERT YOUR ARTICLE TITLE HERE}`
- `\abbr{5-FU, 5-fluorouracil; ... GBM, glioblastoma...}` → real abbreviation list
- `\contributed{Hao Zhang and Pengyue D. Guo...}`, `\dedicated{...Ramanujan...}`
- `\copyright{... AIChE Journal ...}` → correct journal
- `\transtitle`, `\transkeywords{DEM | flexible cylindrical particle...}` → remove/replace
- Fix typos: *Adm→Adam, bacth→batch, exixting→existing, F-1 sore→F1-score, Matric→Metric,
  Mode→Model, "seekers"→"researchers"*.
- Reconcile inconsistent numbers (abstract 99.5% F vs body 99.4% vs commented 99.95%).
- **Confirm the target journal's template** — if submitting to IJIS, use the IJIS/Wiley
  template, not the AIChE one.

---

## 6. Risk / scope notes
- C1 (continual learning) assumes labels for online updates — address with a discussion of
  semi-supervised/confidence-based labeling, or frame updates as periodic operator-confirmed.
- InSDN feature mapping may not be 1:1 with the 8 features — plan a shared-feature subset.
- Keep one coherent narrative; don't present three disconnected experiments.
