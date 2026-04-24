# ACTSNetv2 TODO

Status as of 2026-04-24. Existing LOSO benchmark covers 6 EEG-FM-Bench
datasets (bcic_2a, seed_iv, tusl, tuab, tuev, siena_scalp) with the 270k
pretrained encoder. MI-specific stack (EA + α+β + Prototype + Spectral)
reaches 0.4653 on BCIC-2A LOSO, matching NSR tier-5 (LaBraM ~0.469).

Items are marked with time estimates. Priorities:
- 🔴 Tier 1 — must have before submitting anywhere
- 🟡 Tier 2 — needed to compete at top venues
- 🟢 Tier 3 — engineering / polish

---

## 🔴 Tier 1 — Paper-submittable baseline

### 1. Module ablation table
Prove each v2 module contributes. One run per ablation on a representative
subset of datasets (bcic_2a + tuab + tuev) with the same finetune config.

Ablations to run:
- [ ] `--no_spatial_graph` — drop SpatialSpectralGraph entirely
- [ ] `--no_freqlens` — drop all FreqLens layers
- [ ] `--freqlens_layers 1` — one layer only
- [ ] `--no_subband_fusion` — replace with mean(S)
- [ ] `--no_channel_attention` — replace with mean(C)
- [ ] `--no_revin` vs `--revin_per_sample` vs default (already have some data)
- [ ] `--no_spectral` — drop CBraMod-style spectral injection (already a flag)

Deliverable: one markdown table per dataset with `|module off|BalAcc|Δ|`.
Implementation cost: add `--no_*` flags to run_eegfm_benchmark.py, ~30 lines.
Run cost: 7 ablations × 3 datasets × ~1-7 h = 40-60 h total compute.

### 2. Hyperbolic head: fix or remove
Current state: README describes HyperbolicPrototypicalHead but every benchmark
run uses LinearHead because hyperbolic head collapses to chance during training
(prototypes updated via `.data[k] = EMA` bypasses gradient; SupCon re-normalises
Poincare-ball embeddings to unit sphere, destroying hyperbolic geometry).

Pick one:
- [ ] **Fix**: prototypes as pure `nn.Parameter` learned by backprop; SupCon on
  pre-projection features (not Poincare ball); proper log/exp map consistency.
- [ ] **Remove**: make LinearHead the documented default; move HyperbolicProto
  to an `experiments/` subdirectory; clarify in README.

Cost: fix option ≈ 100 LoC + new ablation runs; remove option ≈ 20 LoC + doc
rewrite.

### 3. NSR-style comparison table
Populate a table comparing ACTSNetv2 to NSR-2026 baselines across all 6
benchmarked datasets. Pull the per-model numbers from the NSR paper / tables.

- [ ] Create `docs/nsr_comparison.md` with format:

  ```
  | Model           | BCIC-2A | SEED-IV | TUSL | TUAB | TUEV | Siena |
  |-----------------|---------|---------|------|------|------|-------|
  | CBraMod         | 53.03   | ...     | ...  | ...  | ...  | ...   |
  | LaBraM          | 46.93   | ...     | ...  | ...  | ...  | ...   |
  | EEGNet          | 44.97   | ...     | ...  | ...  | ...  | ...   |
  | ...             | ...     | ...     | ...  | ...  | ...  | ...   |
  | ACTSNetv2 (our) | 46.53   | 35.09   | 66.12| 75.42| 53.66| 55.83 |
  ```
- [ ] Note protocol differences (fold count, subsampling) in footnotes.

Cost: no compute, ~2 h research + writing.

### 4. siena_scalp class-balance fix
Current result (0.5583 BalAcc, Kappa 0.11, F1 0.99) is a class-collapse
pathology — the model predicts only majority class on most folds.

Options:
- [ ] `--class_weighted_ce` using inverse class frequency.
- [ ] `--focal_loss --focal_gamma 2.0`.
- [ ] Per-batch balanced sampler (equal class draws).

Implementation: ~50 LoC in run_eegfm_benchmark.py.
Re-run cost: siena_scalp LOSO ≈ 7 h.

---

## 🟡 Tier 2 — Competitive numbers for top venues

### 5. Scale pretrain (TUEG corpus)
Current pool: 197 478 samples (tuab/pretrain 123k + 5 finetune splits).
NSR top FMs: 10–100× this (LaBraM 2 500 h EEG, CBraMod 627M tokens).

- [ ] Add TUEG preprocessed shards to pretrain pool
  (source: `/media/meow/Elements/EEGPT/downstream_tueg/` ← only processing
  scripts on disk; need to run them or download processed version).
- [ ] Bump pretrain to 50–100 epochs (currently 30).
- [ ] Target: ~1M samples.

Cost: 1–3 days of preprocessing + 1-2 days of pretraining.

### 6. Linear probe experiments
Standard FM benchmark: freeze encoder, train only the linear head. Measures
how general / transferable the learned features are.

- [ ] Add `--freeze_encoder` ablation run (flag already exists).
- [ ] Report linear-probe vs full-finetune on all 6 datasets.

Cost: 6 datasets × LOSO-frozen ≈ 20 h.

### 7. Within-subject few-shot
NSR reports this for every dataset alongside LOSO. Protocol: LOSO-pretrain,
then fine-tune a few trials (e.g. 10 %) of the target subject and evaluate
on the rest.

- [ ] New runner `run_fewshot_benchmark.py` (~200 LoC).
- [ ] Run on 4–6 datasets.

Cost: ~1 day implementation + ~30 h compute.

### 8. More datasets (catch up with NSR's 13)
Priority additions (available on disk or easy to add):
- [ ] BCIC-2B (`bnci2014004`, 2-class MI, 3 channels)
- [ ] CHB-MIT (2-class seizure, pediatric, bipolar)
- [ ] SEED (3-class emotion, 62 ch) — distinct from seed_iv
- [ ] Sleep-EDFx (5-class sleep staging) — needs 30 s windows, different segmentation
- [ ] Nakanishi2015 (12-class SSVEP) — very different paradigm, good for breadth

Cost: per-dataset ≈ check availability + data preproc + benchmark run; 3–5 days total.

---

## 🟢 Tier 3 — Engineering / polish

### 9. Speed up `load_arrow_all_with_subject` pass 1
Current pass 1 does `np.asarray(table.column("data")[i].as_py())` on every
sample to find max C — took 2.4 h for tuab's 272k samples.

- [ ] Use `len(table.column("chs")[i])` (metadata-only, no data deser).
- [ ] Alternative: assume shape-consistent within a shard, probe one per shard.

Expected: 2.4 h → ~5 min.

### 10. Interpretability figures
`InterpretabilityModule` in `modules/interpretability.py` is wired but never
produces output artifacts.

- [ ] Extract and plot SubBandFusion attention weights per-class on BCIC-2A
  (expect α / β to dominate for MI).
- [ ] Plot FreqLens learned filters (frequency-domain).
- [ ] Plot SpatialSpectralGraph `adj_learnable + adj_norm` as connectivity.

Cost: ~1 day of plotting + writeup.

### 11. Augmentation variety
Current pretrain augmentations are time-shift + noise for view 1 and
channel/sub-band dropout + amplitude scale for view 2.

- [ ] Add SpecAugment (frequency-domain masking).
- [ ] Add Mixup / CutMix at the waveform level.
- [ ] Add time-warping (stretch / compress).

Cost: ~200 LoC; test with a small pretrain re-run.

### 12. Per-dataset batch sampling for pretrain
Current: random mix from ConcatDataset → tuab/pretrain (~125k) dominates
every batch, small datasets rarely seen.

- [ ] Implement a sampler that guarantees one sample from each dataset per
  batch, or proportional-sqrt sampling.

Cost: ~100 LoC + small re-pretrain to validate.

### 13. Documentation pass
- [ ] Consolidate `checkpoints/pretrain_v1_30ep` (v1 74k) naming vs
  `checkpoints/pretrain_v2_270k_30ep` (v2 270k). Currently inconsistent
  (v1/v2 means different things in different places).
- [ ] Add reproducibility section with exact commands per experiment.
- [ ] Separate `experiments/` folder for MI-specific scripts.

---

## Suggested workstream ordering

**Week 1** (paper-submittable baseline):
- Day 1: Task 2 (decide hyperbolic head)
- Day 2–3: Task 1 (module ablation code + initial runs)
- Day 4: Task 4 (siena_scalp class-balance fix)
- Day 5: Task 3 (NSR comparison table writeup)

**Week 2** (competitive numbers):
- Task 9 (loader speed fix — enables faster iteration)
- Task 6 (linear probe — cheap to run, big signal)
- Task 5 (scale pretrain — starts overnight)

**Week 3+**:
- Task 7 (within-subject few-shot)
- Task 8 (more datasets)
- Task 10, 11, 12 (polish)
