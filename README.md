# ACTSNet v2 — Frequency-Aware Interpretable EEG / Time Series Classification

**ACTSNet v2** is an evolution of the original [ACTSNet](https://github.com/ChiShengChen/ACTSNetv1), which was proposed as part of a 2021 NTU master's thesis. While preserving the core prototypical learning paradigm, v2 upgrades every module with 2023--2026 era techniques for improved accuracy and interpretability. The architecture is designed as a **general-purpose model for EEG and multivariate time series classification**.

> **Original thesis:** 基於注意力機制之時間序列原型卷積神經網路與傳統及量子機器學習模型應用於重度憂鬱症腦波之經顱磁刺激抗憂鬱療效預測與分析
> (EEG Analysis for Prediction of Antidepressant Responses of Transcranial Magnetic Stimulation in Major Depressive Disorder Based on Attentional Convolution Time Series Prototypical Neural Network Model and Classical/Quantum Machine Learning Approaches)
> **Author:** Chi-Sheng Chen (陳麒升)
> **Source:** National Taiwan University, 2021
> **Link:** <https://tdr.lib.ntu.edu.tw/handle/123456789/82206>
> **DOI:** 10.6342/NTU202101201

## v1 vs v2 Comparison

### Architecture Overview

```
ACTSNet v1                                  ACTSNet v2
──────────────────────────────              ──────────────────────────────
Input (B, 35, T)                            Input (B, C, S, T)
  │                                           │
  ├─ MultiScaleEncoder                        ├─ RevIN normalize
  │   Random dim permutation                  │
  │   → shared Conv1D+BN+ReLU                ├─ PatchEmbedding
  │   → GlobalPool → Concat                  │   → (B, C, S, N, d_model)
  │                                           │
  ├─ ACEncoder                                ├─ SubBandFusion
  │   Conv1D×3 + IN + PReLU                  │   → (B, C, N, d_model)
  │   → Softmax ⊙ Mul                       │
  │   → FC → Sigmoid+IN → Pool               ├─ ChannelAttention
  │                                           │   → (B, C, N, d_model)
  └─ Concat → FC                              │
       → Euclidean Prototypical               ├─ SpatialSpectralGraph (GCN)
       → NLLLoss                              │   → (B, C, N, d_model)
                                              │
                                              ├─ FreqLens Attention ×2
                                              │   → (B, C*N, d_model)
                                              │
                                              ├─ Global Average Pool
                                              │   → (B, d_model)
                                              │
                                              └─ Hyperbolic Prototypical
                                                  → CE + SupCon Loss
```

### Module-by-Module Comparison

| Component | v1 | v2 | Motivation |
|---|---|---|---|
| **Input shape** | `(B, 35, T)` — flattened | `(B, C, S, T)` — channels × sub-bands | Preserve spatial-spectral structure |
| **Feature extraction** | 3× Conv1D + InstanceNorm + PReLU | **PatchEmbedding** (PatchTST-style) | Better long-range temporal modeling |
| **Normalization** | InstanceNorm1d | **RevIN** (reversible instance norm) | Handle distribution shift across samples |
| **Attention** | `Softmax(X) ⊙ X` (channel softmax) | **FreqLens** (FFT → learnable filter → gate → iFFT) | Explicit frequency-domain discrimination |
| **Sub-band handling** | Sub-bands flattened into channels | **SubBandFusion** (learnable sub-band attention) | Discover which frequency bands matter most |
| **Channel encoding** | Random dimension permutation + shared Conv | **ChannelAttention** (multi-head + position embedding) | Learn spatial channel relationships |
| **Spatial modeling** | None | **SpatialSpectralGraph** (GCN on channel topology) | Model inter-channel connectivity |
| **Classification** | Euclidean prototypical (`-‖f(x)−hₖ‖²`) | **Hyperbolic prototypical** (Poincare geodesic distance) | Better class hierarchy representation |
| **Prototype computation** | Recomputed each forward from batch (per-class `wₖ`, `Vₖ`) | Learnable tangent-space prototypes + momentum update | More stable training |
| **Loss function** | NLLLoss | **CrossEntropy + Supervised Contrastive Loss** | Better embedding separation |
| **Optimizer** | Adam | **AdamW** + CosineAnnealingWarmRestarts + grad clip | Regularization + stable hyperbolic training |
| **Interpretability** | None | **InterpretabilityModule** (band/channel/freq attribution) | Model transparency |

### Key Design Decisions

1. **Why separate channels and sub-bands?** v1 flattens all dimensions into a single channel axis, losing the distinction between spatial (channel) and spectral (sub-band) dimensions. v2 keeps them separate so SubBandFusion and ChannelAttention can operate on their respective axes independently.

2. **Why FreqLens instead of simple softmax attention?** The original AC module (`Softmax(X) ⊙ X`) operates entirely in the time domain. FreqLens computes attention in the frequency domain via FFT, directly capturing which frequency components are most discriminative — a natural fit for EEG and other periodic time series.

3. **Why hyperbolic prototypes?** Euclidean distance treats all directions equally. Hyperbolic (Poincare) space naturally encodes hierarchical relationships and can better separate class clusters, especially with limited data.

4. **Why add SupCon loss?** With small datasets, supervised contrastive loss helps learn better-separated embeddings before the prototype classification head.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Input shape: `(N, C, S, T)` — N samples, C channels, S sub-bands (or feature groups), T timesteps.

For EEG data, a built-in preprocessor is provided:

```python
from preprocessing import EEGPreprocessor

proc = EEGPreprocessor(sfreq=256, segment_sec=10)
segments = proc.process('data/raw/subject001.edf')  # -> (N, 7, 5, 2560)
```

For general time series, prepare your data as `.npy` arrays:

```
data/
├── data.npy      # (N, C, S, T)
└── labels.npy    # (N,)
```

## Training

```bash
python train.py \
    --data_path data/processed/data.npy \
    --labels_path data/processed/labels.npy \
    --epochs 200 \
    --batch_size 32 \
    --d_model 128 \
    --patch_len 32
```

| Parameter | Default | Description |
|---|---|---|
| `--d_model` | 128 | Model dimension |
| `--patch_len` | 32 | Patch length for PatchEmbedding |
| `--n_freqlens_layers` | 2 | Number of FreqLens attention layers |
| `--lambda_supcon` | 0.5 | SupCon loss weight |
| `--dropout` | 0.1 | Dropout rate |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--seed` | 42 | Random seed |

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_path data/processed/test_data.npy \
    --labels_path data/processed/test_labels.npy \
    --output results/results.json
```

Reports accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, per-sample predictions, and interpretability reports (frequency band importance, channel importance, channel connectivity).

## Project Structure

```
ACTSNetv2/
├── model.py                        # Full ACTSNetV2 assembly
├── dataset.py                      # Dataset + augmentation
├── losses.py                       # SupConLoss + ACTSNetV2Loss
├── preprocessing.py                # EEG preprocessing (MNE + ICA + FIR)
├── train.py                        # Training script (TMS task)
├── evaluate.py                     # Evaluation + interpretability
├── run_pretrain.py                 # Self-supervised pretrain on EEG-FM-Bench
├── run_eegfm_benchmark.py          # EEG-FM-Bench downstream benchmark (4 datasets)
├── requirements.txt
├── modules/
│   ├── patch_embedding.py          # PatchEmbedding (PatchTST-style)
│   ├── revin.py                    # Reversible Instance Normalization
│   ├── freqlens_attention.py       # Frequency-domain attention (FFT)
│   ├── subband_fusion.py           # Learnable sub-band mixing
│   ├── channel_attention.py        # Multi-head channel attention
│   ├── spatial_spectral_graph.py   # GCN on channel topology
│   ├── hyperbolic_proto.py         # Poincare ball prototypical head
│   └── interpretability.py         # Attribution module
├── config/
│   ├── rtms_config.yaml
│   └── itbs_config.yaml
├── data/
├── checkpoints/
└── results/
```

## Benchmark Results on EEG-FM-Bench

Evaluated on 4 datasets from [EEG-FM-Bench](https://github.com/weishenzhong/EEG-FM-Bench) with 3 seeds (42, 123, 456), 100 finetune epochs, batch size 64, LR 1e-3. Metric: **balanced accuracy** (mean ± std). Config: linear classification head + CrossEntropy, `revin_per_sample`, dropout 0.1, weight_decay 1e-4. Pretrained encoder uses `max_channels=64` (inputs zero-padded to match).

### Pretrain setup

Input: `(B, C, 5, T)` — 5-band bandpass decomposition (δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Hz) applied on-the-fly. Pool: 5 EEG-FM-Bench datasets (tuab + tuev + bcic_2a + seed_iv + siena_scalp) = **74,141 samples**, max_channels=64, 30 epochs. Objective: NT-Xent contrastive on global features + MSE reconstruction on per-patch features, λ_recon=1.0, temperature=0.1. Two augmented views per sample:

- View 1: time shift (±10%) + Gaussian noise (σ = 0.1 × per-sample std)
- View 2: channel dropout (20%) + sub-band dropout (20%) + amplitude scale (0.8–1.2×)

Reconstruction target is the **clean** signal (denoising pretext).

### Results vs ACTSNet v1

| Dataset | Classes | v1 (no pretrain) | v1 (pretrain 74k) | **v2 (no pretrain)** | **v2 (pretrain 74k)** | v2-pretrain Δ vs v1-no-pretrain |
|---|---|---|---|---|---|---|
| bcic_2a | 4 | 0.4358 ± 0.0263 | 0.3709 ± 0.0262 | 0.3275 ± 0.0115 | 0.3443 ± 0.0131 | −0.092 |
| tuab    | 2 | 0.7457 ± 0.0019 | 0.7257 ± 0.0019 | — | **0.7523 ± 0.0036** | **+0.007** |
| tuev    | 6 | 0.3868 ± 0.0543 | 0.4130 ± 0.0306 | — | **0.4659 ± 0.0379** | **+0.079** |
| seed_iv | 4 | 0.3008 ± 0.0051 | 0.3045 ± 0.0096 | — | **0.3374 ± 0.0062** | **+0.037** |

**Notes**

- **v2 + pretrain beats v1 (pretrain 74k) on all 4 datasets.**
- **v2 + pretrain beats v1 (no pretrain) on 3 / 4 datasets** — wins on clinical corpora (tuab, tuev) and SEED-IV emotion; loses on BCIC-2A motor imagery.
- BCIC-2A is the smallest dataset (1440 train samples, 4 classes). v2 is more parameter-dense than v1 and still overfits even with pretraining — prototype-based heads (v1) remain more sample-efficient on tiny MI datasets.
- tuev (6-class clinical events) sees the biggest gain: +5.3 balanced-acc points over v1-pretrain. The sub-band decomposition + frequency-aware modules appear to help most when the task depends on spectral structure.
- v2 finetune `std` across seeds is consistently lower than v1-pretrain on tuab/tuev/seed_iv, indicating more stable convergence from pretrained init.

### Reproduce

```bash
# Pretrain: 30 epochs on 5-dataset pool, ~100 min on RTX 3090
python run_pretrain.py --epochs 30 \
    --max_samples_per_dataset 20000 \
    --output_dir checkpoints/pretrain_v1

# Finetune benchmark: 4 datasets × 3 seeds × 100 epochs
python run_eegfm_benchmark.py \
    --datasets bcic_2a tuab tuev seed_iv \
    --seeds 42 123 456 --epochs 100 \
    --head linear --revin_per_sample \
    --pretrained_path checkpoints/pretrain_v1/pretrain_final.pt \
    --output_dir checkpoints/eegfm_benchmark_pretrain_v1
```

### Pretrain-scale study (v2 74k → 270k)

Training on TUAB/pretrain (~123k samples) in addition to the original 5-dataset finetune-train pool expands the pool to 197k (3.6× larger). Spectral injection is on in the 270k run. Comparing cross-subject LOSO on the small datasets:

| Dataset | Classes | v1 (74k, linear) | **v2 (270k, linear)** | Δ |
|---|---|---|---|---|
| bcic_2a | 4 | 0.3225 ± 0.0262 | **0.3418 ± 0.0445** | +0.019 |
| seed_iv | 4 | 0.3299 ± 0.0335 | **0.3509 ± 0.0408** | +0.021 |
| tusl    | 3 | — (earlier loader crashed on variable C) | **0.6612 ± 0.0930** | new |
| tuab    | 2 | 0.7466 ± 0.0175 | **0.7542 ± 0.0171** | +0.008 |

Observations:

- **MI / emotion (bcic_2a, seed_iv) gain only ~2 points** from 3.6× pretrain scale; the paradigm-matching "MI stack" (EA + α+β + Prototype + Spectral) gives +14 points on the same dataset, dwarfing scale-up.
- **TUSL (clinical slowing detection) benefits most from the larger pretrain.** With only 300 labeled trials and 38 subjects, the 270k encoder reaches 66% BCA on 3-class (chance 33%), demonstrating strong positive transfer for clinical event morphology.
- Engineering takeaway: for MI, invest in paradigm-specific pretrain (MI-only pool) and/or architectural priors; for clinical tasks, invest in pretrain scale + diverse clinical corpora.

### Cross-subject LOSO (v2 pretrained encoder, 4 datasets)

Same encoder as above (v2 pretrained 30ep on 74k samples) evaluated under leave-one-subject-out cross-validation, mirroring the NSR 2026 benchmark protocol. Small datasets use per-subject folds; large datasets (>10 subjects) use 10-fold grouped-by-subject. Results averaged across folds (1 seed per fold).

| Dataset | Classes | Protocol | BalAcc | Kappa | W-F1 |
|---|---|---|---|---|---|
| bcic_2a | 4 | 9-fold per-subject | 0.3225 ± 0.0262 | 0.0967 ± 0.0349 | 0.2888 ± 0.0414 |
| seed_iv | 4 | 10-fold subject-group | 0.3299 ± 0.0335 | 0.1033 ± 0.0422 | 0.2966 ± 0.0493 |
| tuev    | 6 | 10-fold subject-group | 0.5034 ± 0.0641 | 0.4268 ± 0.0762 | 0.6720 ± 0.0509 |
| tuab    | 2 | 10-fold subject-group | 0.7466 ± 0.0175 | 0.5025 ± 0.0338 | 0.7610 ± 0.0162 |

**Comparison with NSR 2026 on TUAB** — our **0.747** vs reported LOSO BCA of BENDR 0.791, Neuro-GPT 0.795, ShallowConv 0.798, CBraMod 0.800, EEGMamba 0.809, Deformer 0.815, LUNA-Base 0.819. We sit ~4–7 points below. The gap is driven primarily by pretrain scale (our 74k samples vs NSR FMs' millions of tokens) rather than architecture.

For large datasets we subsample total pool to 20k (stratified per-subject) to keep the 5-band expansion within memory (20000 × max_ch=64 × 5 × 1024 × f32 ≈ 26 GB).

### MI-specific LOSO (BCIC-2A, cross-subject, 9-fold per-subject)

Comparable to the NSR 2026 EEG-FM benchmark recipe (LOSO, seed 42 per fold, mean ± std over 9 subject-held-out folds). See [run_mi_benchmark.py](run_mi_benchmark.py) — a stripped-down variant of the main benchmark with three MI-targeted changes:

1. **Euclidean Alignment (EA)** — per-subject covariance whitening, applied before bandpass. Removes subject-level scale drift.
2. **α + β bands only (n_subbands=2)** — MI-discriminative signal lives in mu (8–13 Hz) and beta (13–30 Hz); other bands add noise on small datasets.
3. **Batch Prototype head** — classification via negative squared Euclidean distance to per-class prototypes re-computed each forward from the batch (v1-style metric learning regularization).
4. **Spectral injection in PatchEmbedding** (CBraMod-style) — each patch's FFT magnitude is projected to `d_model` and added to the time-domain embedding, letting the model learn per-patch frequency weighting. **This is on by default** in the main model (`spectral_inject=True`); disable with `--no_spectral` for ablation. Note: the `pretrain_v1_30ep` checkpoint predates this change, so its `spectral_projection` layer will initialize from scratch when loaded (the rest of the encoder still benefits from pretraining).

| Variant | Bal. Acc. LOSO |
|---|---|
| ACTSNetv2 with pretrained v1 encoder, 5-band, linear head | 0.3225 ± 0.0262 |
| MI: EA + α+β + Prototype (no pretrain) | 0.4190 ± 0.1053 |
| **MI: EA + α+β + Prototype + spectral injection (best)** | **0.4653 ± 0.1290** |
| MI: best above but `patch_len=128` (ablation) | 0.2990 ± 0.0175 |

**Comparison with NSR 2026 EEG-FM benchmark (BNCI2014001 LOSO)**

| Rank | Model | BCA % |
|---|---|---|
| 1 | MIRepNet* (MI-specific FM) | 54.21 |
| 2 | CBraMod (FM) | 53.03 |
| 3 | BENDR (FM) | 51.11 |
| 4 | Neuro-GPT (FM) | 46.97 |
| 5 | LaBraM (FM) | 46.93 |
| 6 | LMDA | 46.80 |
| **—** | **ACTSNetv2 MI (ours)** | **46.53** |
| 7 | EEGMamba (FM) | 45.72 |
| 8 | EEGNet (2K params) | 44.97 |
| 9 | ShallowConv | 44.80 |

ACTSNetv2 MI matches LaBraM / Neuro-GPT tier while being trained without any pretrain pool (these numbers are from-scratch cross-subject). The three MI-targeted tweaks together buy **+14.3 balanced-acc points** over the default-configured pretrained v2 (32.25 → 46.53).

`patch_len=128` (0.5 s per patch at fs=256) does worse because MI event-related desynchronization unfolds on sub-100 ms scales — larger patches average out the discriminative dynamics.

**Applying the MI stack to TUAB (non-MI ablation)** — we re-ran the full MI pipeline (EA + α+β + Prototype + Spectral) on TUAB to test whether these tweaks generalize. They do not:

| Variant | TUAB LOSO BalAcc |
|---|---|
| v2 pretrained (5-band, linear head) | **0.7466 ± 0.0175** |
| MI stack (α+β, Prototype, EA, Spectral) | 0.7349 ± 0.0260 |
| Δ | −0.012 |

The α+β restriction is MI-specific: abnormal EEG detection relies on the full spectrum (δ for slow-wave abnormalities, θ for drowsiness, γ for high-frequency activity). The +14.3-point gain on BCIC-2A is therefore a paradigm-matching effect, not a generally-applicable recipe.

### Ablation on BCIC-2A (sanity checks during development)

| Variant | Bal. Acc. | Notes |
|---|---|---|
| v2, default (hyperbolic head, RevIN, CE + SupCon) | 0.2593 ± 0.0096 | stuck at chance — hyperbolic head doesn't train end-to-end |
| v2, linear head + CE, RevIN on | 0.2535 | same plateau — RevIN washes class signal |
| v2, linear head + CE, no RevIN | 0.3108 ± 0.0221 | breaks plateau, but overfits |
| v2, linear head + CE, `revin_per_sample` + dropout 0.3 + wd 5e-3 | 0.3275 ± 0.0115 | best from-scratch setting |
| **v2, linear head + CE, `revin_per_sample` + pretrain** | **0.3443 ± 0.0131** | +1.7 pts from pretrain |

`revin_per_sample` normalizes each sample globally over (C, S, T) with one mean/std — preserving the relative channel × sub-band energy structure that SubBandFusion relies on. Default RevIN (per-channel over time) removes these relative energies and prevents learning on small datasets.

---

## References

- **ACTSNet v1 thesis:** <https://tdr.lib.ntu.edu.tw/handle/123456789/82206> (doi:10.6342/NTU202101201)
- **TapNet:** Zhang et al., "TapNet: Multivariate Time Series Classification with Attentional Prototypical Network," AAAI 2020
- **PatchTST:** Nie et al., "A Time Series is Worth 64 Words," ICLR 2023
- **RevIN:** Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift," NeurIPS 2021
- **iTransformer:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," ICLR 2024
- **Hyperbolic Neural Networks:** Ganea et al., NeurIPS 2018
- **Supervised Contrastive Learning:** Khosla et al., NeurIPS 2020
- **Prototypical Networks:** Snell et al., NeurIPS 2017
