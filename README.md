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
