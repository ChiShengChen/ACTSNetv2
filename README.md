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
├── train.py                        # Training script
├── evaluate.py                     # Evaluation + interpretability
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

## References

- **ACTSNet v1 thesis:** <https://tdr.lib.ntu.edu.tw/handle/123456789/82206> (doi:10.6342/NTU202101201)
- **TapNet:** Zhang et al., "TapNet: Multivariate Time Series Classification with Attentional Prototypical Network," AAAI 2020
- **PatchTST:** Nie et al., "A Time Series is Worth 64 Words," ICLR 2023
- **RevIN:** Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift," NeurIPS 2021
- **iTransformer:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," ICLR 2024
- **Hyperbolic Neural Networks:** Ganea et al., NeurIPS 2018
- **Supervised Contrastive Learning:** Khosla et al., NeurIPS 2020
- **Prototypical Networks:** Snell et al., NeurIPS 2017
