# ACTSNet v2 — Frequency-Aware Interpretable EEG Classification for TMS Response Prediction

**ACTSNet v2** is an evolution of the original [ACTSNet](../ACTSNet/), which was proposed as part of a 2021 NTU master's thesis. While preserving the core prototypical learning paradigm, v2 upgrades every module with 2023--2026 era techniques for improved accuracy and clinical interpretability.

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
Input (B, 35, T)                            Input (B, 7, 5, T)
  │                                           │
  ├─ MultiScaleEncoder                        ├─ RevIN normalize
  │   Random dim permutation                  │
  │   → shared Conv1D+BN+ReLU                ├─ PatchEmbedding
  │   → GlobalPool → Concat                  │   → (B, 7, 5, N, d_model)
  │                                           │
  ├─ ACEncoder                                ├─ SubBandFusion
  │   Conv1D×3 + IN + PReLU                  │   → (B, 7, N, d_model)
  │   → Softmax ⊙ Mul                       │
  │   → FC → Sigmoid+IN → Pool               ├─ ChannelAttention
  │                                           │   → (B, 7, N, d_model)
  └─ Concat → FC                              │
       → Euclidean Prototypical               ├─ SpatialSpectralGraph (GCN)
       → NLLLoss                              │   → (B, 7, N, d_model)
                                              │
                                              ├─ FreqLens Attention ×2
                                              │   → (B, 7*N, d_model)
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
| **Input shape** | `(B, 35, T)` — flattened | `(B, 7, 5, T)` — electrodes × sub-bands | Preserve spatial-spectral structure |
| **Feature extraction** | 3× Conv1D + InstanceNorm + PReLU | **PatchEmbedding** (PatchTST-style) | Better long-range temporal modeling |
| **Normalization** | InstanceNorm1d | **RevIN** (reversible instance norm) | Handle EEG baseline drift across subjects |
| **Attention** | `Softmax(X) ⊙ X` (channel softmax) | **FreqLens** (FFT → learnable filter → gate → iFFT) | Explicit frequency-domain discrimination |
| **Sub-band handling** | 5 sub-bands flattened into 35 channels | **SubBandFusion** (learnable δ/θ/α/β/γ attention) | Discover which frequency bands matter for MDD |
| **Channel encoding** | Random dimension permutation + shared Conv | **ChannelAttention** (multi-head + electrode embedding) | Learn spatial electrode relationships |
| **Spatial modeling** | None | **SpatialSpectralGraph** (GCN on 10-20 topology) | Model inter-electrode connectivity |
| **Classification** | Euclidean prototypical (`-‖f(x)−hₖ‖²`) | **Hyperbolic prototypical** (Poincare geodesic distance) | Better class hierarchy representation |
| **Prototype computation** | Recomputed each forward from batch (per-class `wₖ`, `Vₖ`) | Learnable tangent-space prototypes + momentum update | More stable training |
| **Loss function** | NLLLoss | **CrossEntropy + Supervised Contrastive Loss** | Better embedding separation |
| **Optimizer** | Adam | **AdamW** + CosineAnnealingWarmRestarts + grad clip | Regularization + stable hyperbolic training |
| **Interpretability** | None | **InterpretabilityModule** (band/electrode/freq attribution + clinical report) | Clinical transparency |
| **Preprocessing** | External (load `.npy`) | Built-in `EEGPreprocessor` (MNE + ICA + FIR) | End-to-end pipeline |

### Key Design Decisions

1. **Why separate electrodes and sub-bands?** v1 flattens 7×5=35 channels, losing the distinction between spatial (electrode) and spectral (sub-band) dimensions. v2 keeps them separate so SubBandFusion and ChannelAttention can operate on their respective axes independently.

2. **Why FreqLens instead of simple softmax attention?** The original AC module (`Softmax(X) ⊙ X`) operates entirely in the time domain. FreqLens computes attention in the frequency domain via FFT, directly capturing which frequency components are discriminative for MDD — a natural fit for EEG data.

3. **Why hyperbolic prototypes?** Euclidean distance treats all directions equally. Hyperbolic (Poincare) space naturally encodes hierarchical relationships and can better separate responder/non-responder clusters, especially with limited clinical data.

4. **Why add SupCon loss?** With small clinical datasets (typical in MDD-EEG), supervised contrastive loss helps learn better-separated embeddings before the prototype classification head.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Input shape: `(N, 7, 5, T)` — N segments, 7 electrodes (FP1/FP2/F7/F3/Fz/F4/F8), 5 sub-bands (delta/theta/alpha/beta/gamma), T timesteps.

```python
from preprocessing import EEGPreprocessor

proc = EEGPreprocessor(sfreq=256, segment_sec=10)
segments = proc.process('data/raw/patient001.edf')  # -> (N, 7, 5, 2560)
```

## Training

```bash
python train.py \
    --data_path data/processed/rtms_eeg_processed.npy \
    --labels_path data/processed/rtms_labels.npy \
    --task rtms \
    --epochs 200 \
    --batch_size 32 \
    --d_model 128 \
    --patch_len 32
```

| Parameter | Default | Description |
|---|---|---|
| `--task` | rtms | `rtms` or `itbs` |
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
    --checkpoint checkpoints/actsnet_v2_rtms_best.pt \
    --data_path data/processed/rtms_test.npy \
    --labels_path data/processed/rtms_test_labels.npy \
    --output results/rtms_results.json
```

Reports accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, per-sample predictions, and clinical interpretability reports (frequency band importance, electrode importance, channel connectivity).

## Project Structure

```
ACTSNetv2/
├── model.py                        # Full ACTSNetV2 assembly
├── dataset.py                      # EEGDataset + EEGAugmentation
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
│   ├── channel_attention.py        # Multi-head electrode attention
│   ├── spatial_spectral_graph.py   # GCN on electrode topology
│   ├── hyperbolic_proto.py         # Poincare ball prototypical head
│   └── interpretability.py         # Clinical attribution module
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
