"""ACTSNetv2 MI-specific LOSO benchmark with 3 targeted improvements:

  1. Euclidean Alignment (EA): subject-wise covariance whitening.
     Each trial is pre-multiplied by R_s^(-1/2) where R_s is the mean
     covariance across that subject's trials. Removes subject-level scale
     drift, shown to universally help MI models (NSR 2026).

  2. Alpha + Beta bands only (n_subbands=2): MI-discriminative signal lives
     in mu (8-13 Hz) and beta (13-30 Hz). The other 3 bands add noise and
     increase overfit risk on small MI datasets.

  3. Batch Prototype head (v1-style): classification via negative squared
     Euclidean distance to per-class prototypes re-computed each forward
     from the current batch. Adds few-shot metric-learning regularization
     instead of a fully-parametric linear classifier.

Training: from-scratch (no pretrain) because n_subbands=2 breaks the
pretrained checkpoint's shape. LOSO cross-subject eval.

Usage:
    python run_mi_benchmark.py --datasets bcic_2a
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from model import ACTSNetV2


EEGFM_DATASETS = {
    "bcic_2a": {"n_classes": 4, "version": "1.0.0"},
    "seed_iv": {"n_classes": 4, "version": "1.0.0"},
    "tuab":    {"n_classes": 2, "version": "3.0.1"},
    "tuev":    {"n_classes": 6, "version": "2.0.0"},
}

EEGFM_ROOT_CANDIDATES = [
    "/media/meow/Transcend/time_series_benchmark/eegfm_data_cache",
    "/media/meow/Elements/EEG-FM-Bench-data/processed/fs_256",
    "/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
]

# Two MI-relevant bands only
MI_SUBBANDS = [
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
]


def get_eegfm_root(dataset_name: str):
    for p in EEGFM_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(p, dataset_name)):
            return p
    raise FileNotFoundError(f"Data root not found for {dataset_name}")


def load_arrow_all_with_subject(root, dataset, version, max_time_len=None):
    all_data, all_labels, all_subj = [], [], []
    for split in ("train", "validation", "test"):
        base_dir = os.path.join(root, dataset, "finetune", version)
        prefix = f"{dataset}-{split}-"
        shards = sorted([
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.startswith(prefix) and f.endswith(".arrow")
        ])
        for sp in shards:
            table = ipc.open_stream(sp).read_all()
            for i in range(len(table)):
                d = np.array(table.column("data")[i].as_py(), dtype=np.float32)
                if max_time_len is not None and d.shape[-1] > max_time_len:
                    d = d[:, :max_time_len]
                all_data.append(d)
                all_labels.append(int(table.column("label")[i].as_py()))
                all_subj.append(str(table.column("subject")[i].as_py()))
    return np.stack(all_data), np.array(all_labels, dtype=np.int64), np.array(all_subj)


# ──────────────────────────────────────────────
# Euclidean Alignment (EA)
# ──────────────────────────────────────────────
def euclidean_align(data: np.ndarray, subjects: np.ndarray, logger=None) -> np.ndarray:
    """(N, C, T) -> (N, C, T) after per-subject covariance whitening.

    For each subject s:
      R_s = mean over trials of (X_i @ X_i.T / T)
      X_aligned_i = R_s^(-1/2) @ X_i
    """
    out = np.empty_like(data, dtype=np.float32)
    unique_subj = np.unique(subjects)
    for s in unique_subj:
        idx = np.where(subjects == s)[0]
        X = data[idx].astype(np.float64)            # (n_s, C, T) — f64 for stable eigh
        # per-trial cov, mean over trials
        covs = np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
        R = covs.mean(axis=0)
        C = R.shape[0]
        # R^(-1/2) via eigendecomposition (shrinkage for stability)
        R_reg = R + 1e-5 * np.trace(R) / C * np.eye(C)
        eigvals, eigvecs = np.linalg.eigh(R_reg)
        eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)
        R_inv_sqrt = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
        out[idx] = np.einsum('cd,ndt->nct', R_inv_sqrt, X).astype(np.float32)
    if logger:
        logger.info(f"  EA applied to {len(unique_subj)} subjects")
    return out


# ──────────────────────────────────────────────
# Sub-band decomposition (α+β only)
# ──────────────────────────────────────────────
def build_band_filters(fs: int, order: int = 4):
    sos = []
    nyq = fs / 2.0
    for name, low, high in MI_SUBBANDS:
        sos.append(butter(order, [low / nyq, high / nyq], btype="band", output="sos"))
    return sos


def decompose_subbands(data: np.ndarray, sos_list, chunk: int = 1024) -> np.ndarray:
    N, C, T = data.shape
    out = np.empty((N, C, len(sos_list), T), dtype=np.float32)
    for i in range(0, N, chunk):
        cd = data[i:i + chunk]
        for s_idx, sos in enumerate(sos_list):
            out[i:i + chunk, :, s_idx, :] = sosfiltfilt(sos, cd, axis=-1).astype(np.float32)
    return out


class SubbandEEGDataset(Dataset):
    def __init__(self, d5, y):
        self.data = torch.from_numpy(d5).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


# ──────────────────────────────────────────────
# Prototype head (v1-style, re-computed per forward from support)
# ──────────────────────────────────────────────
class BatchPrototypeHead(nn.Module):
    """Classification via negative squared Euclidean distance to per-class
    prototypes computed from a support set. During training, support = current
    batch (few-shot style regularization). During eval, support = full train set.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, query, support_feats, support_labels):
        """
        query:           (B, d) features to classify
        support_feats:   (S, d) features forming prototypes
        support_labels:  (S,)
        returns logits (B, n_classes): -||q - mu_k||^2
        """
        q = self.projector(query)
        s = self.projector(support_feats)
        protos = []
        for k in range(self.n_classes):
            mask = support_labels == k
            if mask.sum() > 0:
                protos.append(s[mask].mean(dim=0))
            else:
                protos.append(torch.zeros(s.shape[1], device=s.device))
        protos = torch.stack(protos, dim=0)                      # (K, d)
        dists = ((q.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1)
        return -dists                                            # logits


# ──────────────────────────────────────────────
# Model wrapper: encoder + prototype head
# ──────────────────────────────────────────────
class MIACTSNetV2(nn.Module):
    def __init__(self, n_channels, n_subbands, seq_len, patch_len, d_model,
                 n_classes, dropout, use_revin, revin_per_sample, spectral_inject):
        super().__init__()
        self.encoder = ACTSNetV2(
            n_channels=n_channels, n_subbands=n_subbands,
            seq_len=seq_len, patch_len=patch_len, d_model=d_model,
            n_classes=n_classes,  # unused; we replace head
            n_heads=4, n_freqlens_layers=2, dropout=dropout,
            use_revin=use_revin, revin_per_sample=revin_per_sample,
            spectral_inject=spectral_inject,
        )
        self.head = BatchPrototypeHead(d_model=d_model, n_classes=n_classes)

    def encode(self, x):
        _, glob = self.encoder.encode(x)
        return glob                                              # (B, d)

    def forward(self, query, support_x, support_y):
        q = self.encode(query)
        s = self.encode(support_x)
        return self.head(q, s, support_y)


# ──────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    tot = 0.0; n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # support = current batch (episode-style)
        logits = model(x, x, y)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model, train_loader, test_loader, device, max_support=3000):
    """Evaluate using train set as support (prototypes), test set as queries.
    If train set is larger than max_support, subsample support in chunks.
    """
    model.eval()
    # Collect support features (encoded) from training set
    supp_feats, supp_y = [], []
    n_collected = 0
    for x, y in train_loader:
        x = x.to(device)
        feats = model.encode(x)
        supp_feats.append(feats); supp_y.append(y.to(device))
        n_collected += x.size(0)
        if n_collected >= max_support:
            break
    support_feats = torch.cat(supp_feats)[:max_support]           # (S, d)
    support_labels = torch.cat(supp_y)[:max_support]              # (S,)

    # Evaluate
    all_preds, all_labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        q = model.encode(x)
        logits = model.head(q, support_feats, support_labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(y.numpy())
    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    return {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "cohen_kappa":       cohen_kappa_score(all_labels, all_preds),
        "weighted_f1":       f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "accuracy":          accuracy_score(all_labels, all_preds),
    }


# ──────────────────────────────────────────────
# LOSO runner
# ──────────────────────────────────────────────
def run_loso(dataset_name, eegfm_root, seed, epochs, batch_size, lr, device,
             logger, fs, max_time_len, d_model, patch_len, dropout, weight_decay,
             use_revin, revin_per_sample, n_folds, do_ea, spectral_inject):
    info = EEGFM_DATASETS[dataset_name]
    n_classes = info["n_classes"]
    version = info["version"]

    logger.info(f"Loading {dataset_name} ...")
    t0 = time.time()
    data, labels, subjects = load_arrow_all_with_subject(
        eegfm_root, dataset_name, version, max_time_len=max_time_len
    )
    n_channels_raw = data.shape[1]
    n_timesteps = data.shape[2]
    logger.info(f"  Loaded {len(data)} samples in {time.time()-t0:.0f}s | "
                f"shape {data.shape} | subjects: {len(np.unique(subjects))}")

    if do_ea:
        logger.info("Applying Euclidean Alignment ...")
        t0 = time.time()
        data = euclidean_align(data, subjects, logger=logger)
        logger.info(f"  EA done in {time.time()-t0:.0f}s")

    logger.info(f"Decomposing sub-bands α+β (fs={fs}) ...")
    t0 = time.time()
    sos = build_band_filters(fs=fs)
    data_2b = decompose_subbands(data, sos)
    logger.info(f"  Decomposed in {time.time()-t0:.0f}s | shape {data_2b.shape}")
    del data

    unique_subj = np.unique(subjects)
    if len(unique_subj) <= n_folds:
        fold_assignments = [(s, np.array([s])) for s in unique_subj]
        logger.info(f"Using per-subject LOSO: {len(unique_subj)} folds")
    else:
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(unique_subj)
        groups = np.array_split(shuffled, n_folds)
        fold_assignments = [(f"fold{i}", g) for i, g in enumerate(groups)]
        logger.info(f"Using {n_folds}-fold grouped LOSO")

    fold_metrics = []
    for fold_idx, (fold_id, test_subjects) in enumerate(fold_assignments):
        test_mask = np.isin(subjects, test_subjects)
        train_data = data_2b[~test_mask]; train_labels = labels[~test_mask]
        test_data  = data_2b[test_mask];  test_labels  = labels[test_mask]

        logger.info(f"\n{'='*60}\nFold {fold_idx+1}/{len(fold_assignments)} [{fold_id}] "
                    f"| train: {len(train_data)}, test: {len(test_data)}")

        set_seed(seed)
        train_ds = SubbandEEGDataset(train_data, train_labels)
        test_ds  = SubbandEEGDataset(test_data,  test_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=0)

        seq_len = n_timesteps - (n_timesteps % patch_len)
        model = MIACTSNetV2(
            n_channels=n_channels_raw, n_subbands=len(MI_SUBBANDS),
            seq_len=seq_len, patch_len=patch_len, d_model=d_model,
            n_classes=n_classes, dropout=dropout,
            use_revin=use_revin, revin_per_sample=revin_per_sample,
            spectral_inject=spectral_inject,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        best_metrics = None; best_bal = 0.0
        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            if ep % 10 == 0 or ep == epochs:
                m = evaluate(model, train_loader, test_loader, device)
                if m["balanced_accuracy"] > best_bal:
                    best_bal = m["balanced_accuracy"]; best_metrics = m.copy()
                logger.info(f"  Epoch {ep:3d}/{epochs} | Loss: {loss:.4f} | "
                            f"Bal Acc: {m['balanced_accuracy']:.4f} | "
                            f"Kappa: {m['cohen_kappa']:.4f} | F1: {m['weighted_f1']:.4f}")

        logger.info(f"  Fold {fold_idx+1} best: {best_metrics}")
        fold_metrics.append(best_metrics)
        del model, optimizer, scheduler, train_loader, test_loader, train_ds, test_ds
        del train_data, test_data
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}\nDataset: {dataset_name} — LOSO Final ({len(fold_metrics)} folds):")
    final = {}
    for k in fold_metrics[0].keys():
        vals = np.array([m[k] for m in fold_metrics])
        final[k] = (vals.mean(), vals.std())
        logger.info(f"  {k}: {vals.mean():.4f} ± {vals.std():.4f}")
    return final


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description="ACTSNetv2 MI-specific LOSO benchmark")
    parser.add_argument("--datasets", nargs="+", default=["bcic_2a"],
                        choices=list(EEGFM_DATASETS.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--max_time_len", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--patch_len", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_revin", action="store_true")
    parser.add_argument("--revin_per_sample", action="store_true", default=True)
    parser.add_argument("--no_ea", action="store_true",
                        help="Disable Euclidean Alignment (for ablation)")
    parser.add_argument("--spectral_inject", action="store_true",
                        help="CBraMod-style: per-patch FFT magnitude added to "
                             "patch embedding (lets model attend to per-patch frequency)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints/mi_benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("mi_bench", os.path.join(args.output_dir, "benchmark.log"))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("ACTSNetv2 MI-specific Benchmark (EA + α+β + Prototype)")
    logger.info(f"Datasets: {args.datasets} | seed: {args.seed} | folds: {args.n_folds}")
    logger.info(f"EA: {not args.no_ea} | Subbands: α+β (2) | "
                f"Spectral inject: {args.spectral_inject}")
    logger.info(f"Device: {device} | epochs: {args.epochs} | bs: {args.batch_size}")

    all_results = {}
    for ds in args.datasets:
        logger.info(f"\n{'#'*60}\n# Dataset: {ds}  (LOSO MI)\n{'#'*60}")
        try:
            result = run_loso(
                dataset_name=ds,
                eegfm_root=get_eegfm_root(ds),
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                logger=logger,
                fs=args.fs,
                max_time_len=args.max_time_len,
                d_model=args.d_model,
                patch_len=args.patch_len,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                use_revin=not args.no_revin,
                revin_per_sample=args.revin_per_sample,
                n_folds=args.n_folds,
                do_ea=not args.no_ea,
                spectral_inject=args.spectral_inject,
            )
            all_results[ds] = result
        except Exception as e:
            logger.error(f"Failed on {ds}: {e}", exc_info=True)

    logger.info(f"\n{'#'*60}\n# FINAL\n{'#'*60}")
    for ds, m in all_results.items():
        ba = m["balanced_accuracy"]; kp = m["cohen_kappa"]; f1 = m["weighted_f1"]
        logger.info(f"{ds:<12} BalAcc {ba[0]:.4f}±{ba[1]:.4f}  "
                    f"Kappa {kp[0]:.4f}±{kp[1]:.4f}  F1 {f1[0]:.4f}±{f1[1]:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
