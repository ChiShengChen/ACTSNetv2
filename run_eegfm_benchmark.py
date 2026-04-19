"""ACTSNetv2 Benchmark on EEG-FM-Bench datasets.

Variable-channel, sub-band-decomposed adaptation: loads (N, C, T) from arrow
shards, applies 5-band bandpass (delta/theta/alpha/beta/gamma) to produce
(N, C, 5, T), and trains ACTSNetv2 with its native CE + SupCon loss.

Usage:
    python run_eegfm_benchmark.py [--datasets bcic_2a seed_iv tuab tuev]
                                   [--seeds 42 123 456] [--fs 256]
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
from losses import ACTSNetV2Loss


def load_pretrained_encoder(model: nn.Module, pretrained_path: str, logger: logging.Logger):
    """Load encoder_state_dict from a V2Pretrainer checkpoint into ACTSNetV2."""
    ckpt = torch.load(pretrained_path, map_location='cpu')
    encoder_state = ckpt.get('encoder_state_dict', {})
    if not encoder_state:
        logger.warning(f"No encoder_state_dict in {pretrained_path}, skipping")
        return model
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in encoder_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded} pretrained params from {pretrained_path} "
                f"({skipped} skipped on shape mismatch)")
    return model


class LinearHead(nn.Module):
    """Drop-in replacement for HyperbolicPrototypicalHead: plain Linear + CE.

    Returns (logits, embeddings) where embeddings are the pre-head features,
    matching HyperbolicPrototypicalHead's signature so model.forward() works
    unchanged.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, labels=None):
        return self.fc(x), x

# ──────────────────────────────────────────────
# Dataset info (matches ACTSNet v1)
# ──────────────────────────────────────────────
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

SUBBANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def get_eegfm_root(dataset_name: str | None = None):
    for p in EEGFM_ROOT_CANDIDATES:
        if not os.path.isdir(p):
            continue
        if dataset_name and os.path.isdir(os.path.join(p, dataset_name)):
            return p
        if dataset_name is None:
            return p
    raise FileNotFoundError(f"EEG-FM-Bench data root not found for {dataset_name}")


# ──────────────────────────────────────────────
# Arrow loading
# ──────────────────────────────────────────────
def load_arrow_split(
    root: str, dataset: str, version: str, split: str,
    max_samples: int | None = None, max_time_len: int | None = None,
):
    base_dir = os.path.join(root, dataset, "finetune", version)
    prefix = f"{dataset}-{split}-"
    shards = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(".arrow")
    ])
    if not shards:
        raise FileNotFoundError(f"No shards for {dataset}/{split} in {base_dir}")

    total_count = 0
    for shard_path in shards:
        total_count += len(ipc.open_stream(shard_path).read_all())

    if max_samples is not None and total_count > max_samples:
        rng = np.random.RandomState(42)
        keep_indices = set(rng.choice(total_count, max_samples, replace=False).tolist())
    else:
        keep_indices = None

    all_data, all_labels = [], []
    global_idx = 0
    for shard_path in shards:
        table = ipc.open_stream(shard_path).read_all()
        for i in range(len(table)):
            if keep_indices is None or global_idx in keep_indices:
                d = np.array(table.column("data")[i].as_py(), dtype=np.float32)
                if max_time_len is not None and d.shape[-1] > max_time_len:
                    d = d[:, :max_time_len]
                all_data.append(d)
                all_labels.append(int(table.column("label")[i].as_py()))
            global_idx += 1

    data = np.stack(all_data, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    return data, labels


# ──────────────────────────────────────────────
# Sub-band decomposition
# ──────────────────────────────────────────────
def build_band_filters(fs: int, order: int = 4):
    sos_list = []
    nyq = fs / 2.0
    for name, low, high in SUBBANDS:
        high_eff = min(high, nyq - 1e-3)
        sos = butter(order, [low / nyq, high_eff / nyq], btype="band", output="sos")
        sos_list.append(sos)
    return sos_list


def decompose_subbands(data: np.ndarray, sos_list) -> np.ndarray:
    """(N, C, T) -> (N, C, 5, T) via 5 bandpass filters. Zero-phase filtfilt."""
    N, C, T = data.shape
    out = np.empty((N, C, len(sos_list), T), dtype=np.float32)
    for s_idx, sos in enumerate(sos_list):
        filtered = sosfiltfilt(sos, data, axis=-1)
        out[:, :, s_idx, :] = filtered.astype(np.float32)
    return out


class SubbandEEGDataset(Dataset):
    def __init__(self, data_5band: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data_5band).float()   # (N, C, S, T)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ──────────────────────────────────────────────
# Training / eval
# ──────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, loss_fn, optimizer, device, use_supcon: bool):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, embeddings = model(x, labels=y)
        if use_supcon:
            loss, _ = loss_fn(logits, embeddings, y)
        else:
            loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "cohen_kappa": cohen_kappa_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


# ──────────────────────────────────────────────
# Per-dataset driver
# ──────────────────────────────────────────────
def run_single_dataset(
    dataset_name: str,
    eegfm_root: str,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    logger: logging.Logger,
    fs: int,
    max_time_len: int | None,
    max_train_samples: int | None,
    d_model: int,
    patch_len: int,
    lambda_supcon: float,
    weight_decay: float,
    head: str,
    use_revin: bool,
    revin_per_sample: bool,
    dropout: float,
    pretrained_path: str | None,
    max_channels: int | None,
    freeze_encoder: bool,
):
    info = EEGFM_DATASETS[dataset_name]
    n_classes = info["n_classes"]
    version = info["version"]

    logger.info(f"Loading {dataset_name} ...")
    t0 = time.time()
    train_data, train_labels = load_arrow_split(
        eegfm_root, dataset_name, version, "train",
        max_samples=max_train_samples, max_time_len=max_time_len,
    )
    test_data, test_labels = load_arrow_split(
        eegfm_root, dataset_name, version, "test",
        max_time_len=max_time_len,
    )
    n_channels = train_data.shape[1]
    n_timesteps = train_data.shape[2]
    logger.info(
        f"  Loaded in {time.time()-t0:.0f}s | Train: {len(train_data)}, "
        f"Test: {len(test_data)} | Shape: ({n_channels}, {n_timesteps}) | "
        f"Classes: {n_classes}"
    )

    # If a pretrained checkpoint is provided, pad to max_channels (same as pretrain)
    if pretrained_path is not None and max_channels is not None and n_channels < max_channels:
        pad_ch = max_channels - n_channels
        pad_tr = np.zeros((len(train_data), pad_ch, n_timesteps), dtype=np.float32)
        pad_te = np.zeros((len(test_data),  pad_ch, n_timesteps), dtype=np.float32)
        train_data = np.concatenate([train_data, pad_tr], axis=1)
        test_data  = np.concatenate([test_data,  pad_te], axis=1)
        logger.info(f"  Padded channels {n_channels} -> {max_channels} to match pretrain")
        n_channels = max_channels
    elif pretrained_path is not None and max_channels is not None and n_channels > max_channels:
        train_data = train_data[:, :max_channels, :]
        test_data  = test_data[:,  :max_channels, :]
        logger.info(f"  Truncated channels to {max_channels} to match pretrain")
        n_channels = max_channels

    logger.info(f"Decomposing sub-bands (fs={fs}) ...")
    t0 = time.time()
    sos_list = build_band_filters(fs=fs)
    train_5b = decompose_subbands(train_data, sos_list)
    test_5b = decompose_subbands(test_data, sos_list)
    logger.info(
        f"  Decomposed in {time.time()-t0:.0f}s | Train shape: {train_5b.shape} | "
        f"Test shape: {test_5b.shape}"
    )
    del train_data, test_data

    all_seed_results: dict[str, list[float]] = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}\nSeed: {seed}")
        set_seed(seed)

        train_ds = SubbandEEGDataset(train_5b, train_labels)
        test_ds = SubbandEEGDataset(test_5b, test_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2)

        # Ensure T divisible by patch_len
        seq_len = n_timesteps - (n_timesteps % patch_len)
        model = ACTSNetV2(
            n_channels=n_channels,
            n_subbands=len(SUBBANDS),
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model,
            n_classes=n_classes,
            n_heads=4,
            n_freqlens_layers=2,
            dropout=dropout,
            use_revin=use_revin,
            revin_per_sample=revin_per_sample,
        )
        if head == "linear":
            model.proto_head = LinearHead(d_model=d_model, n_classes=n_classes)
        if pretrained_path is not None:
            model = load_pretrained_encoder(model, pretrained_path, logger)
        model = model.to(device)

        if freeze_encoder:
            for name, p in model.named_parameters():
                if not name.startswith("proto_head."):
                    p.requires_grad = False
            logger.info("Encoder frozen (linear probe)")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable params: {n_params:,} | Head: {head}")

        use_supcon = head == "hyperbolic"
        if use_supcon:
            loss_fn = ACTSNetV2Loss(lambda_supcon=lambda_supcon).to(device)
        else:
            loss_fn = nn.CrossEntropyLoss().to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        best_metrics = None
        best_bal_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                         use_supcon=use_supcon)
            scheduler.step()
            if epoch % 10 == 0 or epoch == epochs:
                metrics = evaluate(model, test_loader, device)
                if metrics["balanced_accuracy"] > best_bal_acc:
                    best_bal_acc = metrics["balanced_accuracy"]
                    best_metrics = metrics.copy()
                logger.info(
                    f"  Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
                    f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                    f"Kappa: {metrics['cohen_kappa']:.4f} | "
                    f"F1: {metrics['weighted_f1']:.4f}"
                )

        logger.info(f"Seed {seed} best: {best_metrics}")
        for k, v in best_metrics.items():
            all_seed_results.setdefault(k, []).append(v)

    logger.info(f"\n{'='*60}\nDataset: {dataset_name} — Final results ({len(seeds)} seeds):")
    final = {}
    for k, vals in all_seed_results.items():
        arr = np.array(vals)
        final[k] = (arr.mean(), arr.std())
        logger.info(f"  {k}: {arr.mean():.4f} ± {arr.std():.4f}")
    return final


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description="ACTSNetv2 EEG-FM-Bench Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["bcic_2a"],
                        choices=list(EEGFM_DATASETS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fs", type=int, default=256,
                        help="Sampling rate for sub-band filters (fs_256 arrow shards)")
    parser.add_argument("--max_time_len", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--patch_len", type=int, default=32)
    parser.add_argument("--lambda_supcon", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "hyperbolic"],
                        help="'linear' = Linear(d_model, n_classes) + CE (default); "
                             "'hyperbolic' = v2 native Hyperbolic Prototypical + CE + SupCon")
    parser.add_argument("--no_revin", action="store_true",
                        help="Disable RevIN entirely")
    parser.add_argument("--revin_per_sample", action="store_true",
                        help="RevIN normalizes per-sample globally (one mean/std per sample) "
                             "instead of per-channel over time. Preserves relative "
                             "channel × sub-band energy structure.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrain checkpoint (encoder_state_dict loaded)")
    parser.add_argument("--pretrained_max_channels", type=int, default=64,
                        help="max_channels the pretrain used; input channels will be padded/truncated")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder, train only the head (linear probe)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/eegfm_benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("actsnetv2_bench", os.path.join(args.output_dir, "benchmark.log"))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("ACTSNetv2 EEG-FM-Bench Benchmark")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, "
                f"WD: {args.weight_decay}")
    logger.info(f"Device: {device}")
    logger.info(f"fs: {args.fs} | max_time_len: {args.max_time_len} | "
                f"max_train_samples: {args.max_train_samples}")
    logger.info(f"d_model: {args.d_model} | patch_len: {args.patch_len} | "
                f"lambda_supcon: {args.lambda_supcon} | head: {args.head}")
    revin_desc = (
        "off" if args.no_revin
        else ("per-sample" if args.revin_per_sample else "per-channel-over-time")
    )
    logger.info(f"RevIN: {revin_desc}")
    logger.info(f"Pretrained: {args.pretrained_path or 'None'} | "
                f"freeze_encoder: {args.freeze_encoder}")

    all_results = {}
    for ds in args.datasets:
        logger.info(f"\n{'#'*60}\n# Dataset: {ds}\n{'#'*60}")
        try:
            eegfm_root = get_eegfm_root(ds)
            result = run_single_dataset(
                dataset_name=ds,
                eegfm_root=eegfm_root,
                seeds=args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                logger=logger,
                fs=args.fs,
                max_time_len=args.max_time_len,
                max_train_samples=args.max_train_samples,
                d_model=args.d_model,
                patch_len=args.patch_len,
                lambda_supcon=args.lambda_supcon,
                weight_decay=args.weight_decay,
                head=args.head,
                use_revin=not args.no_revin,
                revin_per_sample=args.revin_per_sample,
                dropout=args.dropout,
                pretrained_path=args.pretrained_path,
                max_channels=args.pretrained_max_channels,
                freeze_encoder=args.freeze_encoder,
            )
            all_results[ds] = result
        except Exception as e:
            logger.error(f"Failed on {ds}: {e}", exc_info=True)

    logger.info(f"\n{'#'*60}\n# FINAL SUMMARY\n{'#'*60}")
    logger.info(f"{'Dataset':<15} {'Bal Acc':>18} {'Kappa':>18} {'W-F1':>18}")
    logger.info("-" * 72)
    for ds, metrics in all_results.items():
        ba = metrics.get("balanced_accuracy", (0, 0))
        kp = metrics.get("cohen_kappa", (0, 0))
        f1 = metrics.get("weighted_f1", (0, 0))
        logger.info(f"{ds:<15} {ba[0]:.4f}±{ba[1]:.4f}    {kp[0]:.4f}±{kp[1]:.4f}    {f1[0]:.4f}±{f1[1]:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
