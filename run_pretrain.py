"""ACTSNetv2 self-supervised pretraining on EEG-FM-Bench data.

Design:
    Input: (B, C, T) raw -> 5-band bandpass -> (B, C, S, T) on-the-fly in collate
    Encoder: ACTSNetV2 up through global pool (pre-head features)
    Losses:
        L_contrast: NT-Xent between two augmented views' global features
        L_recon:    MSE from per-patch features back to clean (B, C, S, T)
    L = L_contrast + lambda_recon * L_recon

Augmentations:
    View1: time shift (±10%) + Gaussian noise (sigma=0.1 * per-sample-std)
    View2: channel dropout (20%) + sub-band dropout (20%) + amplitude scale (0.8-1.2x)

Usage:
    python run_pretrain.py [--epochs 50] [--batch_size 64] [--max_channels 64]
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from model import ACTSNetV2


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


def get_eegfm_root(dataset_name: str):
    for p in EEGFM_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(p, dataset_name)):
            return p
    raise FileNotFoundError(f"Data root not found for {dataset_name}")


def load_arrow_data(root, dataset, version, split, max_samples=None, max_time_len=1024,
                    subfolder=None):
    """Load EEG-FM-Bench arrow shards. subfolder ∈ {None, 'finetune', 'pretrain'}.
    If None, tries finetune first and falls back to pretrain."""
    if subfolder is not None:
        base_dir = os.path.join(root, dataset, subfolder, version)
    else:
        base_dir = os.path.join(root, dataset, "finetune", version)
        if not os.path.isdir(base_dir):
            base_dir = os.path.join(root, dataset, "pretrain", version)
    prefix = f"{dataset}-{split}-"
    shards = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(".arrow")
    ])
    if not shards:
        raise FileNotFoundError(f"No shards for {dataset}/{split}")

    # Pass 1: count + shape probe (no data loaded)
    total = 0
    probe_shape = None
    for s in shards:
        table = ipc.open_stream(s).read_all()
        if probe_shape is None and len(table) > 0:
            d0 = np.asarray(table.column("data")[0].as_py(), dtype=np.float32)
            probe_shape = d0.shape
        total += len(table)
    if probe_shape is None:
        raise RuntimeError(f"No samples found in {base_dir}")
    C, T_raw = probe_shape
    T = min(T_raw, max_time_len) if max_time_len else T_raw

    # Subsample decision
    if max_samples and total > max_samples:
        rng = np.random.RandomState(42)
        keep_arr = np.sort(rng.choice(total, max_samples, replace=False))
        keep_set = set(int(i) for i in keep_arr)
        keep_pos = {g: p for p, g in enumerate(keep_arr.tolist())}
        n_keep = max_samples
    else:
        keep_set = None
        keep_pos = None
        n_keep = total

    # Pass 2: pre-allocate exact-size output + fill only kept indices
    out = np.empty((n_keep, C, T), dtype=np.float32)
    gi = 0
    for s in shards:
        table = ipc.open_stream(s).read_all()
        for i in range(len(table)):
            if keep_set is None or gi in keep_set:
                d = np.asarray(table.column("data")[i].as_py(), dtype=np.float32)
                if max_time_len and d.shape[-1] > max_time_len:
                    d = d[:, :max_time_len]
                pos = gi if keep_pos is None else keep_pos[gi]
                out[pos] = d
            gi += 1
    return out


def build_band_filters(fs: int, order: int = 4):
    sos_list = []
    nyq = fs / 2.0
    for name, low, high in SUBBANDS:
        high_eff = min(high, nyq - 1e-3)
        sos = butter(order, [low / nyq, high_eff / nyq], btype="band", output="sos")
        sos_list.append(sos)
    return sos_list


class PretrainEEGDataset(Dataset):
    """Yields (C_padded, T) tensors. Stored raw at actual channel count;
    padded to max_channels in __getitem__ to avoid the 2× memory spike from
    a big np.concatenate at dataset construction time."""

    def __init__(self, data: np.ndarray, max_channels: int, max_time_len: int):
        if data.shape[-1] > max_time_len:
            data = data[:, :, :max_time_len]
        if data.shape[1] > max_channels:
            data = data[:, :max_channels, :]
        self.max_channels = max_channels
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]                      # (C_actual, T)
        C, T = x.shape
        if C < self.max_channels:
            pad = torch.zeros(self.max_channels - C, T, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        return x                                # (max_channels, T)


class BandpassCollate:
    """Apply 5-band bandpass on batched (B, C, T) numpy, returning (B, C, 5, T)."""
    def __init__(self, sos_list):
        self.sos_list = sos_list

    def __call__(self, batch):
        x = torch.stack(batch, dim=0)              # (B, C, T)
        x_np = x.numpy()
        B, C, T = x_np.shape
        out = np.empty((B, C, len(self.sos_list), T), dtype=np.float32)
        for s_idx, sos in enumerate(self.sos_list):
            out[:, :, s_idx, :] = sosfiltfilt(sos, x_np, axis=-1).astype(np.float32)
        return torch.from_numpy(out)               # (B, C, 5, T)


# ──────────────────────────────────────────────
# Pretrainer module
# ──────────────────────────────────────────────
class V2Pretrainer(nn.Module):
    def __init__(self, encoder: ACTSNetV2, d_model: int, n_subbands: int,
                 patch_len: int, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.n_subbands = n_subbands
        self.patch_len = patch_len

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        # Per-patch decoder: d_model -> S * patch_len
        self.decoder = nn.Linear(d_model, n_subbands * patch_len)

    def encode_pair(self, view):
        per_patch, glob = self.encoder.encode(view)           # (B, C*N, d), (B, d)
        return per_patch, glob

    def project(self, glob):
        return F.normalize(self.projector(glob), dim=-1)      # (B, proj_dim)

    def reconstruct(self, per_patch, B, C, S, T):
        """per_patch: (B, C*N, d) -> reconstructed (B, C, S, T)"""
        out = self.decoder(per_patch)                         # (B, C*N, S*patch_len)
        N = T // self.patch_len
        assert per_patch.shape[1] == C * N, (per_patch.shape, C, N)
        out = out.reshape(B, C, N, S, self.patch_len)
        out = out.permute(0, 1, 3, 2, 4).contiguous()         # (B, C, S, N, patch_len)
        out = out.reshape(B, C, S, N * self.patch_len)        # (B, C, S, T)
        return out


# ──────────────────────────────────────────────
# Augmentations (on (B, C, S, T))
# ──────────────────────────────────────────────
def _time_shift(x: torch.Tensor, max_frac: float = 0.1) -> torch.Tensor:
    B, C, S, T = x.shape
    shifts = torch.randint(-int(T * max_frac), int(T * max_frac) + 1, (B,), device=x.device)
    out = x.clone()
    for i in range(B):
        out[i] = torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1)
    return out


def aug_view1(x: torch.Tensor, noise_sigma: float = 0.1) -> torch.Tensor:
    """View1: time shift + Gaussian noise (sigma relative to per-sample std)."""
    v = _time_shift(x, max_frac=0.1)
    per_sample_std = v.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
    v = v + torch.randn_like(v) * (noise_sigma * per_sample_std)
    return v


def aug_view2(x: torch.Tensor, ch_drop: float = 0.2, sb_drop: float = 0.2) -> torch.Tensor:
    """View2: channel dropout + sub-band dropout + amplitude scale."""
    B, C, S, T = x.shape
    v = x.clone()
    ch_mask = (torch.rand(B, C, 1, 1, device=x.device) > ch_drop).float()
    sb_mask = (torch.rand(B, 1, S, 1, device=x.device) > sb_drop).float()
    v = v * ch_mask * sb_mask
    scale = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device)
    v = v * scale
    return v


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature: float = 0.1):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def pretrain_loop(pretrainer, dataloader, epochs, lr, device, logger, ckpt_dir,
                  lambda_recon: float, temperature: float):
    pretrainer = pretrainer.to(device)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=lr,
                                  weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = epochs * len(dataloader)
    warmup_steps = min(2 * len(dataloader), total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = sum(p.numel() for p in pretrainer.parameters() if p.requires_grad)
    logger.info(f"Starting pretrain: {epochs} epochs, {total_steps} total steps")
    logger.info(f"Pretrainer params: {n_params:,}")

    for epoch in range(1, epochs + 1):
        pretrainer.train()
        epoch_loss = epoch_cl = epoch_rl = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device, non_blocking=True)            # (B, C, S, T) clean
            B, C, S, T = x.shape

            v1 = aug_view1(x)
            v2 = aug_view2(x)

            p1, g1 = pretrainer.encode_pair(v1)
            p2, g2 = pretrainer.encode_pair(v2)

            z1 = pretrainer.project(g1)
            z2 = pretrainer.project(g2)
            loss_cl = nt_xent_loss(z1, z2, temperature=temperature)

            # Reconstruct clean signal from each view's per-patch features
            recon1 = pretrainer.reconstruct(p1, B, C, S, T)
            recon2 = pretrainer.reconstruct(p2, B, C, S, T)
            loss_rl = 0.5 * (F.mse_loss(recon1, x) + F.mse_loss(recon2, x))

            loss = loss_cl + lambda_recon * loss_rl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_cl += loss_cl.item()
            epoch_rl += loss_rl.item()
            n_batches += 1

            if (batch_idx + 1) % 500 == 0:
                logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} (CL: {loss_cl.item():.4f}, "
                    f"RL: {loss_rl.item():.4f})"
                )

        lr_now = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/n_batches:.4f} "
            f"(CL: {epoch_cl/n_batches:.4f}, RL: {epoch_rl/n_batches:.4f}) | "
            f"LR: {lr_now:.6f} | {time.time()-t0:.0f}s"
        )

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"pretrain_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'full_state_dict': pretrainer.state_dict(),
                'encoder_state_dict': pretrainer.encoder.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "pretrain_final.pt")
    torch.save({
        'encoder_state_dict': pretrainer.encoder.state_dict(),
    }, final_path)
    logger.info(f"Pretrain complete! Final encoder: {final_path}")


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description="ACTSNetv2 Self-Supervised Pretraining")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_channels", type=int, default=64)
    parser.add_argument("--max_time_len", type=int, default=1024)
    parser.add_argument("--max_samples_per_dataset", type=int, default=20000,
                        help="Cap per original finetune-train dataset")
    parser.add_argument("--tuab_pretrain_samples", type=int, default=200000,
                        help="Cap samples from tuab/pretrain/3.0.1 (~1M available). "
                             "Total pool = 5×max_samples_per_dataset + this. Raw memory "
                             "= N × 64 × 1024 × 4 bytes; 200k ≈ 52 GB.")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--patch_len", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--revin_per_sample", action="store_true", default=True,
                        help="Use per-sample RevIN (recommended for EEG-FM pretrain)")
    parser.add_argument("--no_revin", action="store_true")
    parser.add_argument("--no_spectral", action="store_true",
                        help="Disable CBraMod-style spectral injection (default: on)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("actsnetv2_pretrain", os.path.join(args.output_dir, "pretrain.log"))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Each entry: (dataset, version, split, max_samples, subfolder)
    pretrain_info = [
        # Large TUAB pretrain split (~1M samples available; capped by --tuab_pretrain_samples)
        ("tuab",        "3.0.1", "train", args.tuab_pretrain_samples, "pretrain"),
        # Original 5-dataset finetune-train pool
        ("tuab",        "3.0.1", "train", args.max_samples_per_dataset, "finetune"),
        ("tuev",        "2.0.0", "train", args.max_samples_per_dataset, "finetune"),
        ("bcic_2a",     "1.0.0", "train", None,                         "finetune"),
        ("seed_iv",     "1.0.0", "train", args.max_samples_per_dataset, "finetune"),
        ("siena_scalp", "1.0.0", "train", args.max_samples_per_dataset, "finetune"),
    ]

    logger.info(f"ACTSNetv2 Pretrain | device: {device}")
    logger.info(f"max_channels={args.max_channels}, T={args.max_time_len}, "
                f"fs={args.fs}, patch_len={args.patch_len}, d_model={args.d_model}")
    logger.info(f"lambda_recon={args.lambda_recon}, temperature={args.temperature}")
    logger.info(f"revin_per_sample={args.revin_per_sample}, no_revin={args.no_revin}, "
                f"dropout={args.dropout}")

    datasets = []
    total = 0
    for ds_name, version, split, max_s, subfolder in pretrain_info:
        try:
            root = get_eegfm_root(ds_name)
            logger.info(f"Loading {ds_name}/{subfolder}/{split} from {root} ...")
            data = load_arrow_data(root, ds_name, version, split,
                                   max_samples=max_s, max_time_len=args.max_time_len,
                                   subfolder=subfolder)
            ds = PretrainEEGDataset(data, max_channels=args.max_channels,
                                    max_time_len=args.max_time_len)
            datasets.append(ds)
            total += len(ds)
            logger.info(f"  {ds_name}/{subfolder}: {len(ds)} samples, raw shape={data.shape}")
        except Exception as e:
            logger.warning(f"  Skipping {ds_name}: {e}")

    if not datasets:
        logger.error("No datasets loaded")
        return

    sos_list = build_band_filters(fs=args.fs)
    combined = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=BandpassCollate(sos_list),
        pin_memory=True,
    )
    logger.info(f"Total: {total} samples, {len(dataloader)} batches")

    seq_len = args.max_time_len - (args.max_time_len % args.patch_len)
    encoder = ACTSNetV2(
        n_channels=args.max_channels,
        n_subbands=len(SUBBANDS),
        seq_len=seq_len,
        patch_len=args.patch_len,
        d_model=args.d_model,
        n_classes=2,  # placeholder, proto_head unused during pretrain
        n_heads=4,
        n_freqlens_layers=2,
        dropout=args.dropout,
        use_revin=not args.no_revin,
        revin_per_sample=args.revin_per_sample,
        spectral_inject=not args.no_spectral,
    )
    pretrainer = V2Pretrainer(
        encoder=encoder, d_model=args.d_model, n_subbands=len(SUBBANDS),
        patch_len=args.patch_len, proj_dim=args.proj_dim,
    )

    pretrain_loop(
        pretrainer, dataloader, args.epochs, args.lr, device, logger,
        args.output_dir, lambda_recon=args.lambda_recon, temperature=args.temperature,
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()
