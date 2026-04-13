import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import time
from pathlib import Path

from model import ACTSNetV2
from dataset import EEGDataset, EEGAugmentation
from losses import ACTSNetV2Loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_actsnet_v2(config: dict):
    """
    Full training pipeline for ACTSNet v2.

    Config keys:
        data_path: path to preprocessed .npy data
        labels_path: path to labels .npy
        batch_size, lr, weight_decay, epochs, d_model, patch_len,
        n_freqlens_layers, lambda_supcon, val_ratio, seed, device,
        save_path, task
    """
    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # --- Data ---
    data = np.load(config['data_path'])      # (N, 7, 5, T)
    labels = np.load(config['labels_path'])   # (N,)
    print(f"Data shape: {data.shape}, Labels: {labels.shape}, "
          f"Class dist: {np.bincount(labels.astype(int))}")

    # Split indices first, then create separate datasets for train (with aug) and val (without)
    n_total = len(data)
    val_size = int(n_total * config.get('val_ratio', 0.3))
    train_size = n_total - val_size
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(config.get('seed', 42)))
    train_idx = indices[:train_size].numpy()
    val_idx = indices[train_size:].numpy()

    train_dataset = EEGDataset(data[train_idx], labels[train_idx], transform=EEGAugmentation())
    val_dataset = EEGDataset(data[val_idx], labels[val_idx], transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=config.get('batch_size', 32),
        shuffle=True, drop_last=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get('batch_size', 32),
        shuffle=False, num_workers=4,
    )

    # --- Model ---
    seq_len = data.shape[-1]
    model = ACTSNetV2(
        n_channels=7,
        n_subbands=5,
        seq_len=seq_len,
        patch_len=config.get('patch_len', 32),
        d_model=config.get('d_model', 128),
        n_classes=2,
        n_heads=4,
        n_freqlens_layers=config.get('n_freqlens_layers', 2),
        dropout=config.get('dropout', 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # --- Loss ---
    criterion = ACTSNetV2Loss(
        lambda_supcon=config.get('lambda_supcon', 0.5),
        lambda_proto=0.1,
    )

    # --- Training Loop ---
    save_dir = Path(config.get('save_path', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    epochs = config.get('epochs', 200)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # == Train ==
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits, embeddings = model(batch_x, labels=batch_y)
            loss, loss_dict = criterion(logits, embeddings, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # == Validate ==
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, embeddings = model(batch_x)
                loss, _ = criterion(logits, embeddings, batch_y)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )

        # Save best
        if f1 > best_val_f1:
            best_val_f1 = f1
            task = config.get('task', 'rtms')
            ckpt_path = save_dir / f'actsnet_v2_{task}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': f1,
                'val_auc': auc,
                'val_acc': acc,
                'config': config,
            }, ckpt_path)
            print(f"  -> Saved best model (F1={f1:.4f}, AUC={auc:.4f})")

    print(f"Training complete. Best val F1: {best_val_f1:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ACTSNet v2")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed .npy data (N, 7, 5, T)")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to labels .npy (N,)")
    parser.add_argument("--task", type=str, default="rtms", choices=["rtms", "itbs"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--patch_len", type=int, default=32)
    parser.add_argument("--n_freqlens_layers", type=int, default=2)
    parser.add_argument("--lambda_supcon", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()

    config = vars(args)
    train_actsnet_v2(config)


if __name__ == '__main__':
    main()
