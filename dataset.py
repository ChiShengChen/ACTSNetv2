import torch
import numpy as np
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    Args:
        data: np.ndarray of shape (N, 7, 5, T) — segments x channels x subbands x time
        labels: np.ndarray of shape (N,) — 0=non-responder, 1=responder
        transform: optional augmentation
    """

    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]  # (7, 5, T)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class EEGAugmentation:
    """Data augmentation for EEG time series."""

    def __init__(self, noise_std=0.01, time_mask_ratio=0.1):
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise injection
        if torch.rand(1) < 0.5:
            x = x + torch.randn_like(x) * self.noise_std

        # Random time masking
        if torch.rand(1) < 0.5:
            T = x.shape[-1]
            mask_len = int(T * self.time_mask_ratio)
            if mask_len > 0 and T > mask_len:
                start = torch.randint(0, T - mask_len, (1,)).item()
                x[..., start:start + mask_len] = 0.0

        return x
