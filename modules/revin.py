import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Normalizes input by removing instance-level statistics,
    then re-applies them after the model forward pass.
    Crucial for EEG where baseline drift varies across subjects.

    Reference: Kim et al., "Reversible Instance Normalization
    for Accurate Time-Series Forecasting against Distribution Shift"
    (NeurIPS 2021)

    Args:
        num_features: number of feature channels
        eps: epsilon for numerical stability
        affine: whether to learn affine parameters
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
        self._mean = None
        self._std = None

    def normalize(self, x):
        """
        x: (B, C, T) or (B, C, ...)
        Stores mean/std for later denormalization.
        """
        dim = tuple(range(2, x.ndim))  # normalize over time dims
        self._mean = x.mean(dim=dim, keepdim=True).detach()
        self._std = (x.var(dim=dim, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x):
        """Re-apply stored statistics."""
        if self.affine:
            x = (x - self.beta) / self.gamma
        x = x * self._std + self._mean
        return x
