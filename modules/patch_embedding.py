import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Replaces the original 3x Conv1D + IN + PReLU backbone.

    Splits each sub-band time series into non-overlapping patches,
    then projects each patch into a d_model-dimensional embedding.
    Adds learnable positional encoding.

    Reference: PatchTST (Nie et al., ICLR 2023)

    Args:
        patch_len: length of each patch (e.g., 16 or 32 samples)
        stride: patch stride (= patch_len for non-overlapping)
        d_model: embedding dimension (e.g., 128)
        dropout: dropout rate
    """

    def __init__(self, patch_len=16, stride=16, d_model=128, dropout=0.1, max_n_patches=512):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # Pre-allocate positional encoding as a proper registered parameter
        self.pos_encoding = nn.Parameter(
            torch.randn(max_n_patches, d_model) * 0.02
        )

    def forward(self, x):
        """
        x: (B, C, S, T) — batch, channels(7), subbands(5), time
        returns: (B, C, S, N_patches, d_model)
        """
        B, C, S, T = x.shape
        # Reshape to (B*C*S, T) for patch extraction
        x = x.reshape(B * C * S, T)
        # Unfold into patches: (B*C*S, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        n_patches = x.shape[1]
        # Project patches
        x = self.projection(x)  # (B*C*S, n_patches, d_model)
        # Add positional encoding (slice to actual n_patches)
        x = self.dropout(x + self.pos_encoding[:n_patches])
        # Reshape back
        x = x.reshape(B, C, S, n_patches, -1)
        return x
