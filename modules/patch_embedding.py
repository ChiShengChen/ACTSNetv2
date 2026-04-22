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

    def __init__(self, patch_len=16, stride=16, d_model=128, dropout=0.1,
                 max_n_patches=512, use_spectral=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.use_spectral = use_spectral
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # Pre-allocate positional encoding as a proper registered parameter
        self.pos_encoding = nn.Parameter(
            torch.randn(max_n_patches, d_model) * 0.02
        )
        # CBraMod-style spectral injection: FFT magnitude per patch
        # is projected to d_model and added to the time-domain embedding.
        # Lets the model attend to per-patch frequency content automatically
        # rather than relying on fixed bandpass filters.
        if use_spectral:
            n_freqs = patch_len // 2 + 1
            self.spectral_projection = nn.Linear(n_freqs, d_model)

    def forward(self, x):
        """
        x: (B, C, S, T) — batch, channels(7), subbands(5), time
        returns: (B, C, S, N_patches, d_model)
        """
        B, C, S, T = x.shape
        # Reshape to (B*C*S, T) for patch extraction
        x = x.reshape(B * C * S, T)
        # Unfold into patches: (B*C*S, n_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        n_patches = patches.shape[1]
        # Time-domain projection
        emb = self.projection(patches)  # (B*C*S, n_patches, d_model)
        # CBraMod-style spectral injection
        if self.use_spectral:
            spectral = torch.fft.rfft(patches, dim=-1, norm='forward')
            spectral_mag = torch.abs(spectral)
            emb = emb + self.spectral_projection(spectral_mag)
        # Add positional encoding (slice to actual n_patches)
        emb = self.dropout(emb + self.pos_encoding[:n_patches])
        # Reshape back
        return emb.reshape(B, C, S, n_patches, -1)
