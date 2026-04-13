import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Learnable inter-channel (electrode) attention.

    Replaces TapNet's random dimension permutation with a proper
    attention mechanism over EEG channels. Learns spatial relationships
    between electrodes (e.g., FP1-FP2 interhemispheric asymmetry,
    frontal coherence patterns).

    Inspired by: iTransformer (Liu et al., ICLR 2024)

    Args:
        n_channels: number of EEG channels (default: 7)
        d_model: feature dimension
        n_heads: number of attention heads
        dropout: dropout rate
    """

    def __init__(self, n_channels=7, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.electrode_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8']

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable electrode position embedding
        self.electrode_embedding = nn.Parameter(
            torch.randn(n_channels, d_model) * 0.02
        )

        self._channel_attn = None

    def forward(self, x):
        """
        x: (B, C, N, d) — batch, channels(7), patches, d_model
        returns: (B, C, N, d) — channel-attended features
        """
        B, C, N, d = x.shape
        residual = x

        # Pool over patches to get per-channel representation
        x_pooled = x.mean(dim=2)  # (B, C, d)

        # Add electrode position embedding
        x_pooled = x_pooled + self.electrode_embedding.unsqueeze(0)

        # Multi-head attention over channels
        Q = self.q_proj(x_pooled).view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x_pooled).view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x_pooled).view(B, C, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        self._channel_attn = attn.detach()

        out = torch.matmul(attn, V)  # (B, n_heads, C, d_head)
        out = out.transpose(1, 2).reshape(B, C, d)  # (B, C, d)

        out = self.out_proj(out)
        out = self.dropout(out)

        # Broadcast back to (B, C, N, d)
        out = out.unsqueeze(2) + residual
        out = self.layer_norm(out)

        return out

    def get_channel_connectivity(self):
        """Return electrode-to-electrode attention matrix for visualization."""
        if self._channel_attn is not None:
            # Average over heads and batch: (C, C)
            connectivity = self._channel_attn.mean(dim=[0, 1]).cpu().numpy()
            return connectivity, self.electrode_names
        return None, self.electrode_names
