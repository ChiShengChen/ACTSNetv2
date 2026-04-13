import torch
import torch.nn as nn
import torch.nn.functional as F


class SubBandFusion(nn.Module):
    """
    Learnable sub-band mixing module.

    Original ACTSNet simply concatenated the 5 sub-band features.
    This module learns optimal mixing weights across delta/theta/alpha/beta/gamma bands,
    allowing the model to discover which frequency bands matter most
    for MDD classification.

    Args:
        n_subbands: number of frequency sub-bands (default: 5)
        d_model: feature dimension
        fusion_type: 'attention' | 'gated' | 'weighted_sum'
    """

    def __init__(self, n_subbands=5, d_model=128, fusion_type='attention'):
        super().__init__()
        self.n_subbands = n_subbands
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

        if fusion_type == 'attention':
            self.band_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.band_key = nn.Linear(d_model, d_model)
            self.band_value = nn.Linear(d_model, d_model)
            self.scale = d_model ** -0.5

        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(n_subbands * d_model, n_subbands),
                nn.Softmax(dim=-1),
            )

        elif fusion_type == 'weighted_sum':
            self.band_weights = nn.Parameter(torch.ones(n_subbands) / n_subbands)

        self._band_attention = None

    def forward(self, x):
        """
        x: (B, C, S, N, d) — batch, channels, subbands(5), patches, d_model
        returns: (B, C, N, d) — fused across subbands
        """
        B, C, S, N, d = x.shape

        if self.fusion_type == 'attention':
            # Reshape for attention: (B*C*N, S, d)
            x_reshaped = x.permute(0, 1, 3, 2, 4).reshape(B * C * N, S, d)
            Q = self.band_query.expand(B * C * N, -1, -1)
            K = self.band_key(x_reshaped)
            V = self.band_value(x_reshaped)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)  # (B*C*N, 1, S)
            out = torch.matmul(attn, V).squeeze(1)  # (B*C*N, d)
            out = out.reshape(B, C, N, d)
            # Store attention weights for interpretability
            self._band_attention = attn.reshape(B, C, N, S).mean(dim=[0, 2])  # (C, S)

        elif self.fusion_type == 'gated':
            x_cat = x.permute(0, 1, 3, 2, 4).reshape(B * C * N, S * d)
            gates = self.gate(x_cat)  # (B*C*N, S)
            x_reshaped = x.permute(0, 1, 3, 2, 4).reshape(B * C * N, S, d)
            out = (x_reshaped * gates.unsqueeze(-1)).sum(dim=1)
            out = out.reshape(B, C, N, d)

        elif self.fusion_type == 'weighted_sum':
            weights = F.softmax(self.band_weights, dim=0)
            out = (x * weights[None, None, :, None, None]).sum(dim=2)

        return out

    def get_band_importance(self):
        """Return sub-band importance scores for interpretability."""
        if self.fusion_type == 'attention' and self._band_attention is not None:
            return dict(zip(self.band_names, self._band_attention.mean(0).cpu().tolist()))
        elif self.fusion_type == 'weighted_sum':
            weights = F.softmax(self.band_weights, dim=0)
            return dict(zip(self.band_names, weights.detach().cpu().tolist()))
        return None
