import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.patch_embedding import PatchEmbedding
from modules.revin import RevIN
from modules.freqlens_attention import FreqLensAttention
from modules.subband_fusion import SubBandFusion
from modules.channel_attention import ChannelAttention
from modules.spatial_spectral_graph import SpatialSpectralGraph
from modules.hyperbolic_proto import HyperbolicPrototypicalHead
from modules.interpretability import InterpretabilityModule


class ACTSNetV2(nn.Module):
    """
    ACTSNet v2: Frequency-Aware Interpretable EEG Classification
    for TMS Response Prediction.

    Full architecture:
        Input (B, 7, 5, T)
        -> RevIN normalize
        -> PatchEmbedding         -> (B, 7, 5, N_patches, d_model)
        -> SubBandFusion           -> (B, 7, N_patches, d_model)
        -> ChannelAttention        -> (B, 7, N_patches, d_model)
        -> SpatialSpectralGraph    -> (B, 7, N_patches, d_model)
        -> Reshape & FreqLens      -> (B, 7*N_patches, d_model)
        -> Global Average Pool     -> (B, d_model)
        -> HyperbolicPrototypical  -> logits (B, n_classes)

    Args:
        n_channels: number of EEG channels (7)
        n_subbands: number of frequency sub-bands (5)
        seq_len: time series length per segment
        patch_len: patch length for PatchEmbedding
        d_model: model dimension
        n_classes: number of output classes (2)
        n_heads: number of attention heads for ChannelAttention
        n_freqlens_layers: number of FreqLens attention layers
        dropout: dropout rate
    """

    def __init__(
        self,
        n_channels=7,
        n_subbands=5,
        seq_len=2560,
        patch_len=32,
        d_model=128,
        n_classes=2,
        n_heads=4,
        n_freqlens_layers=2,
        dropout=0.1,
        use_revin=True,
        revin_per_sample=False,
        spectral_inject=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_subbands = n_subbands
        self.use_revin = use_revin

        # --- Core Modules ---
        self.revin = RevIN(
            num_features=n_channels * n_subbands,
            per_sample_only=revin_per_sample,
        )
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len, stride=patch_len, d_model=d_model, dropout=dropout,
            use_spectral=spectral_inject,
        )
        self.subband_fusion = SubBandFusion(
            n_subbands=n_subbands, d_model=d_model, fusion_type='attention'
        )
        self.channel_attention = ChannelAttention(
            n_channels=n_channels, d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.spatial_graph = SpatialSpectralGraph(
            n_channels=n_channels, d_model=d_model, n_layers=2, dropout=dropout
        )
        self.freqlens_layers = nn.ModuleList([
            FreqLensAttention(d_model=d_model, dropout=dropout)
            for _ in range(n_freqlens_layers)
        ])

        # --- Classification Head ---
        self.proto_head = HyperbolicPrototypicalHead(
            d_model=d_model, d_embed=64, n_classes=n_classes, curvature=1.0
        )

        # --- Interpretability ---
        self.interpretability = InterpretabilityModule()

        # Storage for intermediate features (interpretability)
        self._last_patch_features = None

    def encode(self, x):
        """Pretrain-friendly forward: returns (per_patch_features, global_feature)
        without the classification head.

        x: (B, C, S, T)
        returns:
            per_patch: (B, C*N_patches, d_model)
            global:    (B, d_model)
        """
        B, C, S, T = x.shape
        if self.use_revin:
            x_flat = x.reshape(B, C * S, T)
            x_flat = self.revin.normalize(x_flat)
            x = x_flat.reshape(B, C, S, T)
        x = self.patch_embed(x)                   # (B, C, S, N, d)
        x = self.subband_fusion(x)                # (B, C, N, d)
        x = self.channel_attention(x)             # (B, C, N, d)
        x = self.spatial_graph(x)                 # (B, C, N, d)
        B, C, N, d = x.shape
        x = x.reshape(B, C * N, d)
        for fl in self.freqlens_layers:
            x = fl(x)                             # (B, C*N, d)
        global_feat = x.mean(dim=1)               # (B, d)
        return x, global_feat

    def forward(self, x, labels=None):
        """
        x: (B, 7, 5, T) — batch, channels, subbands, time
        labels: (B,) — optional, for prototype update during training
        returns: logits (B, n_classes), embeddings (B, d_embed)
        """
        B, C, S, T = x.shape

        # 1. RevIN normalize (optional)
        if self.use_revin:
            x_flat = x.reshape(B, C * S, T)
            x_flat = self.revin.normalize(x_flat)
            x = x_flat.reshape(B, C, S, T)

        # 2. Patch embedding
        x = self.patch_embed(x)  # (B, C, S, N, d_model)

        # 3. Sub-band fusion
        x = self.subband_fusion(x)  # (B, C, N, d_model)

        # 4. Channel attention
        x = self.channel_attention(x)  # (B, C, N, d_model)

        # 5. Spatial-spectral graph
        x = self.spatial_graph(x)  # (B, C, N, d_model)

        # 6. FreqLens attention (over flattened channel-patch dimension)
        B, C, N, d = x.shape
        x = x.reshape(B, C * N, d)
        self._last_patch_features = x.detach()  # for interpretability
        for fl_layer in self.freqlens_layers:
            x = fl_layer(x)  # (B, C*N, d_model)

        # 7. Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        # 8. Hyperbolic prototypical classification
        logits, embeddings = self.proto_head(x, labels=labels)

        return logits, embeddings
