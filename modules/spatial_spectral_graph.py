import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSpectralGraph(nn.Module):
    """
    Graph neural network module that models EEG electrode topology.

    Constructs a graph where:
    - Nodes = electrodes (7 channels)
    - Edges = spatial adjacency + learned functional connectivity

    Uses GCN layers to propagate information between
    neighboring electrodes, capturing spatial dependencies.

    Args:
        n_channels: number of EEG channels (nodes)
        d_model: feature dimension
        n_layers: number of GCN layers
        dropout: dropout rate
    """

    def __init__(self, n_channels=7, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels

        # Predefined adjacency based on 10-20 system electrode positions
        # FP1(0) FP2(1) F7(2) F3(3) Fz(4) F4(5) F8(6)
        adj = torch.zeros(n_channels, n_channels)
        edges = [
            (0, 1),  # FP1-FP2
            (0, 2),  # FP1-F7
            (0, 3),  # FP1-F3
            (1, 5),  # FP2-F4
            (1, 6),  # FP2-F8
            (2, 3),  # F7-F3
            (3, 4),  # F3-Fz
            (4, 5),  # Fz-F4
            (5, 6),  # F4-F8
        ]
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        # Add self-loops
        adj = adj + torch.eye(n_channels)
        # Normalize: D^{-1/2} A D^{-1/2}
        D = adj.sum(dim=1).pow(-0.5)
        D[D == float('inf')] = 0
        self.register_buffer('adj_norm', D.unsqueeze(1) * adj * D.unsqueeze(0))

        # Learnable adjacency residual
        self.adj_learnable = nn.Parameter(torch.zeros(n_channels, n_channels))

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, N, d) — batch, channels(7), patches, d_model
        returns: (B, C, N, d) — graph-convolved features
        """
        B, C, N, d = x.shape

        # Combine fixed + learnable adjacency
        adj = self.adj_norm + torch.sigmoid(self.adj_learnable)

        # Pool over patches for graph convolution
        x_pooled = x.mean(dim=2)  # (B, C, d)

        # GCN layers
        h = x_pooled
        for gcn, norm in zip(self.gcn_layers, self.norms):
            h_new = torch.matmul(adj, h)  # (B, C, d)
            h_new = gcn(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)  # residual

        # Broadcast back and add to original
        out = x + h.unsqueeze(2)
        return out
