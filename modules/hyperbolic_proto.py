import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareOperations:
    """Poincare ball model operations."""

    @staticmethod
    def mobius_add(x, y, c=1.0):
        """Mobius addition in Poincare ball."""
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c**2 * x_sq * y_sq
        return num / denom.clamp(min=1e-7)

    @staticmethod
    def exp_map(x, v, c=1.0):
        """Exponential map from tangent space to Poincare ball."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        x_norm = (x * x).sum(dim=-1, keepdim=True)
        lambda_x = 2.0 / (1 - c * x_norm).clamp(min=1e-7)
        second = (torch.tanh(lambda_x * v_norm / 2) / v_norm) * v
        return PoincareOperations.mobius_add(x, second, c)

    @staticmethod
    def hyperbolic_distance(x, y, c=1.0):
        """Poincare distance."""
        diff = x - y
        diff_sq = (diff * diff).sum(dim=-1)
        x_sq = (x * x).sum(dim=-1)
        y_sq = (y * y).sum(dim=-1)
        denom = (1 - c * x_sq) * (1 - c * y_sq)
        arg = 1 + 2 * c * diff_sq / denom.clamp(min=1e-7)
        return (1.0 / c**0.5) * torch.acosh(arg.clamp(min=1.0 + 1e-7))


class HyperbolicPrototypicalHead(nn.Module):
    """
    Prototypical classification head in hyperbolic (Poincare) space.

    Upgrades the original Euclidean prototypical network:
    - Embeddings live in the Poincare ball for better hierarchy capture
    - Distance computed via hyperbolic geodesic

    Args:
        d_model: input feature dimension
        d_embed: prototype embedding dimension
        n_classes: number of classes
        curvature: Poincare ball curvature (c > 0)
    """

    def __init__(self, d_model=128, d_embed=64, n_classes=2, curvature=1.0):
        super().__init__()
        self.d_embed = d_embed
        self.n_classes = n_classes
        self.c = curvature
        self.poincare = PoincareOperations()

        # Projection to embedding space
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_embed),
        )

        # Learnable prototype embeddings (in tangent space, mapped to Poincare)
        self.prototypes_tangent = nn.Parameter(
            torch.randn(n_classes, d_embed) * 0.01
        )

        # Attention for prototype computation (from TapNet)
        self.proto_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_embed, d_embed // 2),
                nn.Tanh(),
                nn.Linear(d_embed // 2, 1),
            )
            for _ in range(n_classes)
        ])

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def get_prototypes(self):
        """Map prototypes from tangent space to Poincare ball."""
        origin = torch.zeros_like(self.prototypes_tangent)
        return self.poincare.exp_map(origin, self.prototypes_tangent, self.c)

    def forward(self, x, labels=None):
        """
        x: (B, d_model) — global feature vector
        labels: (B,) — class labels (used during training for prototype update)
        returns: logits (B, n_classes), embeddings (B, d_embed)
        """
        # Project to embedding space
        embeddings = self.projector(x)
        # Normalize to stay inside Poincare ball (||x|| < 1/sqrt(c))
        max_norm = (1.0 / self.c**0.5) - 1e-3
        norms = embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings / norms.clamp(min=1e-7) * norms.clamp(max=max_norm)

        # Get prototypes
        prototypes = self.get_prototypes()  # (n_classes, d_embed)

        # If training with labels, update prototypes with attention
        if labels is not None and self.training:
            for k in range(self.n_classes):
                mask = labels == k
                if mask.sum() > 0:
                    class_embeds = embeddings[mask]  # (n_k, d_embed)
                    attn_scores = self.proto_attention[k](class_embeds)  # (n_k, 1)
                    attn_weights = F.softmax(attn_scores, dim=0)
                    weighted_mean_tangent = (class_embeds * attn_weights).sum(0)
                    # Update prototype with momentum
                    with torch.no_grad():
                        self.prototypes_tangent.data[k] = (
                            0.9 * self.prototypes_tangent.data[k]
                            + 0.1 * weighted_mean_tangent
                        )
            prototypes = self.get_prototypes()

        # Compute hyperbolic distances
        distances = []
        for k in range(self.n_classes):
            d = self.poincare.hyperbolic_distance(
                embeddings, prototypes[k].unsqueeze(0).expand_as(embeddings), self.c
            )
            distances.append(d)
        distances = torch.stack(distances, dim=-1)  # (B, n_classes)

        # Convert distances to logits (negative distance = higher similarity)
        logits = -distances * F.softplus(self.temperature)

        return logits, embeddings
