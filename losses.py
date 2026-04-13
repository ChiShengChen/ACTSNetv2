import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, d_embed) — L2-normalized embeddings
        labels: (B,) — class labels
        """
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # Log-sum-exp trick
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits)
        # Exclude self-similarity
        self_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        log_prob = logits - torch.log((exp_logits * self_mask).sum(dim=1, keepdim=True) + 1e-7)

        # Mean over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob.mean()

        return loss


class ACTSNetV2Loss(nn.Module):
    """
    Combined loss for ACTSNet v2 training.

    L_total = L_CE + lambda_supcon * L_SupCon

    Args:
        lambda_supcon: weight for supervised contrastive loss
        lambda_proto: weight for prototype alignment loss (reserved)
        temperature: temperature for SupCon
    """

    def __init__(self, lambda_supcon=0.5, lambda_proto=0.1, temperature=0.07):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.lambda_supcon = lambda_supcon
        self.lambda_proto = lambda_proto

    def forward(self, logits, embeddings, labels):
        l_ce = self.ce_loss(logits, labels)
        l_supcon = self.supcon_loss(embeddings, labels)
        loss = l_ce + self.lambda_supcon * l_supcon
        return loss, {'ce': l_ce.item(), 'supcon': l_supcon.item()}
