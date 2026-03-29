"""
Loss functions for CausalFakeNet.

L_total = L_cls
        + λ1 * L_credible   (Beta/digamma — from Event-Radar)
        + λ2 * L_contrastive (push low-credibility features away from fused rep)
        + λ3 * L_causal      (NOVEL: regularise causal edge weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma


class CredibleLoss(nn.Module):
    """
    Beta-distribution credible loss (Event-Radar §3.5).
    Encourages confident per-view classification.
    """

    def forward(
        self,
        logits: torch.Tensor,   # (B, 2)  per-view logits
        labels: torch.Tensor,   # (B,)    ground truth 0/1
    ) -> torch.Tensor:
        evidence = F.softplus(logits)          # (B, 2)
        beta     = 1.0 + evidence              # (B, 2)
        S        = beta.sum(dim=-1)            # (B,)

        # one-hot labels
        y = F.one_hot(labels, num_classes=2).float()  # (B, 2)

        loss = (y * (digamma(S.unsqueeze(-1)) - digamma(beta))).sum(-1)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss from Event-Radar §3.5.
    Pulls high-credibility view representations toward the fused rep,
    pushes low-credibility ones away.
    """

    def forward(
        self,
        x_c: torch.Tensor,     # (B, d)
        x_e: torch.Tensor,     # (B, d)
        x_p: torch.Tensor,     # (B, d)
        fused: torch.Tensor,   # (B, d)
        credibility: torch.Tensor,  # (B, 3)  [q_c, q_e, q_p]
    ) -> torch.Tensor:
        views = torch.stack([x_c, x_e, x_p], dim=1)  # (B, 3, d)
        # Cosine similarity of each view with the fused representation
        sim = F.cosine_similarity(
            views, fused.unsqueeze(1).expand_as(views), dim=-1
        )  # (B, 3)

        q_min, _ = credibility.min(dim=-1)   # (B,)

        # Weighted pull/push
        loss = (sim * (1 - credibility) + (1 - sim) * credibility).mean()
        return loss


class CausalEdgeLoss(nn.Module):
    """
    NOVEL regulariser for causal edge weights.

    Encourages the model to:
      (a) assign HIGH weight to causal edges that correctly separate
          real vs. fake news (fake news tends to have lower causal
          consistency, so we want those edges to be active)
      (b) not collapse all edge weights to zero

    Implementation: a simple entropy regulariser that penalises
    degenerate (all-zero or all-one) edge weight distributions,
    applied only to cause→effect edges.
    """

    def forward(
        self,
        edge_weights_list: list,   # list of (E,) tensors, one per graph
        causal_masks: list,        # list of (E,) bool tensors
    ) -> torch.Tensor:
        total = torch.tensor(0.0)
        count = 0

        for ew, mask in zip(edge_weights_list, causal_masks):
            if mask.sum() == 0:
                continue
            causal_ew = ew[mask].clamp(1e-6, 1 - 1e-6)
            # Binary entropy: -p log p - (1-p) log(1-p)
            entropy = -(causal_ew * causal_ew.log()
                        + (1 - causal_ew) * (1 - causal_ew).log())
            # We want HIGH entropy (spread of weights), not collapsed
            # Penalise low entropy (degenerate distributions)
            total = total + (1.0 - entropy.mean())
            count += 1

        if count == 0:
            return torch.tensor(0.0)
        return total / count


class CausalFakeNetLoss(nn.Module):
    """
    Combined loss function.
    """

    def __init__(self, lambda1: float = 0.4, lambda2: float = 0.2,
                 lambda3: float = 0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.cls_loss       = nn.CrossEntropyLoss()
        self.credible_loss  = CredibleLoss()
        self.contrast_loss  = ContrastiveLoss()
        self.causal_loss    = CausalEdgeLoss()

    def forward(
        self,
        logits: torch.Tensor,           # (B, 2)
        labels: torch.Tensor,           # (B,)
        credibility: torch.Tensor,      # (B, 3)
        x_c: torch.Tensor,             # (B, d)
        x_e: torch.Tensor,             # (B, d)
        x_p: torch.Tensor,             # (B, d)
        fused: torch.Tensor,           # (B, d)
        edge_weights_list: list,        # list[(E,)]
        causal_masks: list,             # list[(E,)] bool
        view_logits: dict,              # {'c': (B,2), 'e': (B,2), 'p': (B,2)}
    ) -> dict:
        L_cls = self.cls_loss(logits, labels)

        L_cred = (
            self.credible_loss(view_logits["c"], labels)
            + self.credible_loss(view_logits["e"], labels)
            + self.credible_loss(view_logits["p"], labels)
        ) / 3.0

        L_contrast = self.contrast_loss(x_c, x_e, x_p, fused, credibility)

        L_causal = self.causal_loss(edge_weights_list, causal_masks)

        L_total = (L_cls
                   + self.lambda1 * L_cred
                   + self.lambda2 * L_contrast
                   + self.lambda3 * L_causal)

        return {
            "total":      L_total,
            "cls":        L_cls,
            "credible":   L_cred,
            "contrastive": L_contrast,
            "causal":     L_causal,
        }
