"""
Baseline models for FYP comparison table.

Baselines:
    1. CLIP + MLP   — simplest multimodal model
    2. TextOnly     — ignores image completely
    3. ImageOnly    — ignores text completely

These give you a comparison table in your FYP report showing
CausalFakeNet outperforms simpler approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPWithMLP(nn.Module):
    """
    Simplest possible baseline:
    Concatenate CLIP text CLS + image CLS → MLP → classify.
    No graph, no causal edges, no credibility.
    """

    def __init__(self, feat_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, text_feats, image_feats, srl_roles=None):
        text_cls  = text_feats[:, 0, :]    # (B, d)
        image_cls = image_feats[:, 0, :]   # (B, d)
        combined  = torch.cat([text_cls, image_cls], dim=-1)  # (B, 2d)
        return self.classifier(combined)


class TextOnlyMLP(nn.Module):
    """
    Text-only baseline — ignores image entirely."""

    def __init__(self, feat_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, text_feats, image_feats=None, srl_roles=None):
        text_cls = text_feats[:, 0, :]
        return self.classifier(text_cls)


class ImageOnlyMLP(nn.Module):
    """
    Image-only baseline — ignores text entirely.
    """

    def __init__(self, feat_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, text_feats=None, image_feats=None, srl_roles=None):
        image_cls = image_feats[:, 0, :]
        return self.classifier(image_cls)


def train_baseline(model, train_loader, val_loader,
                   num_epochs=30, lr=1e-3, device="cuda",
                   patience=10):
    """
    Simple training loop for baseline models.
    Returns best val_f1 achieved.
    """
    from sklearn.metrics import f1_score, accuracy_score

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            tf  = batch["text_feats"].to(device)
            vf  = batch["image_feats"].to(device)
            y   = batch["labels"].to(device)
            srl = batch["srl_roles"]

            logits = model(tf, vf, srl)
            loss   = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                tf  = batch["text_feats"].to(device)
                vf  = batch["image_feats"].to(device)
                y   = batch["labels"]
                srl = batch["srl_roles"]
                logits = model(tf, vf, srl)
                preds  = logits.argmax(-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())

        val_f1  = f1_score(all_labels, all_preds, average="binary",
                           zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_f1, val_acc
