"""
Synthetic dataset for CausalFakeNet prototype.

Since the real Twitter/Weibo/Pheme datasets require registration,
this module generates realistic synthetic CLIP-like features for
quick Colab prototyping.

To swap in real data: subclass CausalNewsDataset and override __getitem__
to load your actual CLIP features and SRL annotations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random


# ─────────────────────────────────────────────────────────────────────────────
# SRL role vocabulary (mimics Stanford CoreNLP / TextSmart output)
# ─────────────────────────────────────────────────────────────────────────────

FAKE_SRL_EXAMPLES = [
    {
        "subject": 1, "predicate": 2, "object": 3,
        "cause": 4, "effect": 5, "location": 6,
        "cause_text": "protesters blocking roads",
        "effect_text": "police using force",
    },
    {
        "subject": 1, "predicate": 2, "object": 3,
        "cause": 1, "effect": 3, "location": 0,
        "cause_text": "flooding from storm",
        "effect_text": "houses being destroyed",
    },
    {
        "subject": 2, "predicate": 1, "object": 4,
        "cause": 2, "effect": 5, "location": 3,
        "cause_text": "government policy change",
        "effect_text": "mass unemployment",
    },
]

REAL_SRL_EXAMPLES = [
    {
        "subject": 0, "predicate": 1, "object": 2,
        "cause": 0, "effect": 2, "location": 3,
        "cause_text": "earthquake measuring 6.2",
        "effect_text": "buildings collapsing in city centre",
    },
    {
        "subject": 1, "predicate": 0, "object": 3,
        "cause": 1, "effect": 4, "location": 2,
        "cause_text": "vaccine trial success",
        "effect_text": "regulatory approval granted",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

class CausalNewsDataset(Dataset):
    """
    Generates synthetic CLIP-like features.

    Fake news: image and text features are deliberately misaligned
               (low cosine similarity between cross-modal entities)
    Real news: image and text features are well aligned

    This simulates the multimodal inconsistency that CausalFakeNet
    is trained to detect.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        feat_dim: int = 512,
        num_text_tokens: int = 12,
        num_image_patches: int = 8,
        fake_ratio: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        random.seed(seed)

        self.feat_dim = feat_dim
        self.num_text_tokens = num_text_tokens
        self.num_image_patches = num_image_patches

        self.data = []
        num_fake = int(num_samples * fake_ratio)
        num_real = num_samples - num_fake

        for i in range(num_real):
            self.data.append(self._make_sample(label=0))

        for i in range(num_fake):
            self.data.append(self._make_sample(label=1))

        random.shuffle(self.data)

    def _make_sample(self, label: int) -> dict:
        d = self.feat_dim
        m = self.num_text_tokens
        n = self.num_image_patches

        if label == 0:
            # Real: image and text share the SAME semantic base direction
            # Very low noise = strong alignment = clearly real
            base_t = F.normalize(torch.randn(d), dim=0)
            base_v = base_t  # same base for text and image
            noise_t = 0.05
            noise_v = 0.05
        else:
            # Fake: image comes from a DIFFERENT semantic cluster
            # Text and image point in completely different directions
            base_t = F.normalize(torch.randn(d), dim=0)
            base_v = F.normalize(torch.randn(d), dim=0)  # independent base
            # Make sure they are dissimilar by flipping half the dims
            base_v = F.normalize(base_v - 2 * (base_v * base_t).sum() * base_t, dim=0)
            noise_t = 0.05
            noise_v = 0.05

        text_feats  = base_t.unsqueeze(0) + noise_t * torch.randn(m, d)
        image_feats = base_v.unsqueeze(0) + noise_v * torch.randn(n, d)

        # L2-normalise (CLIP outputs are normalised)
        text_feats  = F.normalize(text_feats,  dim=-1)
        image_feats = F.normalize(image_feats, dim=-1)

        # Sample SRL roles
        if label == 1:
            srl = random.choice(FAKE_SRL_EXAMPLES).copy()
        else:
            srl = random.choice(REAL_SRL_EXAMPLES).copy()

        return {
            "text_feats":  text_feats,
            "image_feats": image_feats,
            "srl_roles":   srl,
            "label":       label,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def import_F():
    """Lazy import so the module works without torch.nn.functional at top level."""
    import torch.nn.functional as F
    return F


# Need F for _make_sample — pull it in at module level
import torch.nn.functional as F


def collate_fn(batch: list) -> dict:
    """
    Custom collate: stacks tensors, keeps srl_roles as a plain list.
    """
    text_feats  = torch.stack([s["text_feats"]  for s in batch])
    image_feats = torch.stack([s["image_feats"] for s in batch])
    labels      = torch.tensor([s["label"]      for s in batch], dtype=torch.long)
    srl_roles   = [s["srl_roles"] for s in batch]

    return {
        "text_feats":  text_feats,
        "image_feats": image_feats,
        "srl_roles":   srl_roles,
        "labels":      labels,
    }


def get_dataloaders(
    num_train: int = 3000,
    num_val:   int = 500,
    num_test:  int = 500,
    batch_size: int = 32,
    feat_dim:   int = 512,
):
    train_ds = CausalNewsDataset(num_train, feat_dim, seed=42)
    val_ds   = CausalNewsDataset(num_val,   feat_dim, seed=43)
    test_ds  = CausalNewsDataset(num_test,  feat_dim, seed=44)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
