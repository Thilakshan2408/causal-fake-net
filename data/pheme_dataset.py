"""
Pheme Dataset Loader with CLIP Feature Extraction
===================================================
Loads the real Pheme fake news dataset and extracts
CLIP features ready for CausalFakeNet.

Usage in Colab:
    from data.pheme_dataset import get_pheme_dataloaders
    train_loader, val_loader, test_loader = get_pheme_dataloaders(
        pheme_root='/content/drive/MyDrive/FYP_data/pheme-rnr-dataset'
    )
"""

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import requests
from io import BytesIO


# ─────────────────────────────────────────────────────────────────────────────
# SRL heuristic extractor
# (Simple rule-based — no external NLP tool needed)
# ─────────────────────────────────────────────────────────────────────────────

CAUSE_KEYWORDS  = ["because", "caused", "due to", "as a result of",
                   "following", "after", "triggered by", "led to"]
EFFECT_KEYWORDS = ["resulting in", "causing", "which caused", "therefore",
                   "consequently", "so that", "leading to"]


def extract_srl_heuristic(text: str) -> dict:
    """
    Simple heuristic SRL extraction from text.
    Splits text around cause/effect keywords.
    Falls back to first/second half of text if no keywords found.
    """
    text = text.lower().strip()
    words = text.split()
    n = len(words)

    # Default: split text in half
    mid = max(1, n // 2)
    cause_text  = " ".join(words[:mid])
    effect_text = " ".join(words[mid:]) if mid < n else cause_text

    # Try to find cause keyword
    for kw in CAUSE_KEYWORDS:
        if kw in text:
            parts = text.split(kw, 1)
            cause_text  = parts[0].strip()
            effect_text = parts[1].strip() if len(parts) > 1 else cause_text
            break

    # Try to find effect keyword
    for kw in EFFECT_KEYWORDS:
        if kw in text:
            parts = text.split(kw, 1)
            cause_text  = parts[0].strip()
            effect_text = parts[1].strip() if len(parts) > 1 else cause_text
            break

    # Token indices (approximate — map to first few tokens)
    return {
        "subject":     0,
        "predicate":   1,
        "object":      2,
        "cause":       0,
        "effect":      min(3, max(1, n // 3)),
        "location":    min(4, n - 1),
        "cause_text":  cause_text[:100],   # truncate for display
        "effect_text": effect_text[:100],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class CLIPExtractor:
    """
    Extracts CLIP features from raw text and images.
    Downloads CLIP model automatically on first use.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading CLIP model ({model_name})...")
        self.model     = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP model loaded.")

    @torch.no_grad()
    def extract_text(self, text: str,
                     num_tokens: int = 12) -> torch.Tensor:
        """
        Returns (num_tokens, 512) text features.
        """
        # Truncate text to CLIP max length
        text = text[:512] if len(text) > 512 else text
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        outputs = self.model.text_model(**inputs)
        # last_hidden_state: (1, seq_len, 512)
        hidden = outputs.last_hidden_state[0]   # (seq_len, 512)

        # Pad or truncate to num_tokens
        if hidden.size(0) >= num_tokens:
            hidden = hidden[:num_tokens]
        else:
            pad = torch.zeros(num_tokens - hidden.size(0), 512,
                              device=self.device)
            hidden = torch.cat([hidden, pad], dim=0)

        return F.normalize(hidden, dim=-1).cpu()

    @torch.no_grad()
    def extract_image(self, image,
                      num_patches: int = 8) -> torch.Tensor:
        """
        Returns (num_patches, 512) image features.
        Accepts PIL Image or None (returns zeros if None).
        """
        if image is None:
            return torch.zeros(num_patches, 512)

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.vision_model(**inputs)
            # last_hidden_state: (1, num_patches+1, 512)
            hidden = outputs.last_hidden_state[0]   # (num_patches+1, 512)

            # Skip CLS token, take patch tokens
            patches = hidden[1:]   # (num_patches_actual, 512)

            if patches.size(0) >= num_patches:
                patches = patches[:num_patches]
            else:
                pad = torch.zeros(num_patches - patches.size(0), 512,
                                  device=self.device)
                patches = torch.cat([patches, pad], dim=0)

            return F.normalize(patches, dim=-1).cpu()

        except Exception as e:
            print(f"Image extraction failed: {e}")
            return torch.zeros(num_patches, 512)


# ─────────────────────────────────────────────────────────────────────────────
# Pheme Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PhemeDataset(Dataset):
    """
    Loads the Pheme dataset from disk and extracts CLIP features.

    Pheme folder structure:
        pheme-rnr-dataset/
        ├── charliehebdo/
        │   ├── non-rumours/
        │   │   └── <thread_id>/
        │   │       ├── source-tweet/
        │   │       │   └── <tweet_id>.json
        │   │       └── images/          (may not exist)
        │   │           └── *.jpg
        │   └── rumours/
        │       └── <thread_id>/
        │           ├── source-tweet/
        │           │   └── <tweet_id>.json
        │           └── images/
    """

    EVENTS = [
        "charliehebdo",
        "ebola",
        "ferguson",
        "germanwings-crash",
        "ottawashooting",
    ]

    def __init__(
        self,
        pheme_root: str,
        clip_extractor: CLIPExtractor,
        split: str = "train",        # "train", "val", "test"
        train_ratio: float = 0.7,
        val_ratio:   float = 0.15,
        num_text_tokens: int = 12,
        num_image_patches: int = 8,
        seed: int = 42,
        cache_dir: str = "clip_cache",
    ):
        super().__init__()
        self.clip      = clip_extractor
        self.n_tok     = num_text_tokens
        self.n_patch   = num_image_patches
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Load all samples
        all_samples = self._load_pheme(pheme_root)
        print(f"Total Pheme samples loaded: {len(all_samples)}")

        # Count labels
        n_fake = sum(1 for s in all_samples if s["label"] == 1)
        n_real = sum(1 for s in all_samples if s["label"] == 0)
        print(f"  Real: {n_real}  Fake: {n_fake}")

        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val:]

        print(f"Split '{split}': {len(self.samples)} samples")

    def _load_pheme(self, root: str) -> list:
        samples = []

        for event in self.EVENTS:
            event_path = os.path.join(root, event)
            if not os.path.exists(event_path):
                # Try without hyphens or with underscores
                event_path = os.path.join(root, event.replace("-", "_"))
            if not os.path.exists(event_path):
                print(f"  Warning: event folder not found: {event}")
                continue

            for label_str, label_int in [("rumours", 1),
                                          ("non-rumours", 0)]:
                label_path = os.path.join(event_path, label_str)
                if not os.path.exists(label_path):
                    continue

                for thread_id in os.listdir(label_path):
                    thread_path = os.path.join(label_path, thread_id)
                    if not os.path.isdir(thread_path):
                        continue

                    # Load source tweet text
                    tweet_path = os.path.join(thread_path, "source-tweet")
                    text = self._load_tweet_text(tweet_path)
                    if text is None:
                        continue

                    # Load image if available
                    image_path = os.path.join(thread_path, "images")
                    image = self._load_image(image_path)

                    samples.append({
                        "text":     text,
                        "image":    image,
                        "label":    label_int,
                        "event":    event,
                        "thread_id": thread_id,
                    })

        return samples

    def _load_tweet_text(self, tweet_dir: str):
        if not os.path.exists(tweet_dir):
            return None
        for fname in os.listdir(tweet_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(tweet_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Pheme tweet JSON has 'text' field
                    text = data.get("text", "")
                    if text:
                        return text
                except Exception:
                    continue
        return None

    def _load_image(self, image_dir: str):
        if not os.path.exists(image_dir):
            return None
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                fpath = os.path.join(image_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    return img
                except Exception:
                    continue
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Check cache
        cache_key = f"{sample['event']}_{sample['thread_id']}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            cached = torch.load(cache_path)
            text_feats  = cached["text_feats"]
            image_feats = cached["image_feats"]
        else:
            # Extract CLIP features
            text_feats  = self.clip.extract_text(
                sample["text"], self.n_tok
            )
            image_feats = self.clip.extract_image(
                sample["image"], self.n_patch
            )
            # Save to cache
            torch.save({
                "text_feats":  text_feats,
                "image_feats": image_feats,
            }, cache_path)

        # Extract SRL roles from text
        srl_roles = extract_srl_heuristic(sample["text"])

        return {
            "text_feats":  text_feats,
            "image_feats": image_feats,
            "srl_roles":   srl_roles,
            "label":       sample["label"],
            "text":        sample["text"],
        }


def collate_pheme(batch: list) -> dict:
    text_feats  = torch.stack([s["text_feats"]  for s in batch])
    image_feats = torch.stack([s["image_feats"] for s in batch])
    labels      = torch.tensor([s["label"] for s in batch],
                                dtype=torch.long)
    srl_roles   = [s["srl_roles"] for s in batch]
    texts       = [s["text"]      for s in batch]

    return {
        "text_feats":  text_feats,
        "image_feats": image_feats,
        "srl_roles":   srl_roles,
        "labels":      labels,
        "texts":       texts,
    }


def get_pheme_dataloaders(
    pheme_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    cache_dir:  str = "clip_cache",
):
    """
    Main entry point.

    Args:
        pheme_root: path to pheme-rnr-dataset folder
        batch_size: samples per batch
        cache_dir:  where to save extracted CLIP features

    Returns:
        train_loader, val_loader, test_loader, clip_extractor
    """
    clip = CLIPExtractor()

    train_ds = PhemeDataset(pheme_root, clip, split="train",
                            cache_dir=cache_dir)
    val_ds   = PhemeDataset(pheme_root, clip, split="val",
                            cache_dir=cache_dir)
    test_ds  = PhemeDataset(pheme_root, clip, split="test",
                            cache_dir=cache_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_pheme,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pheme,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pheme,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, clip
