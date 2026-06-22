import os
import json
import random
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor


CAUSE_KEYWORDS  = ["because", "caused", "due to",
                   "as a result of", "following",
                   "after", "triggered by", "led to"]
EFFECT_KEYWORDS = ["resulting in", "causing",
                   "which caused", "therefore",
                   "consequently", "leading to"]


def extract_srl_heuristic(text: str) -> dict:
    text_lower = text.lower().strip()
    words      = text_lower.split()
    n          = len(words)
    mid        = max(1, n // 2)
    cause_text  = " ".join(words[:mid])
    effect_text = " ".join(words[mid:]) if mid < n else cause_text

    for kw in CAUSE_KEYWORDS:
        if kw in text_lower:
            parts       = text_lower.split(kw, 1)
            cause_text  = parts[0].strip()
            effect_text = (parts[1].strip()
                           if len(parts) > 1 else cause_text)
            break

    for kw in EFFECT_KEYWORDS:
        if kw in text_lower:
            parts       = text_lower.split(kw, 1)
            cause_text  = parts[0].strip()
            effect_text = (parts[1].strip()
                           if len(parts) > 1 else cause_text)
            break

    return {
        "subject":     0,
        "predicate":   1,
        "object":      2,
        "cause":       0,
        "effect":      min(3, max(1, n // 3)),
        "location":    min(4, n - 1),
        "cause_text":  cause_text[:100],
        "effect_text": effect_text[:100],
    }


class CLIPExtractor:
    """
    Extracts CLIP features from text and images.
    """

    def __init__(self, device: str = "auto", num_text_tokens: int = 6, num_image_patches: int = 4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device           = device
        self.num_text_tokens  = num_text_tokens
        self.num_img_patches  = num_image_patches

        print(f"Loading CLIP on {device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model.eval()
        print("CLIP loaded.")

    @torch.no_grad()
    def extract_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text[:512]], return_tensors="pt", padding=True, truncation=True,max_length=77).to(self.device)
        hidden = self.model.text_model(**inputs).last_hidden_state[0]
        n = self.num_text_tokens
        if hidden.size(0) >= n:
            hidden = hidden[:n]
        else:
            # pad = torch.zeros(n - hidden.size(0), 512, device=self.device)
	    	pad = torch.zeros(n - hidden.size(0), hidden.size(-1), device=self.device)
            hidden = torch.cat([hidden, pad], dim=0)
        return F.normalize(hidden, dim=-1).cpu()

    @torch.no_grad()
    def extract_image(self, image) -> torch.Tensor:
        n = self.num_img_patches
        if image is None:
            return torch.zeros(n, 512)
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            patches = self.model.vision_model(**inputs).last_hidden_state[0][1:]
            if patches.size(0) >= n:
                patches = patches[:n]
            else:
                # pad = torch.zeros(n - patches.size(0), 512, device=self.device)
				pad = torch.zeros(n - patches.size(0), patches.size(-1), device=self.device)
                patches = torch.cat([patches, pad], dim=0)
			
	    	if pathches.size(-1) == 768:
			# Unsqueeze to add a temporary batch dimension for interpolate, then squeeze back
                patches = F.interpolate(patches.unsqueeze(0), size=512, mode='linear', align_corners=False).squeeze(0)

            return F.normalize(patches, dim=-1).cpu()
        except Exception:
            return torch.zeros(n, 512)


class PhemeDataset(Dataset):
    """
    Loads Pheme dataset from a dict of event paths.
    Each event path points to a folder containing
    rumours/ and non-rumours/ subfolders.

    Features are cached to Drive so extraction
    only happens once.
    """

    def __init__(
        self,
        event_paths: dict,
        clip_extractor: CLIPExtractor,
        split: str         = "train",
        train_ratio: float = 0.7,
        val_ratio:   float = 0.15,
        seed: int          = 42,
        cache_dir: str     = "/content/drive/MyDrive/FYP_data/clip_cache",
    ):
        super().__init__()
        self.clip      = clip_extractor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        all_samples = self._load_all(event_paths)
        print(f"Total samples : {len(all_samples)}")
        print(f"  Real : {sum(1 for s in all_samples if s['label']==0)}")
        print(f"  Fake : {sum(1 for s in all_samples if s['label']==1)}")

        random.seed(seed)
        random.shuffle(all_samples)
        n       = len(all_samples)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val:]

        print(f"Split '{split}' : {len(self.samples)} samples")

    def _load_all(self, event_paths: dict) -> list:
        samples = []
        for event_name, event_path in event_paths.items():
            if not os.path.exists(event_path):
                print(f"  Skipping missing: {event_name}")
                continue
            print(f"  Loading: {event_name}")
            for label_str, label_int in [("rumours",     1), ("non-rumours", 0)]:
                label_path = os.path.join(event_path, label_str)
                if not os.path.exists(label_path):
                    continue
                for thread_id in os.listdir(label_path):
                    thread_path = os.path.join(label_path,
                                               thread_id)
                    if not os.path.isdir(thread_path):
                        continue
                    text = None
                    for fn in ["source-tweets", "source-tweet"]:
                        tp = os.path.join(thread_path, fn)
                        if os.path.exists(tp):
                            text = self._load_tweet(tp)
                            if text:
                                break
                    if not text:
                        continue
                    image = self._load_image(os.path.join(thread_path, "images"))
                    samples.append({
                        "text":      text,
                        "image":     image,
                        "label":     label_int,
                        "event":     event_name,
                        "thread_id": thread_id,
                    })
        return samples

    def _load_tweet(self, tweet_dir: str):
        for fname in os.listdir(tweet_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(tweet_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
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
                try:
                    return Image.open(os.path.join(image_dir, fname)).convert("RGB")
                except Exception:
                    continue
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s          = self.samples[idx]
        cache_key  = f"{s['event']}_{s['thread_id']}"
        cache_path = os.path.join(self.cache_dir,
                                  f"{cache_key}.pt")

        if os.path.exists(cache_path):
            cached      = torch.load(cache_path,
                                     weights_only=True)
            text_feats  = cached["text_feats"]
            image_feats = cached["image_feats"]
        else:
            text_feats  = self.clip.extract_text(s["text"])
            image_feats = self.clip.extract_image(s["image"])
            torch.save({"text_feats":  text_feats,
                        "image_feats": image_feats},
                       cache_path)

        return {
            "text_feats":  text_feats,
            "image_feats": image_feats,
            "srl_roles":   extract_srl_heuristic(s["text"]),
            "label":       s["label"],
            "text":        s["text"],
        }


def collate_pheme(batch: list) -> dict:
    return {
        "text_feats":  torch.stack(
            [s["text_feats"]  for s in batch]),
        "image_feats": torch.stack(
            [s["image_feats"] for s in batch]),
        "labels":      torch.tensor(
            [s["label"] for s in batch],
            dtype=torch.long),
        "srl_roles":   [s["srl_roles"] for s in batch],
        "texts":       [s["text"]      for s in batch],
    }
