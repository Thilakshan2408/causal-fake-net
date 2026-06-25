# ── CELL 1: Install ──────────────────────────────────────────────
!pip install torch_geometric transformers --quiet
print("Done.")

# ── CELL 2: Mount Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted.")

# ── CELL 3: Clone project from GitHub ───────────────────────────
import os

!git clone https://github.com/Thilakshan2408/causal-fake-net.git

os.chdir('/content/causal-fake-net/')
print("Working directory:", os.getcwd())
print("Files:", os.listdir())

# ── CELL 4: Copy Pheme from Drive to local storage ───────────────
import os

PHEME_DRIVE = '/content/drive/MyDrive/FYP_data/pheme-rnr-dataset'
PHEME_LOCAL = '/content/pheme'

print("Copying Pheme from Drive to local storage...")
print("Local storage is much faster than Drive for reading files.")
os.system(f"cp -r '{PHEME_DRIVE}' '{PHEME_LOCAL}'")
print("Done.")
print("Contents:", os.listdir(PHEME_LOCAL))

# ── CELL 5: Setup ────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# 3 balanced events only
EVENT_PATHS = {
    "ferguson":          os.path.join(PHEME_LOCAL, "ferguson"),
    "germanwings-crash": os.path.join(PHEME_LOCAL, "germanwings-crash"),
    "ottawashooting":    os.path.join(PHEME_LOCAL, "ottawashooting"),
}

# Cache saved to Drive so it survives disconnects
CACHE_DIR = '/content/drive/MyDrive/FYP_data/clip_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

device            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TEXT_TOKENS   = 6
NUM_IMAGE_PATCHES = 4

print(f"Device   : {device}")
print(f"Cache at : {CACHE_DIR}")
print("\nChecking event folders:")
for name, path in EVENT_PATHS.items():
    print(f"  {name:<22} exists={os.path.exists(path)}")

# ── CELL 6: Extract and save all CLIP features to Drive ──────────
import json, random
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
from data.pheme_colab import (
    CLIPExtractor, PhemeDataset,
    extract_srl_heuristic
)

clip_extractor = CLIPExtractor(
    device=device,
    num_text_tokens=NUM_TEXT_TOKENS,
    num_image_patches=NUM_IMAGE_PATCHES,
)

print("Building dataset splits...")
train_ds = PhemeDataset(EVENT_PATHS, clip_extractor,
                        split="train", cache_dir=CACHE_DIR)
val_ds   = PhemeDataset(EVENT_PATHS, clip_extractor,
                        split="val",   cache_dir=CACHE_DIR)
test_ds  = PhemeDataset(EVENT_PATHS, clip_extractor,
                        split="test",  cache_dir=CACHE_DIR)

print("\nExtracting CLIP features and saving to Drive...")
print("This is the only slow step — runs once only.\n")

total = len(train_ds) + len(val_ds) + len(test_ds)
done  = 0

for ds_name, ds in [("train", train_ds),
                     ("val",   val_ds),
                     ("test",  test_ds)]:
    for i in range(len(ds)):
        _ = ds[i]
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{total} saved to Drive...")

print(f"\nAll {total} samples cached to Drive.")
print("=" * 50)
print("SESSION 1 COMPLETE.")
print("You can now disconnect Colab safely.")
print("Open session_2.ipynb next time to train and get results.")
print("=" * 50)
