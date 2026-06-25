# ── CELL 1: Install ──────────────────────────────────────────────
!pip install torch_geometric transformers scikit-learn matplotlib --quiet
print("Done.")

# ── CELL 2: Mount Drive and clone project ────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os

!git clone https://github.com/Thilakshan2408/causal-fake-net.git

os.chdir('/content/causal-fake-net/')  # 'causal_fake_net' last part
print("Working directory:", os.getcwd())
print("Files:", os.listdir())

# ── CELL 3: Setup ────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

PHEME_DRIVE = '/content/drive/MyDrive/FYP_data/pheme-rnr-dataset'

EVENT_PATHS = {
    "ferguson":          os.path.join(PHEME_DRIVE, "ferguson"),
    "germanwings-crash": os.path.join(PHEME_DRIVE, "germanwings-crash"),
    "ottawashooting":    os.path.join(PHEME_DRIVE, "ottawashooting"),
}

CACHE_DIR  = '/content/drive/MyDrive/FYP_data/clip_cache'
CKPT_PATH  = '/content/drive/MyDrive/FYP_data/model.pt'

device            = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIM          = 512
NUM_TEXT_TOKENS   = 6
NUM_IMAGE_PATCHES = 4
BATCH_SIZE        = 8
MAX_EPOCHS        = 20
PATIENCE          = 7

# Check samples count and checkpoint path
cache_count   = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
model_exists  = os.path.exists(CKPT_PATH)

print(f"Device          : {device}")
print(f"Cached features : {cache_count} files")
print(f"Trained model   : "
      f"{'FOUND in Drive' if model_exists else 'not found yet'}")

if cache_count == 0:
    print("\nWARNING: No cached features found.")
    print("Please run session_1.ipynb first.")

# ── CELL 4: Load dataset from cache (Drive) ────────────────────
from torch.utils.data import DataLoader
from data.pheme_colab import (
    CLIPExtractor, PhemeDataset,
    collate_pheme, extract_srl_heuristic,
)

clip_extractor = CLIPExtractor(
    device=device,
    num_text_tokens=NUM_TEXT_TOKENS,
    num_image_patches=NUM_IMAGE_PATCHES,
)

print("Loading dataset from Drive cache...")
train_ds = PhemeDataset(EVENT_PATHS, clip_extractor, split="train", cache_dir=CACHE_DIR)
val_ds   = PhemeDataset(EVENT_PATHS, clip_extractor, split="val",   cache_dir=CACHE_DIR)
test_ds  = PhemeDataset(EVENT_PATHS, clip_extractor, split="test",  cache_dir=CACHE_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pheme)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pheme)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pheme)

print(f"\nDataset loaded instantly from cache.")
print(f"Train : {len(train_loader)} batches")
print(f"Val   : {len(val_loader)} batches")
print(f"Test  : {len(test_loader)} batches")

# ── CELL 5: Train CausalFakeNet ──────────────────────────────────
# If not already trained

from models.causal_fake_net import CausalFakeNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model     = CausalFakeNet(feat_dim=FEAT_DIM, num_gcn_layers=1).to(device)
optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

history    = {"train_loss": [], "val_acc": [], "val_f1": []}
best_f1    = 0.0
no_improve = 0

print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"Device     : {device}")
print(f"Max epochs : {MAX_EPOCHS}")

# Create the specific subdirectories inside base CKPT_PATH directory
best_dir = os.path.join(WORKING_DIR, "best")
last_dir = os.path.join(WORKING_DIR, "last")

os.makedirs(best_dir, exist_ok=True)
os.makedirs(last_dir, exist_ok=True)

# Define the full file paths for both checkpoints
BEST_CKPT_FILE = os.path.join(best_dir, "model.pt")
LAST_CKPT_FILE = os.path.join(last_dir, "model.pt")

print("\nTraining...\n")

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        tf  = batch["text_feats"].to(device)
        vf  = batch["image_feats"].to(device)
        srl = batch["srl_roles"]
        y   = batch["labels"].to(device)
        optimizer.zero_grad()
        logits, cred, ew, fused = model(tf, vf, srl)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    # Validate
    model.eval()
    vp, vl = [], []
    with torch.no_grad():
        for batch in val_loader:
            tf  = batch["text_feats"].to(device)
            vf  = batch["image_feats"].to(device)
            srl = batch["srl_roles"]
            logits, *_ = model(tf, vf, srl)
            vp.extend(logits.argmax(-1).cpu().tolist())
            vl.extend(batch["labels"].tolist())

    val_acc  = accuracy_score(vl, vp)
    val_f1   = f1_score(vl, vp, average="binary", zero_division=0)
    avg_loss = epoch_loss / len(train_loader)

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
          f"loss={avg_loss:.4f}  "
          f"val_acc={val_acc:.4f}  "
          f"val_f1={val_f1:.4f}")

    # --- CHECKPOINT SAVING LOGIC ---
    # Save the absolute BEST model based on val_f1
    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        torch.save(
            {"model_state": model.state_dict(),
             "val_f1":      val_f1,
             "epoch":       epoch,
             "history":     history},
            BEST_CKPT_FILE)  # <--- Changed to save in 'best' folder
        print(f"  ✓ Saved to Drive/best (val_f1={val_f1:.4f})")
    else:
        no_improve += 1

    # Save the LAST model explicitly if it reaches the final epoch
    if epoch == MAX_EPOCHS:
        torch.save(
            {"model_state": model.state_dict(),
             "val_f1":      val_f1,
             "epoch":       epoch,
             "history":     history},
            LAST_CKPT_FILE)  # <--- Changed to save in 'last' folder
        print(f"  💾 Final 30th epoch model saved to Drive/last")

print("\nTraining complete. Model saved to Drive.")

# ── CELL 6: Load model and test evaluation ───────────────────────
# loads model directly from Drive

from models.causal_fake_net import CausalFakeNet

model = CausalFakeNet(feat_dim=FEAT_DIM, num_gcn_layers=1).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

history = ckpt.get(
    "history",
    {"train_loss": [], "val_acc": [], "val_f1": []})

print(f"Loaded model from epoch {ckpt['epoch']} "
      f"(val_f1={ckpt['val_f1']:.4f})")

# Test
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        tf  = batch["text_feats"].to(device)
        vf  = batch["image_feats"].to(device)
        srl = batch["srl_roles"]
        logits, *_ = model(tf, vf, srl)
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["labels"].tolist())

acc  = accuracy_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)
prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)

print("\n" + "=" * 50)
print("CausalFakeNet — Pheme Test Results")
print("=" * 50)
print(f"  Accuracy  : {acc:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print("=" * 50)
print("Screenshot this for your FYP report.")

# ── CELL 7: Confusion matrix ─────────────────────────────────────
# %%
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("CausalFakeNet — Pheme test set")
plt.tight_layout()
save_path = '/content/drive/MyDrive/FYP_data/confusion_matrix.png'
plt.savefig(save_path, dpi=150)
plt.show()
print(f"Saved to Drive: confusion_matrix.png")

# ── CELL 8: Training curves ──────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"],
             label="Train loss", color="steelblue")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training loss — CausalFakeNet")
axes[0].legend()

axes[1].plot(history["val_acc"],
             label="Val accuracy", color="green")
axes[1].plot(history["val_f1"],
             label="Val F1",       color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_title("Validation metrics")
axes[1].legend()

plt.tight_layout()
save_path = '/content/drive/MyDrive/FYP_data/training_curves.png'
plt.savefig(save_path, dpi=150)
plt.show()
print("Saved to Drive: training_curves.png")

# ── CELL 9: Baseline comparison ──────────────────────────────────
import torch.nn as nn

class CLIPWithMLP(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 2),
        )
    def forward(self, tf, vf, srl=None):
        return self.clf(
            torch.cat([tf[:, 0, :],
                       vf[:, 0, :]], dim=-1))

class TextOnly(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 2),
        )
    def forward(self, tf, vf=None, srl=None):
        return self.clf(tf[:, 0, :])

def run_baseline(bl_model, name):
    bl_model = bl_model.to(device)
    opt      = AdamW(bl_model.parameters(), lr=1e-3)
    best     = 0.0
    no_imp   = 0

    for epoch in range(1, 21):
        bl_model.train()
        for batch in train_loader:
            tf = batch["text_feats"].to(device)
            vf = batch["image_feats"].to(device)
            y  = batch["labels"].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(bl_model(tf, vf), y)
            loss.backward()
            opt.step()

        bl_model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                tf = batch["text_feats"].to(device)
                vf = batch["image_feats"].to(device)
                vp.extend(
                    bl_model(tf, vf).argmax(-1).cpu().tolist())
                vl.extend(batch["labels"].tolist())

        vf1 = f1_score(vl, vp, average="binary",
                       zero_division=0)
        if vf1 > best:
            best   = vf1
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 7:
                break

    # Test
    bl_model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            tf = batch["text_feats"].to(device)
            vf = batch["image_feats"].to(device)
            tp.extend(
                bl_model(tf, vf).argmax(-1).cpu().tolist())
            tl.extend(batch["labels"].tolist())

    return accuracy_score(tl, tp), \
           f1_score(tl, tp, average="binary", zero_division=0)

print("Training baselines for comparison table...")
print("Takes about 5-10 minutes on GPU.\n")

r_text = run_baseline(TextOnly(),    "Text Only")
print(f"  Text Only  : acc={r_text[0]:.4f}  f1={r_text[1]:.4f}")

r_clip = run_baseline(CLIPWithMLP(), "CLIP + MLP")
print(f"  CLIP + MLP : acc={r_clip[0]:.4f}  f1={r_clip[1]:.4f}")

print("\n" + "=" * 58)
print("COMPARISON TABLE — Pheme dataset")
print("(ferguson + germanwings-crash + ottawashooting)")
print("=" * 58)
print(f"{'Method':<22} {'Accuracy':>10} {'F1 Score':>10}")
print("-" * 58)
print(f"{'Text Only':<22} "
      f"{r_text[0]:>10.4f} {r_text[1]:>10.4f}")
print(f"{'CLIP + MLP':<22} "
      f"{r_clip[0]:>10.4f} {r_clip[1]:>10.4f}")
print(f"{'CausalFakeNet':<22} "
      f"{acc:>10.4f} {f1:>10.4f}  <- Ours")
print("=" * 58)
print("\nScreenshot this table for your FYP report.")

# ── CELL 10: Rationale demo on real tweets ───────────────────────
# %%
model.eval()

demo_batch = next(iter(test_loader))
tf    = demo_batch["text_feats"][:6].to(device)
vf    = demo_batch["image_feats"][:6].to(device)
srl   = demo_batch["srl_roles"][:6]
y     = demo_batch["labels"][:6]
texts = demo_batch["texts"][:6]

explanations = model.explain(tf, vf, srl)

print("=" * 60)
print("RATIONALE GENERATION — Real Pheme tweets")
print("=" * 60)

for i, (expl, label, text) in enumerate(
        zip(explanations, y.tolist(), texts)):
    gt = "FAKE" if label == 1 else "REAL"
    print(f"\n[Sample {i+1}] Ground truth: {gt}")
    print(f"  Tweet  : {text[:80]}...")
    print(f"  Cause  : {srl[i].get('cause_text',  'N/A')}")
    print(f"  Effect : {srl[i].get('effect_text', 'N/A')}")
    print(f"  Result : {expl}")

print("\nScreenshot this for your FYP report.")
