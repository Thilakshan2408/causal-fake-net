"""
Training loop for CausalFakeNet.

Run in Colab:
    !pip install torch torchvision torch_geometric
    from train import train

Or from command line:
    python train.py
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
import os
import json


def train(
    feat_dim:    int   = 512,
    num_epochs:  int   = 50,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    lambda1:     float = 0.1,
    lambda2:     float = 0.05,
    lambda3:     float = 0.05,
    patience:    int   = 10,
    device:      str   = "auto",
    save_dir:    str   = "checkpoints",
):
    """
    Full training run.

    Returns:
        model       — trained CausalFakeNet
        history     — dict with train/val loss and metrics per epoch
    """
    # ── Imports (inside function so Colab cells work independently) ──
    from models.causal_fake_net import CausalFakeNet
    from models.losses import CausalFakeNetLoss
    from data.dataset import get_dataloaders

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, feat_dim=feat_dim
    )

    # ── Model ──
    model = CausalFakeNet(feat_dim=feat_dim).to(device)
    criterion = CausalFakeNetLoss(lambda1, lambda2, lambda3)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [],
    }
    best_val_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            tf  = batch["text_feats"].to(device)   # (B, m, d)
            vf  = batch["image_feats"].to(device)  # (B, n, d)
            srl = batch["srl_roles"]
            y   = batch["labels"].to(device)

            optimizer.zero_grad()

            logits, credibility, edge_weights, fused = model(tf, vf, srl)

            # We also need per-view logits for the credible loss.
            # Re-run credibility heads on the intermediate features.
            # (In a cleaner impl these would be returned from forward;
            #  here we compute them efficiently by reusing stored tensors.)
            with torch.no_grad():
                graphs = model.graph_builder(tf, vf, srl)
                x_c_list = []
                for g in graphs:
                    h, _ = model.causal_gcn(g)
                    tm = (g.node_type == 0); im = (g.node_type == 1)
                    hP = h[tm].mean(0); hI = h[im].mean(0)
                    comp = torch.cat([hP, hI, hP - hI, hP * hI])
                    x_c_list.append(model.compare_proj(comp))
                x_c = torch.stack(x_c_list)

            x_e = model.emotion_enc(tf[:, 0, :])
            x_p = model.pattern_enc(vf)

            logits_c, _ = model.cred_c(x_c)
            logits_e, _ = model.cred_e(x_e)
            logits_p, _ = model.cred_p(x_p)

            # Causal masks for edge loss
            causal_masks = []
            with torch.no_grad():
                graphs2 = model.graph_builder(tf, vf, srl)
                for g2 in graphs2:
                    causal_masks.append(g2.causal_mask.to(device))

            loss_dict = criterion(
                logits=logits,
                labels=y,
                credibility=credibility,
                x_c=x_c, x_e=x_e, x_p=x_p,
                fused=fused,
                edge_weights_list=edge_weights,
                causal_masks=causal_masks,
                view_logits={"c": logits_c, "e": logits_e, "p": logits_p},
            )

            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss_dict["total"].item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──
        val_loss, val_acc, val_f1 = evaluate(model, val_loader,
                                              criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}")

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✓ New best model saved (val_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ── Test ──
    print("\n── Test set evaluation ──")
    test_loss, test_acc, test_f1 = evaluate(model, test_loader,
                                             criterion, device)
    print(f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        tf  = batch["text_feats"].to(device)
        vf  = batch["image_feats"].to(device)
        srl = batch["srl_roles"]
        y   = batch["labels"].to(device)

        logits, credibility, edge_weights, fused = model(tf, vf, srl)

        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    avg_loss = total_loss / len(loader)

    return avg_loss, acc, f1


if __name__ == "__main__":
    model, history = train(num_epochs=30, batch_size=16)
    print("Training complete.")
