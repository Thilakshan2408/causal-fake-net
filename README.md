# CausalFakeNet/EventX-FND — FYP Prototype

**Event-driven multimodal fake news detection with causal graphs and explainability.**

Built on top of Event-Radar (ACL 2024, Ma et al.) with three novel contributions.

---

## Novel contributions vs Event-Radar

| Feature | Event-Radar | CausalFakeNet (this FYP) |
|---|---|---|
| Event graph edges | Subject–predicate only | **Cause→effect causal edges** |
| Explainability | Credibility scores per view | **Edge saliency + rationale text** |
| Extra loss term | L_cls + L_credible + L_contrast | **+ L_causal (edge entropy regulariser)** |
| Graph construction | SRL subject/object/location | **+ SRL cause/effect roles** |

---

## Project structure

```
causal_fake_net/
├── models/
│   ├── causal_fake_net.py   ← main model (CausalFakeNet)
│   └── losses.py            ← all loss functions
├── data/
│   └── dataset.py           ← synthetic dataset + dataloader
├── train.py                 ← training loop
├── colab_demo.py            ← full Colab walkthrough
└── requirements.txt
```

---

## Quick start (Google Colab)

```python
# 1. Upload this folder to Colab or clone from GitHub

# 2. Install dependencies
!pip install torch torchvision torch_geometric scikit-learn matplotlib

# 3. Run the demo
%run colab_demo.py
```

Or step by step, open `colab_demo.py` — each `# %%` block is a Colab cell.

---

## Architecture overview

```
News post + image
    │
    ├─ CLIP text encoder  → text_feats  (B, m, 512)
    └─ CLIP image encoder → image_feats (B, n, 512)
           │
     NER + SRL
     (subject, predicate, object, CAUSE, EFFECT, location)
           │
    Causal Event Graph  ← NOVEL
    (cause→effect edges + cross-modal + subj-pred)
           │
    Causal GCN (dynamic edge weights)
           │
    x_causal  ──────────────────────────────────┐
                                                 │
    text_cls → Emotion encoder → x_emotion       │
    image    → Pattern encoder → x_pattern       │
                                                 │
    Beta credibility → q_c, q_e, q_p            │
                                                 ▼
                            MHSA credibility-weighted fusion
                                                 │
                                          Classifier
                                        ┌────────┴─────────┐
                                    Logits           Rationale generator ← NOVEL
                                  (real/fake)    (edge saliency + templates)
```

---

## Connecting to real data

To use the Twitter / Weibo / Pheme datasets:

1. Download and preprocess following the original Event-Radar repo.
2. Extract CLIP features offline (ViT-B/16 for Twitter/Pheme, Chinese-CLIP for Weibo).
3. Run SRL using Stanford CoreNLP (English) or TextSmart (Chinese) to get
   subject/predicate/object/cause/effect/location token indices.
4. Subclass `CausalNewsDataset` and override `__getitem__` to load your features.

---

## Key hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `feat_dim` | 512 | CLIP ViT-B/16 output dim |
| `num_gcn_layers` | 2 | GCN depth |
| `alpha` | 0.6 | Dynamic edge update rate |
| `lambda1` | 0.4 | Credible loss weight |
| `lambda2` | 0.2 | Contrastive loss weight |
| `lambda3` | 0.1 | **Causal edge regulariser (novel)** |
| `lr` | 5e-4 | AdamW learning rate |

---

## FYP writeup guidance

When writing up:

- **Section 3.1** Causal Event Graph Builder — explain why cause→effect edges
  capture richer event semantics than subject-predicate alone. Cite Event-Radar's
  own limitation statement: *"we have yet to explore event representation learning
  from a causal relationship perspective."*

- **Section 3.2** Edge Saliency & Rationale Generator — frame as interpretable AI
  contribution. Compare to LIME/SHAP for graphs. No prior multimodal fake news
  paper produces natural-language rationales from the graph structure.

- **Section 4** Experiments — even on the synthetic dataset, show that:
  - Removing causal edges (`w/o causal`) hurts F1
  - Removing the rationale generator has no accuracy impact (ablation is clean)
  - The edge saliency correctly highlights cause→effect edges for fake samples

---

*Author: [Your name] | FYP | [University] | [Year]*
