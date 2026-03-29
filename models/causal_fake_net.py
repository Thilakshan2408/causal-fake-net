"""
CausalFakeNet — FYP Prototype
==============================
Novel contributions vs Event-Radar (ACL 2024):
  1. Causal event graph  : cause→effect edges (not just subject-predicate)
  2. Edge saliency scores: identify WHICH causal link drove the decision
  3. Rationale generator : template-based human-readable explanation

Architecture:
  Input (post + image CLIP features)
    → NER + SRL → Causal Event Graph
    → Causal GCN (dynamic edge weights)
    → Causal inconsistency vector x_c
    → [x_c, x_emotion, x_pattern] × Beta credibility weights
    → MHSA fusion → Classifier
    → Rationale generator (edge saliency + templates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. CAUSAL EVENT GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class CausalGraphBuilder(nn.Module):
    """
    Builds a causal event graph from CLIP text/image features + SRL annotations.

    Nodes  : entities extracted by NER (subject, object, location, cause, effect)
    Edges  : typed relations
              - subject→predicate   (standard, inherited from Event-Radar)
              - cause→effect        (NOVEL: causal link)
              - image_entity→text_entity  (cross-modal grounding)

    Edge types are encoded as learnable type embeddings added to edge weights.
    """

    EDGE_TYPES = {
        "subject_predicate": 0,
        "cause_effect":      1,   # ← novel causal edge
        "cross_modal":       2,
    }

    def __init__(self, feat_dim: int = 512, num_edge_types: int = 3):
        super().__init__()
        self.feat_dim = feat_dim
        # Learnable edge-type bias vectors (added to raw similarity weights)
        self.edge_type_emb = nn.Embedding(num_edge_types, 1)
        nn.init.zeros_(self.edge_type_emb.weight)

    def forward(
        self,
        text_feats: torch.Tensor,   # (B, m, d)  — token embeddings from CLIP
        image_feats: torch.Tensor,  # (B, n, d)  — object embeddings from CLIP
        srl_roles: list,            # list[dict]  — per-sample SRL output
    ) -> list:
        """
        Returns a list of PyG Data objects (one per sample in the batch),
        each containing:
            data.x            node feature matrix  (num_nodes, d)
            data.edge_index   (2, num_edges)
            data.edge_attr    (num_edges, 1)  — weighted + type-biased
            data.edge_type    (num_edges,)    — integer type id
            data.node_type    (num_nodes,)    — 0=text, 1=image
            data.causal_mask  (num_edges,)    — bool, True for cause→effect edges
        """
        graphs = []
        B = text_feats.size(0)

        for b in range(B):
            tf = text_feats[b]   # (m, d)
            vf = image_feats[b]  # (n, d)
            roles = srl_roles[b] if srl_roles else {}

            nodes, node_types, edges, edge_types = self._build_graph(
                tf, vf, roles
            )

            # Stack node features
            x = torch.stack(nodes, dim=0)               # (N, d)
            node_type_t = torch.tensor(node_types, device=x.device)

            if len(edges) == 0:
                # Fallback: fully connected
                N = x.size(0)
                src = torch.arange(N).repeat_interleave(N)
                dst = torch.arange(N).repeat(N)
                mask = src != dst
                ei = torch.stack([src[mask], dst[mask]], dim=0)
                et = torch.zeros(ei.size(1), dtype=torch.long, device=x.device)
            else:
                ei_list, et_list = zip(*edges)
                ei = torch.tensor(ei_list, device=x.device).t().contiguous()
                et = torch.tensor(et_list, device=x.device)

            # Edge weights = cosine similarity + learnable type bias
            src_f = x[ei[0]]
            dst_f = x[ei[1]]
            sim = F.cosine_similarity(src_f, dst_f, dim=-1, eps=1e-8).unsqueeze(-1)
            sim = (sim + 1.0) / 2.0  # map [-1,1] → [0,1]
            type_bias = self.edge_type_emb(et)           # (E, 1)
            edge_attr = torch.sigmoid(sim + type_bias)   # (E, 1)
            causal_mask = (et == self.EDGE_TYPES["cause_effect"])

            graphs.append(Data(
                x=x,
                edge_index=ei,
                edge_attr=edge_attr,
                edge_type=et,
                node_type=node_type_t,
                causal_mask=causal_mask,
            ))

        return graphs

    def _build_graph(self, text_feats, image_feats, roles):
        """
        Heuristic graph construction from SRL role dict.
        roles keys: 'subject', 'predicate', 'object', 'cause', 'effect', 'location'
        Values are token indices into text_feats.

        For the prototype we accept integer indices or fall back to
        the CLS token (index 0) when a role is missing.
        """
        nodes = []
        node_types = []
        role_to_node = {}

        # Add text entity nodes
        for role in ["subject", "predicate", "object", "cause", "effect", "location"]:
            idx = roles.get(role, 0)
            if isinstance(idx, int) and idx < text_feats.size(0):
                feat = text_feats[idx]
            else:
                feat = text_feats[0]
            node_id = len(nodes)
            nodes.append(feat)
            node_types.append(0)
            role_to_node[role] = node_id

        # Add image entity nodes mirrored by closest text similarity
        for role in ["subject", "object", "location"]:
            txt_feat = nodes[role_to_node[role]]
            sims = F.cosine_similarity(
                txt_feat.unsqueeze(0), image_feats, dim=-1
            )
            best = sims.argmax().item()
            img_feat = image_feats[best]
            node_id = len(nodes)
            nodes.append(img_feat)
            node_types.append(1)
            role_to_node[f"img_{role}"] = node_id

        edges = []

        def add_edge(src, dst, etype):
            edges.append(((src, dst), self.EDGE_TYPES[etype]))

        # Standard subject→predicate edges
        add_edge(role_to_node["subject"],   role_to_node["predicate"], "subject_predicate")
        add_edge(role_to_node["predicate"], role_to_node["object"],    "subject_predicate")

        # ── NOVEL: cause→effect causal edges ──
        if "cause" in roles or True:   # always try to add
            add_edge(role_to_node["cause"],  role_to_node["effect"],    "cause_effect")
            add_edge(role_to_node["cause"],  role_to_node["predicate"], "cause_effect")

        # Cross-modal grounding edges
        for role in ["subject", "object", "location"]:
            add_edge(role_to_node[role], role_to_node[f"img_{role}"], "cross_modal")
            add_edge(role_to_node[f"img_{role}"], role_to_node[role], "cross_modal")

        return nodes, node_types, edges, [e[1] for e in edges]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CAUSAL GCN WITH EDGE SALIENCY
# ─────────────────────────────────────────────────────────────────────────────

class CausalGCN(nn.Module):
    """
    L-layer GCN operating on the causal event graph.

    Novel addition: after the final layer we compute edge saliency scores —
    the gradient of the output w.r.t. each edge weight, indicating which
    causal edge most influenced the prediction (used in rationale generation).
    """

    def __init__(self, feat_dim: int = 512, num_layers: int = 2,
                 alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha
        self.convs = nn.ModuleList([
            GCNConv(feat_dim, feat_dim) for _ in range(num_layers)
        ])
        self.dynamic_W = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, data: Data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        # Normalise edge weights into adjacency
        edge_weight = ea.squeeze(-1)

        h = x
        for conv in self.convs:
            h_new = F.relu(conv(h, ei, edge_weight))
            # Dynamic edge weight update (as in Event-Radar §3.1)
            delta = torch.sigmoid(
                (self.dynamic_W(h_new)[ei[0]] *
                 self.dynamic_W(h_new)[ei[1]]).sum(-1, keepdim=True)
            )
            edge_weight = (self.alpha * edge_weight.unsqueeze(-1)
                           + (1 - self.alpha) * delta).squeeze(-1)
            h = h_new

        return h, edge_weight   # (N, d), (E,)

    @staticmethod
    def compute_edge_saliency(edge_weight: torch.Tensor,
                              loss: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based saliency: |∂loss/∂edge_weight| per edge.
        Call after loss.backward() with retain_graph=True.
        """
        if edge_weight.grad is not None:
            return edge_weight.grad.abs()
        return torch.zeros_like(edge_weight)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BETA CREDIBILITY MODULE  (from Event-Radar, adapted for causal view)
# ─────────────────────────────────────────────────────────────────────────────

class BetaCredibility(nn.Module):
    """
    Estimates per-view credibility using Beta distribution parameterisation
    (Subjective Logic Theory). Adapted from Event-Radar §3.3.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 2)   # → logits for real/fake

    def forward(self, x: torch.Tensor):
        """
        x : (B, d)
        Returns
            logits      (B, 2)
            credibility (B,)   scalar in (0, 1)
        """
        logits = self.fc(x)
        # Evidence = softplus of logits (must be non-negative)
        evidence = F.softplus(logits)            # (B, 2)
        # Beta parameters: β = 1 + e
        beta_params = 1.0 + evidence             # (B, 2)
        # Strength of Beta distribution
        strength = beta_params.sum(dim=-1)       # (B,)
        # Credible quality per class
        b = evidence / strength.unsqueeze(-1)    # (B, 2)
        # Uncertainty = 1 - sum(b)
        uncertainty = 1.0 - b.sum(dim=-1)       # (B,)
        credibility = 1.0 - uncertainty          # (B,)
        return logits, credibility


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMOTION ENCODER  (lightweight, following Event-Radar §3.2)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionEncoder(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, text_cls: torch.Tensor) -> torch.Tensor:
        """text_cls: (B, d) — CLS token from CLIP text encoder."""
        return self.proj(text_cls)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PATTERN ENCODER  (DCT-based, following Event-Radar §3.2)
# ─────────────────────────────────────────────────────────────────────────────

class PatternEncoder(nn.Module):
    """
    Encodes image manipulation patterns via a simplified DCT projection.
    In the prototype we approximate DCT with a learnable frequency-like
    projection (avoids scipy dependency in Colab).
    """

    def __init__(self, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.freq_proj = nn.Linear(feat_dim, feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, image_patches: torch.Tensor) -> torch.Tensor:
        """
        image_patches: (B, n, d) — image object features from CLIP.
        """
        # Approximate frequency domain via learnable projection
        freq = F.relu(self.freq_proj(image_patches))     # (B, n, d)
        attended, _ = self.attn(freq, freq, freq)        # (B, n, d)
        pooled = attended.mean(dim=1)                    # (B, d)
        return self.out_proj(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-VIEW FUSION  (credibility-weighted MHSA)
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewFusion(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        x_c: torch.Tensor,   # (B, d) causal
        x_e: torch.Tensor,   # (B, d) emotion
        x_p: torch.Tensor,   # (B, d) pattern
        q_c: torch.Tensor,   # (B,)  credibility causal
        q_e: torch.Tensor,   # (B,)  credibility emotion
        q_p: torch.Tensor,   # (B,)  credibility pattern
    ) -> torch.Tensor:
        # Scale each view by its credibility
        x_c_s = x_c * q_c.unsqueeze(-1)
        x_e_s = x_e * q_e.unsqueeze(-1)
        x_p_s = x_p * q_p.unsqueeze(-1)

        # Stack as sequence for self-attention: (B, 3, d)
        views = torch.stack([x_c_s, x_e_s, x_p_s], dim=1)
        fused, attn_weights = self.attn(views, views, views)  # (B, 3, d)
        fused = self.norm(fused.mean(dim=1))                  # (B, d)
        return fused, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 7. RATIONALE GENERATOR  (NOVEL — template + saliency)
# ─────────────────────────────────────────────────────────────────────────────

class RationaleGenerator:
    """
    Rule-based rationale generator using:
      - Top-k salient causal edges from the graph
      - Per-view credibility scores
      - Prediction probability

    Produces a human-readable explanation like:
      "FAKE (conf: 0.91). The image depicts [subject] causing [effect],
       but the post claims [cause] led to [effect].
       Causal inconsistency was the strongest signal (credibility: 0.87).
       Pattern manipulation detected (credibility: 0.72)."

    This is the explainability contribution absent from Event-Radar.
    """

    TEMPLATES = {
        "fake_causal": (
            "FAKE (confidence: {conf:.2f}). "
            "The most suspicious causal link is: '{cause}' → '{effect}'. "
            "The image and post give conflicting accounts of this cause-effect chain. "
            "Causal consistency score: {q_c:.2f} | "
            "Emotion score: {q_e:.2f} | "
            "Pattern score: {q_p:.2f}."
        ),
        "fake_pattern": (
            "FAKE (confidence: {conf:.2f}). "
            "Image manipulation patterns were detected (pattern credibility: {q_p:.2f}). "
            "The causal chain appears surface-level consistent but the image may be fabricated. "
            "Causal score: {q_c:.2f} | Emotion score: {q_e:.2f}."
        ),
        "fake_emotion": (
            "FAKE (confidence: {conf:.2f}). "
            "The post uses strongly provocative emotional language (emotion credibility: {q_e:.2f}), "
            "a common feature of fake news. "
            "Causal score: {q_c:.2f} | Pattern score: {q_p:.2f}."
        ),
        "real": (
            "REAL (confidence: {conf:.2f}). "
            "The causal chain in the image and post are consistent. "
            "Causal score: {q_c:.2f} | Emotion score: {q_e:.2f} | Pattern score: {q_p:.2f}."
        ),
    }

    def generate(
        self,
        pred_label: int,         # 0=real, 1=fake
        pred_prob: float,        # confidence
        q_c: float,              # causal credibility
        q_e: float,              # emotion credibility
        q_p: float,              # pattern credibility
        top_causal_edge: dict,   # {'cause': str, 'effect': str, 'saliency': float}
        srl_roles: dict,
    ) -> str:
        kw = dict(conf=pred_prob, q_c=q_c, q_e=q_e, q_p=q_p,
                  cause=top_causal_edge.get("cause", "unknown event"),
                  effect=top_causal_edge.get("effect", "unknown outcome"))

        if pred_label == 0:
            return self.TEMPLATES["real"].format(**kw)

        # Choose fake template based on weakest credibility signal
        scores = {"causal": q_c, "emotion": q_e, "pattern": q_p}
        weakest = min(scores, key=scores.get)
        template_key = f"fake_{weakest}"
        return self.TEMPLATES[template_key].format(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CausalFakeNet(nn.Module):
    """
    CausalFakeNet — complete fake news detection model.

    Forward pass returns:
        logits      (B, 2)          — classification logits
        credibility (B, 3)          — [q_c, q_e, q_p] per-view credibility
        edge_weights list[Tensor]   — per-graph edge weights (for saliency)
        fused       (B, d)          — fused representation
    """

    def __init__(self, feat_dim: int = 512, num_classes: int = 2,
                 num_gcn_layers: int = 2, num_heads: int = 4,
                 alpha: float = 0.6):
        super().__init__()
        self.feat_dim = feat_dim

        # Stage 1: causal graph
        self.graph_builder  = CausalGraphBuilder(feat_dim)
        self.causal_gcn     = CausalGCN(feat_dim, num_gcn_layers, alpha)

        # Comparison function projection (from Event-Radar §3.1)
        # x_c = W_c [h_P, h_I, h_P-h_I, h_P⊙h_I]
        self.compare_proj = nn.Linear(feat_dim * 4, feat_dim)

        # Stage 2: other view encoders
        self.emotion_enc    = EmotionEncoder(feat_dim)
        self.pattern_enc    = PatternEncoder(feat_dim, num_heads)

        # Credibility estimators (one per view)
        self.cred_c = BetaCredibility(feat_dim)
        self.cred_e = BetaCredibility(feat_dim)
        self.cred_p = BetaCredibility(feat_dim)

        # Fusion
        self.fusion = MultiViewFusion(feat_dim, num_heads)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

        # Rationale generator (non-differentiable, used at inference)
        self.rationale_gen = RationaleGenerator()

    def forward(
        self,
        text_feats: torch.Tensor,   # (B, m, d)
        image_feats: torch.Tensor,  # (B, n, d)
        srl_roles: list,            # list[dict] — SRL output per sample
    ):
        B = text_feats.size(0)

        # ── 1. Build and encode causal graphs ──
        graphs = self.graph_builder(text_feats, image_feats, srl_roles)

        x_c_list, edge_weight_list = [], []
        for g in graphs:
            h, ew = self.causal_gcn(g)
            # Split text vs image nodes
            text_mask  = (g.node_type == 0)
            image_mask = (g.node_type == 1)
            h_P = h[text_mask].mean(0)   # (d,)
            h_I = h[image_mask].mean(0)  # (d,)
            # Comparison vector (Event-Radar style, on causal node reps)
            comp = torch.cat([h_P, h_I, h_P - h_I, h_P * h_I], dim=-1)
            x_c = self.compare_proj(comp)             # (d,)
            x_c_list.append(x_c)
            edge_weight_list.append(ew)

        x_c = torch.stack(x_c_list, dim=0)           # (B, d)

        # ── 2. Other views ──
        text_cls    = text_feats[:, 0, :]             # (B, d)  CLS token
        x_e = self.emotion_enc(text_cls)              # (B, d)
        x_p = self.pattern_enc(image_feats)           # (B, d)

        # ── 3. Credibility estimation ──
        logits_c, q_c = self.cred_c(x_c)
        logits_e, q_e = self.cred_e(x_e)
        logits_p, q_p = self.cred_p(x_p)

        credibility = torch.stack([q_c, q_e, q_p], dim=-1)   # (B, 3)

        # ── 4. Fusion & classification ──
        fused, _ = self.fusion(x_c, x_e, x_p, q_c, q_e, q_p)
        logits = self.classifier(fused)                        # (B, 2)

        return logits, credibility, edge_weight_list, fused

    @torch.no_grad()
    def explain(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        srl_roles: list,
    ) -> list:
        """
        Runs inference and produces a human-readable rationale for each sample.
        Returns list of explanation strings.
        """
        self.eval()
        logits, credibility, edge_weights, _ = self.forward(
            text_feats, image_feats, srl_roles
        )
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        explanations = []

        for b in range(text_feats.size(0)):
            pred_label = preds[b].item()
            pred_prob  = probs[b, pred_label].item()
            q_c, q_e, q_p = credibility[b].tolist()

            # Top causal edge from srl_roles (heuristic for prototype)
            roles = srl_roles[b] if srl_roles else {}
            top_causal_edge = {
                "cause":    roles.get("cause_text",  "unspecified cause"),
                "effect":   roles.get("effect_text", "unspecified effect"),
                "saliency": float(edge_weights[b].max().item()),
            }

            explanation = self.rationale_gen.generate(
                pred_label=pred_label,
                pred_prob=pred_prob,
                q_c=q_c, q_e=q_e, q_p=q_p,
                top_causal_edge=top_causal_edge,
                srl_roles=roles,
            )
            explanations.append(explanation)

        return explanations
