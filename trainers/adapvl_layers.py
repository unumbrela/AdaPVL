"""
AdaPVL layers: Alignment-Adaptive Probabilistic Vision-Language Fusion.

Three innovations:
  1. AAGF  – Alignment-Adaptive Gated Fusion (per-layer directional gates)
  2. CMAS  – Cross-Modal Alignment Scoring (alignment regularization)
  3. MLFA  – Multi-Layer Feature Aggregation (learnable multi-scale aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TwoWayTransformerLayer


class AdaPVL_Adapter(nn.Module):
    """PVL Adapter with learnable per-layer directional gates (AAGF).

    Instead of fixed bidirectional or one-directional fusion, each adapter
    learns two scalar gates that control how much cross-modal information
    flows in each direction:
        h_v += sigmoid(gate_vis) * delta_h_v
        h_t += sigmoid(gate_txt) * delta_h_t

    When gate_vis -> -inf: one-directional (text absorbs vision only).
    When both gates -> +inf: bidirectional (standard MedCLIPSeg).
    """

    def __init__(self, in_channels_vis, in_channels_txt, adapter_channels,
                 beta, gate_init, gate_init_vis=-3.0, gate_init_txt=3.0):
        super().__init__()

        # Down-projection to bottleneck
        self.proj_vis_down = nn.Linear(in_channels_vis, adapter_channels, bias=False)
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

        # Up-projection back to encoder dim
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # Cross-modal interaction (unchanged from MedCLIPSeg)
        self.two_way = TwoWayTransformerLayer(adapter_channels, beta, gate_init)

        # === AAGF: learnable directional gates ===
        # Vision gate: initialized conservatively closed (sigmoid(-3) ≈ 0.047).
        # This prevents spatial feature corruption during early training for
        # non-aligned encoders. Pre-aligned encoders will learn to open it.
        # Text gate: initialized open (sigmoid(+3) ≈ 0.953).
        # Text always benefits from absorbing vision context.
        self.gate_vis = nn.Parameter(torch.tensor(gate_init_vis))
        self.gate_txt = nn.Parameter(torch.tensor(gate_init_txt))

    def forward(self, vis, text):
        """
        Args:
            vis:  [B, N_v, D_v] vision tokens
            text: [B, N_t, D_t] text tokens
        Returns:
            vis_out:     [B, N_v, D_v] gated vision update
            txt_out:     [B, N_t, D_t] gated text update
            align_score: scalar, cosine similarity in bottleneck space
        """
        # Down-project to shared bottleneck space
        v = self.proj_vis_down(vis)   # [B, N_v, d_a]
        t = self.proj_txt_down(text)  # [B, N_t, d_a]

        # === CMAS: compute alignment score in bottleneck space ===
        # Measure raw alignment BEFORE cross-attention
        v_mean = v.mean(dim=1)  # [B, d_a]
        t_mean = t.mean(dim=1)  # [B, d_a]
        align_score = F.cosine_similarity(v_mean, t_mean, dim=-1).mean()

        # Bidirectional cross-attention (same as original PVL)
        v_fused, t_fused = self.two_way(v, t)

        # Up-project
        vis_raw = self.proj_vis_up(v_fused)
        txt_raw = self.proj_txt_up(t_fused)

        # === AAGF: apply directional gates ===
        alpha_v = torch.sigmoid(self.gate_vis)
        alpha_t = torch.sigmoid(self.gate_txt)

        vis_out = alpha_v * vis_raw
        txt_out = alpha_t * txt_raw

        return vis_out, txt_out, align_score


class MultiLayerAggregator(nn.Module):
    """MLFA: Multi-Layer Feature Aggregation.

    Collects intermediate vision features from selected fusion layers,
    projects them to segmentation dimension, and combines them with
    learnable importance weights.
    """

    def __init__(self, in_dim, out_dim, num_collect_layers):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_collect_layers)
        ])
        # Learnable importance logits (initialized to 0 = uniform)
        self.importance_logits = nn.Parameter(torch.zeros(num_collect_layers))

    def forward(self, intermediate_features, final_features):
        """
        Args:
            intermediate_features: list of [B, N, in_dim], length = num_collect_layers
            final_features: [B, N, out_dim] — features from the last encoder layer
        Returns:
            aggregated: [B, N, out_dim]
        """
        weights = F.softmax(self.importance_logits, dim=0)

        aggregated = torch.zeros_like(final_features)
        for w, proj, feat in zip(weights, self.projections, intermediate_features):
            # Handle potential sequence length mismatch (different prefix tokens)
            if feat.size(1) != final_features.size(1):
                feat = feat[:, :final_features.size(1), :]
            aggregated = aggregated + w * proj(feat)

        return final_features + aggregated


def compute_alignment_loss(adapters):
    """CMAS regularization loss.

    Penalizes vision gate for exceeding the measured alignment score:
        L_align = (1/K) * sum_k max(0, sigmoid(g_v^k) - relu(a^k))

    This provides inductive bias: don't open the vision gate wider than
    the actual cross-modal alignment supports.
    """
    loss = 0.0
    count = 0
    for adapter in adapters:
        alpha_v = torch.sigmoid(adapter.gate_vis)
        # align_score is computed during forward pass and cached
        if not hasattr(adapter, "_cached_align_score"):
            continue
        align = torch.relu(adapter._cached_align_score)
        loss = loss + torch.relu(alpha_v - align)
        count += 1
    return loss / max(count, 1)
