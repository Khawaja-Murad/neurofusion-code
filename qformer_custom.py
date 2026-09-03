"""
qformer_custom.py — Custom Q-Former with exposed cross-attention weights.

Why this exists: PyTorch's nn.TransformerDecoder doesn't expose attention weights
without monkey-patching. NeuroFusion needs them for the L_ground loss
(IoU between Q-Former cross-attention and GT segmentation). This module
provides a drop-in replacement that:

  1. Exposes per-layer cross-attention weights via a hook list
  2. Returns the LAST layer's cross-attention weights as `last_cross_attn`
  3. Reshapes to [N_lesions, n_queries, d, h, w] for IoU computation
  4. Maintains numerical equivalence with nn.TransformerDecoder when initialized
     identically (same residuals, same FFN expansion, same LayerNorm placement)

Usage in model.py PerLesionQFormer:

    if cfg.use_custom_qformer:
        from qformer_custom import CustomQFormerDecoder
        self.decoder = CustomQFormerDecoder(
            d_model=D, nhead=cfg.qformer_n_heads,
            num_layers=cfg.qformer_n_layers,
            dim_feedforward=D * 4, dropout=0.1, norm_first=True,
        )
    else:
        # original code path: nn.TransformerDecoder

Then in forward, after self.decoder(q, x):
    self._last_attn = self.decoder.last_cross_attn  # [N, n_q, n_kv]

For IoU: reshape last_cross_attn from [N, n_q, n_kv=d*h*w] to [N, n_q, d, h, w]
matching the lesion crop's feature-map dims (passed in as `feat_dhw`).

Smoke test:
    python qformer_custom.py
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-head attention with weight saving
# ---------------------------------------------------------------------------


class MultiHeadAttentionWithWeights(nn.Module):
    """Standard scaled-dot-product MHA but stores attention weights as `.last_attn`.

    Equivalent to nn.MultiheadAttention(batch_first=True) on the forward path,
    but always returns weights so we don't have to dance around `need_weights`.

    Shapes:
        query : [B, Lq, D]
        key   : [B, Lk, D]
        value : [B, Lk, D]
    Returns:
        out:   [B, Lq, D]
        attn:  [B, n_heads, Lq, Lk]   (per-head; aggregate by mean over heads if you want a single map)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} not divisible by nhead={nhead}"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.last_attn: torch.Tensor | None = None  # [B, n_heads, Lq, Lk]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Lq, D = query.shape
        _, Lk, _ = key.shape

        q = self.q_proj(query).view(B, Lq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        # Each: [B, n_heads, L, head_dim]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, Lq, Lk]
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attn = attn.detach() if not self.training else attn

        out = torch.matmul(attn, v)                                  # [B, n_heads, Lq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)        # [B, Lq, D]
        return self.out_proj(out), attn


# ---------------------------------------------------------------------------
# Decoder block (self-attn + cross-attn + FFN)
# ---------------------------------------------------------------------------


class CustomQFormerBlock(nn.Module):
    """One Q-Former block: self-attn + cross-attn + FFN.

    norm_first=True is the conservative choice for small-data fine-tuning
    (more stable gradients than post-norm). Matches Q-Former paper.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttentionWithWeights(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttentionWithWeights(d_model, nhead, dropout=dropout)

        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.last_cross_attn: torch.Tensor | None = None

    def forward(
        self,
        tgt: torch.Tensor,            # queries [B, Lq, D]
        memory: torch.Tensor,         # encoder features [B, Lk, D]
    ) -> torch.Tensor:
        if self.norm_first:
            # Pre-norm: each sub-block normalizes then adds residual
            t = self.norm1(tgt)
            sa_out, _ = self.self_attn(t, t, t)
            tgt = tgt + self.dropout1(sa_out)

            t = self.norm2(tgt)
            ca_out, ca_attn = self.cross_attn(t, memory, memory)
            tgt = tgt + self.dropout2(ca_out)

            t = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.ffn(t))
        else:
            # Post-norm
            sa_out, _ = self.self_attn(tgt, tgt, tgt)
            tgt = self.norm1(tgt + self.dropout1(sa_out))

            ca_out, ca_attn = self.cross_attn(tgt, memory, memory)
            tgt = self.norm2(tgt + self.dropout2(ca_out))

            tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))

        self.last_cross_attn = ca_attn  # [B, n_heads, Lq, Lk]
        return tgt


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------


class CustomQFormerDecoder(nn.Module):
    """Stack of CustomQFormerBlock; exposes per-layer cross-attention weights.

    `forward(tgt, memory)` matches nn.TransformerDecoder's signature, so this
    drops in via a one-line swap in PerLesionQFormer.__init__.

    After forward, the following are available:
        self.last_cross_attn       — last layer's cross-attn, mean over heads:
                                     [B, Lq, Lk]
        self.per_layer_cross_attn  — list of all layers' cross-attn (head-mean):
                                     list of [B, Lq, Lk]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomQFormerBlock(d_model, nhead, dim_feedforward, dropout, norm_first, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

        self.last_cross_attn: torch.Tensor | None = None       # [B, Lq, Lk]
        self.per_layer_cross_attn: list[torch.Tensor] = []     # list of [B, Lq, Lk]

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        out = tgt
        self.per_layer_cross_attn = []

        for layer in self.layers:
            out = layer(out, memory)
            # Mean over heads -> [B, Lq, Lk]
            self.per_layer_cross_attn.append(layer.last_cross_attn.mean(dim=1))

        self.last_cross_attn = self.per_layer_cross_attn[-1]
        return self.norm(out)


# ---------------------------------------------------------------------------
# Reshape attention to feature-grid for IoU
# ---------------------------------------------------------------------------


def reshape_attn_to_grid(
    attn: torch.Tensor,         # [B, Lq, d*h*w]
    feat_dhw: tuple[int, int, int],
) -> torch.Tensor:
    """Reshape flattened cross-attention to [B, Lq, d, h, w] for IoU computation."""
    d, h, w = feat_dhw
    B, Lq, n_kv = attn.shape
    assert n_kv == d * h * w, f"attn n_kv={n_kv} mismatches d*h*w={d*h*w}"
    return attn.view(B, Lq, d, h, w)


def grounding_iou_loss(
    cross_attn: torch.Tensor,        # [N_total, Lq, d, h, w]
    seg_gt_feat: torch.Tensor,        # [N_total, d, h, w]   binary lesion mask at feat resolution
    quantile: float = 0.9,
    eps: float = 1e-6,
    soft: bool = True,
) -> torch.Tensor:
    """L_ground = 1 - mean_q IoU(attn_q, GT_lesion_mask).

    soft=True: use the raw attention values (differentiable through softmax).
                IoU computed with attn as soft mask (sum-based, divisible).
    soft=False: hard threshold at `quantile` per query (NOT differentiable;
                only useful for validation reporting).
    """
    N, Lq, d, h, w = cross_attn.shape
    assert seg_gt_feat.shape == (N, d, h, w)

    gt = seg_gt_feat.float().unsqueeze(1)  # [N, 1, d, h, w]

    if soft:
        # Treat attention as a soft probability mass; normalize per-query
        attn_flat = cross_attn.flatten(2)                                  # [N, Lq, n_kv]
        attn_norm = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + eps)
        attn_grid = attn_norm.view(N, Lq, d, h, w)
        # Soft IoU: intersection = sum(attn * gt), union = sum(attn) + sum(gt) - sum(attn*gt)
        inter = (attn_grid * gt).flatten(2).sum(dim=-1)                    # [N, Lq]
        a_sum = attn_grid.flatten(2).sum(dim=-1)                            # [N, Lq]
        g_sum = gt.flatten(2).sum(dim=-1)                                   # [N, 1]
        union = a_sum + g_sum - inter
        iou = (inter + eps) / (union + eps)
    else:
        # Hard threshold at per-query quantile
        attn_flat = cross_attn.flatten(2)
        thr = attn_flat.quantile(quantile, dim=-1, keepdim=True)
        attn_bin = (attn_flat >= thr).float().view(N, Lq, d, h, w)
        inter = (attn_bin * gt).flatten(2).sum(dim=-1)
        union = (attn_bin + gt - attn_bin * gt).flatten(2).sum(dim=-1)
        iou = (inter + eps) / (union + eps)

    return (1.0 - iou.mean()).clamp_min(0.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== CustomQFormerDecoder smoke test ===")
    B, Lq, Lk, D = 2, 32, 27, 768
    decoder = CustomQFormerDecoder(d_model=D, nhead=12, num_layers=6,
                                    dim_feedforward=D * 4, dropout=0.0)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"params: {n_params:,}")

    queries = torch.randn(B, Lq, D)
    memory = torch.randn(B, Lk, D)
    out = decoder(queries, memory)
    print(f"out shape: {tuple(out.shape)}  (expected ({B}, {Lq}, {D}))")
    print(f"last_cross_attn shape: {tuple(decoder.last_cross_attn.shape)}  (expected ({B}, {Lq}, {Lk}))")
    print(f"per-layer attn count: {len(decoder.per_layer_cross_attn)}  (expected 6)")
    # Each query's attention should sum to 1 over keys
    s = decoder.last_cross_attn.sum(dim=-1)
    print(f"attention rows sum to 1: {torch.allclose(s, torch.ones_like(s), atol=1e-5)}")

    print()
    print("=== grounding_iou_loss smoke test ===")
    feat_dhw = (3, 3, 3)
    Lk = 27
    attn_flat = decoder.last_cross_attn       # [B, Lq, 27]
    attn_grid = reshape_attn_to_grid(attn_flat, feat_dhw)
    print(f"attn_grid shape: {tuple(attn_grid.shape)}  (expected (2, 32, 3, 3, 3))")

    seg_gt_feat = torch.zeros(B, *feat_dhw, dtype=torch.long)
    seg_gt_feat[0, 1:, 1:, 1:] = 1  # lesion in upper octant
    seg_gt_feat[1, :2, :2, :2] = 1
    loss = grounding_iou_loss(attn_grid, seg_gt_feat, soft=True)
    print(f"soft IoU loss: {loss.item():.4f}  (lower is better, range [0, 1])")
    loss_hard = grounding_iou_loss(attn_grid, seg_gt_feat, soft=False)
    print(f"hard IoU loss: {loss_hard.item():.4f}")

    # Backward pass
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())
    print(f"gradients flow through soft IoU: {has_grad}")
    print("\nOK")
