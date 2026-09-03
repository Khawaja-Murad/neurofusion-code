"""
mrope_4d.py — 4-axis Multimodal Rotary Position Embedding for NeuroFusion v4.

Retrofits MedGemma 1.5 4B's 1D-RoPE attention with 4-axis rotary encoding over
(t, d, h, w) = (modality, depth, in-plane H, in-plane W).

Design:
  - head_dim partitioned into 4 contiguous chunks, each rotated by its own axis position
  - text tokens use (p, p, p, p) => reduces exactly to 1D RoPE (identity reduction)
  - this means on a text-only batch, output is numerically identical to pretrained MedGemma
  - visual tokens carry distinct (t, d, h, w) from voxel coordinates

Implementation notes:
  - this file provides the position-tensor builder + the rotary application function
  - to plug into MedGemma, monkey-patch `apply_rotary_pos_emb` in the attention class
    with `apply_mrope_4d` below
  - inspired by Qwen2-VL M-RoPE (arXiv:2409.12191) + VideoRoPE (arXiv:2502.05173)
  - identity-reduction guarantee is tested in `_test_identity_reduction()`

Channel partition (default for MedGemma 4B head_dim=256):
  channels  0-63   ( 32 pairs): rotary half, axis t (modality)
  channels 64-127  ( 32 pairs): rotary half, axis d (slice depth)
  channels 128-191 ( 32 pairs): rotary half, axis h (in-plane H)
  channels 192-255 ( 32 pairs): rotary half, axis w (in-plane W)

Wait -- that's 128 pairs = 256 rotary channels total. We want HALF of head_dim rotary,
matching standard RoPE practice. Correct partition for head_dim=256, rotary_dim=128:
  channels  0-31   (16 pairs): axis t
  channels 32-63   (16 pairs): axis d
  channels 64-95   (16 pairs): axis h
  channels 96-127  (16 pairs): axis w
  channels 128-255:            non-rotary (unchanged from base)

We expose the partition as a config field for flexibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class MRope4DConfig:
    """Configuration for 4-axis Multimodal RoPE.

    ``head_dim``: attention head dimension of the target LM (256 for MedGemma 4B).
    ``rotary_dim``: how many channels out of head_dim to rotate (typically head_dim or head_dim // 2).
    ``axis_channels``: (n_t, n_d, n_h, n_w) pairs per axis; must sum to rotary_dim // 2.
    ``rotary_base``: theta base (standard 10000.0 matches pretrained MedGemma).
    ``low_freq_axis``: axis to place on lowest-frequency channels (VideoRoPE LTA refinement).
    """

    head_dim: int = 256
    rotary_dim: int = 128                     # half of head_dim
    axis_channels: tuple[int, int, int, int] = (16, 16, 16, 16)  # pairs per axis (t, d, h, w)
    rotary_base: float = 10000.0
    # LTA (Low-Frequency Temporal Allocation, VideoRoPE arXiv:2502.05173): move one axis
    # to the lowest-frequency channels. Breaks exact identity reduction to 1D RoPE but can
    # improve training stability at long contexts. Default OFF to preserve identity guarantee.
    low_freq_axis: Literal["t", "d", "h", "w", "none"] = "none"
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        total_pairs = self.rotary_dim // 2
        assert sum(self.axis_channels) == total_pairs, (
            f"axis_channels must sum to rotary_dim // 2 = {total_pairs}, "
            f"got {sum(self.axis_channels)}"
        )
        assert self.rotary_dim <= self.head_dim, "rotary_dim cannot exceed head_dim"


# --------------------------------------------------------------------------
# Position tensor builder
# --------------------------------------------------------------------------


def build_position_ids(
    seq_len: int,
    token_kinds: torch.Tensor,        # [seq_len], dtype long: 0=text, 1=visual
    modality_ids: torch.Tensor,       # [seq_len], long: 0..3 for T1/T1CE/T2/FLAIR, -1 for text
    voxel_coords: torch.Tensor,       # [seq_len, 3], long: (d, h, w) per visual token; (0,0,0) for text
    device: torch.device,
) -> torch.Tensor:
    """Build 4-axis position tensor shape [4, seq_len] per the M-RoPE convention.

    Returns:
        pos: [4, seq_len] long tensor with rows (t, d, h, w).
             For text tokens (kind=0): t=d=h=w=p where p is cumulative text position index.
             For visual tokens (kind=1): t=modality_id, (d,h,w)=voxel_coords.
    """
    out = torch.zeros((4, seq_len), dtype=torch.long, device=device)

    is_text = (token_kinds == 0)
    # text tokens: assign cumulative position p (among text tokens) to all four rows
    text_positions = torch.cumsum(is_text.long(), dim=0) - 1
    # replace negatives (before first text token) with 0; they'll be overwritten for visual tokens anyway
    text_positions = text_positions.clamp(min=0)

    for axis in range(4):
        out[axis] = torch.where(
            is_text,
            text_positions,
            # visual tokens
            modality_ids if axis == 0 else voxel_coords[:, axis - 1],
        )
    return out


# --------------------------------------------------------------------------
# Rotary angle table
# --------------------------------------------------------------------------


class MRope4DAngleTable(nn.Module):
    """Precomputed inv_freq + on-the-fly cos/sin for 4-axis M-RoPE.

    The inv_freq table is shared across axes (standard RoPE practice) but applied to different
    channel slices per axis. Optional Low-Frequency Temporal Allocation (LTA) permutes the
    channel ordering so the slowest-varying axis receives the lowest-frequency pairs.

    Construct via:
      - ``MRope4DAngleTable(cfg)`` — derives ``inv_freq`` from ``cfg.rotary_base``. Use only
        for standalone math tests; does NOT match HF's per-layer-type rope when retrofitting
        Gemma 3 / MedGemma.
      - ``MRope4DAngleTable.from_hf_rotary_emb(...)`` — copies ``inv_freq`` directly from one
        of HF's ``Gemma3RotaryEmbedding`` modules so identity-mode 4-axis reduction is
        bit-equivalent to that layer's 1D RoPE (per-layer base + scaling inherited).
    """

    def __init__(self, cfg: MRope4DConfig, *, inv_freq: torch.Tensor | None = None,
                 attention_scaling: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.attention_scaling = float(attention_scaling)

        n_pairs = cfg.rotary_dim // 2

        if inv_freq is None:
            # Standard RoPE inverse frequencies, one per pair (scalar i indexes pair)
            freqs = torch.arange(0, n_pairs, dtype=torch.float32)
            inv_freq = 1.0 / (cfg.rotary_base ** (2 * freqs / cfg.rotary_dim))
        else:
            # Trust caller's inv_freq (e.g. lifted from HF rotary_emb); validate shape only.
            inv_freq = inv_freq.to(torch.float32).flatten()
            if inv_freq.shape[0] != n_pairs:
                raise ValueError(
                    f"inv_freq must have length {n_pairs} (=rotary_dim/2={cfg.rotary_dim}/2); "
                    f"got {inv_freq.shape[0]}"
                )

        # Channel slice per axis: cumulative partition over pairs
        offsets = [0]
        for nc in cfg.axis_channels:
            offsets.append(offsets[-1] + nc)
        self.axis_slice = {
            "t": slice(offsets[0], offsets[1]),
            "d": slice(offsets[1], offsets[2]),
            "h": slice(offsets[2], offsets[3]),
            "w": slice(offsets[3], offsets[4]),
        }

        # LTA: permute inv_freq so low_freq_axis sees the smallest frequencies.
        # Skip if 'none' so that identity reduction is preserved.
        perm = list(range(n_pairs))
        axis_order = ["t", "d", "h", "w"]
        if cfg.low_freq_axis != "none" and cfg.low_freq_axis in axis_order:
            # Move low_freq_axis slice to the tail of pair ordering
            low_slice = self.axis_slice[cfg.low_freq_axis]
            head = [i for i in perm if not (low_slice.start <= i < low_slice.stop)]
            tail = list(range(low_slice.start, low_slice.stop))
            perm = head + tail
            # Reassign slices because positions changed
            offsets = [0]
            for axis in axis_order:
                if axis == cfg.low_freq_axis:
                    continue
                offsets.append(offsets[-1] + cfg.axis_channels[axis_order.index(axis)])
            # tail axis gets remaining
            final_axis_slice: dict[str, slice] = {}
            non_low = [a for a in axis_order if a != cfg.low_freq_axis]
            for axis, off_hi in zip(non_low, offsets[1:]):
                off_lo = offsets[offsets.index(off_hi) - 1]
                final_axis_slice[axis] = slice(off_lo, off_hi)
            final_axis_slice[cfg.low_freq_axis] = slice(offsets[-1], n_pairs)
            self.axis_slice = final_axis_slice

        inv_freq_perm = inv_freq[perm]
        self.register_buffer("inv_freq", inv_freq_perm, persistent=False)

    @classmethod
    def from_hf_rotary_emb(
        cls,
        inv_freq: torch.Tensor,                  # length head_dim // 2; already per-layer-scaled
        attention_scaling: float,                # HF "attention_factor"; 1.0 for default/linear
        head_dim: int,
        axis_channels: tuple[int, int, int, int],
        low_freq_axis: Literal["t", "d", "h", "w", "none"] = "none",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MRope4DAngleTable":
        """Build a table whose cos/sin reproduces HF's 1D RoPE for that layer when fed
        identity-mode positions (all 4 axes equal). Uses HF's pre-scaled inv_freq directly,
        so per-layer rope_theta and any rope_scaling (linear/yarn/etc.) carry through."""
        rotary_dim = head_dim                      # full rotary, matching HF
        if sum(axis_channels) != rotary_dim // 2:
            raise ValueError(
                f"axis_channels must sum to rotary_dim/2 = head_dim/2 = {rotary_dim // 2}; "
                f"got {sum(axis_channels)} from {axis_channels}"
            )
        cfg = MRope4DConfig(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            axis_channels=axis_channels,
            rotary_base=float("nan"),              # unused on this path
            low_freq_axis=low_freq_axis,
            dtype=dtype,
        )
        return cls(cfg, inv_freq=inv_freq, attention_scaling=attention_scaling)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos, sin tables of shape [seq_len, rotary_dim].

        Args:
            position_ids: [4, seq_len] long tensor with axis positions.
        Returns:
            cos, sin: [seq_len, rotary_dim] each.
        """
        assert position_ids.shape[0] == 4, "position_ids must have 4 axis rows"
        seq_len = position_ids.shape[1]
        n_pairs = self.cfg.rotary_dim // 2
        device = position_ids.device

        freqs = torch.zeros((seq_len, n_pairs), dtype=self.inv_freq.dtype, device=device)
        axis_map = {"t": 0, "d": 1, "h": 2, "w": 3}
        for axis_name, ax_idx in axis_map.items():
            sl = self.axis_slice[axis_name]
            pos = position_ids[ax_idx].float()                   # [seq_len]
            freqs[:, sl] = pos[:, None] * self.inv_freq[sl][None, :]

        # Expand per-pair freqs to per-channel: interleave (f, f)
        emb = torch.cat([freqs, freqs], dim=-1)                   # [seq_len, rotary_dim]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


# --------------------------------------------------------------------------
# Rotary apply function (monkey-patch target)
# --------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate second half of the last dim by 180 degrees (RoPE trick)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope_4d(
    q: torch.Tensor,                 # [batch, n_heads, seq_len, head_dim]
    k: torch.Tensor,                 # [batch, n_kv_heads, seq_len, head_dim]
    cos: torch.Tensor,               # [seq_len, rotary_dim]
    sin: torch.Tensor,               # [seq_len, rotary_dim]
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 4-axis M-RoPE rotation to q and k. Only the first ``rotary_dim`` channels rotate."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_rotated = q_rot * cos + _rotate_half(q_rot) * sin
    k_rotated = k_rot * cos + _rotate_half(k_rot) * sin

    q_out = torch.cat([q_rotated, q_pass], dim=-1)
    k_out = torch.cat([k_rotated, k_pass], dim=-1)
    return q_out, k_out


# --------------------------------------------------------------------------
# Identity-reduction test (the core correctness guarantee)
# --------------------------------------------------------------------------


def _test_identity_reduction() -> None:
    """When all 4 axis positions share the same value p, M-RoPE must reduce to 1D RoPE.

    This is the guarantee that lets us retrofit MedGemma without breaking pretrained weights.
    """
    import torch

    cfg = MRope4DConfig(head_dim=64, rotary_dim=32, axis_channels=(4, 4, 4, 4))
    table = MRope4DAngleTable(cfg)

    seq_len = 8
    batch = 2
    n_heads = 4

    # Build position tensor with all 4 axes = p (text-token-like)
    p = torch.arange(seq_len)
    pos_mrope = torch.stack([p, p, p, p], dim=0)           # [4, seq_len]
    cos_m, sin_m = table(pos_mrope)

    # Reference 1D RoPE
    inv_freq_ref = 1.0 / (cfg.rotary_base ** (torch.arange(0, cfg.rotary_dim, 2).float() / cfg.rotary_dim))
    freqs_ref = p.float()[:, None] * inv_freq_ref[None, :]
    emb_ref = torch.cat([freqs_ref, freqs_ref], dim=-1)
    cos_r, sin_r = emb_ref.cos(), emb_ref.sin()

    q = torch.randn(batch, n_heads, seq_len, cfg.head_dim)
    k = torch.randn(batch, n_heads, seq_len, cfg.head_dim)

    # Apply both
    q_m, k_m = apply_mrope_4d(q, k, cos_m, sin_m, cfg.rotary_dim)
    q_r, k_r = apply_mrope_4d(q, k, cos_r, sin_r, cfg.rotary_dim)  # same function, different freqs

    # They should be equal up to channel permutation induced by LTA. With low_freq_axis='d',
    # the inv_freq ordering in M-RoPE is [t, h, w, d] instead of naive [t, d, h, w].
    # For identity reduction test, we want cos_m == cos_r up to permutation - but since all
    # axis positions are equal, the permutation doesn't matter for the final result.
    # Check: results should be numerically equal.
    max_diff_q = (q_m - q_r).abs().max().item()
    max_diff_k = (k_m - k_r).abs().max().item()
    assert max_diff_q < 1e-4, f"identity reduction FAILED on q: max diff {max_diff_q}"
    assert max_diff_k < 1e-4, f"identity reduction FAILED on k: max diff {max_diff_k}"
    print(f"[ok] identity reduction test passed (max diff q={max_diff_q:.2e}, k={max_diff_k:.2e})")


def _test_identity_reduction_hf_path() -> None:
    """Identity reduction when inv_freq comes from HF's per-layer rotary_emb.

    Simulates both Gemma 3 layer types:
      - sliding_attention: base 10000, default scaling, attention_factor=1.0
      - full_attention:    base 1000000 with linear factor 8.0, attention_factor=1.0

    For each, builds an inv_freq the way HF's _compute_*_rope_parameters would, constructs
    an MRope4DAngleTable via from_hf_rotary_emb, and verifies that identity-mode 4-axis
    output equals HF's reference 1D cos/sin (computed inline) within fp32 tolerance.
    """
    import torch

    head_dim = 256
    n_pairs = head_dim // 2                          # 128
    axis_channels = (n_pairs // 4,) * 4              # (32, 32, 32, 32)
    seq_len = 16
    batch = 1
    n_heads = 4

    cases = [
        ("sliding", 10_000.0, 1.0),                  # rope_theta_local, no linear scaling
        ("full",    1_000_000.0, 1.0 / 8.0),         # rope_theta_global, linear factor=8.0
    ]

    for name, base, pos_scale in cases:
        # Compute inv_freq HF-style. For "default" rope_type:
        #   inv_freq[i] = 1 / (base ** (2i / dim))   for i in [0, dim/2)
        # For "linear" rope_type:
        #   inv_freq /= factor                       (== positions / factor)
        freqs = torch.arange(0, n_pairs, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (2 * freqs / head_dim))
        inv_freq = inv_freq * pos_scale              # linear scaling: divide by factor

        table = MRope4DAngleTable.from_hf_rotary_emb(
            inv_freq=inv_freq,
            attention_scaling=1.0,
            head_dim=head_dim,
            axis_channels=axis_channels,
            low_freq_axis="none",
            dtype=torch.float32,
        )

        # 4-axis input with all axes equal: must reduce to HF's 1D RoPE for that base+scaling
        p = torch.arange(seq_len)
        pos_mrope = torch.stack([p, p, p, p], dim=0)
        cos_m, sin_m = table(pos_mrope)              # [seq_len, head_dim]

        # HF's reference 1D cos/sin
        freqs_ref = p.float()[:, None] * inv_freq[None, :]
        emb_ref = torch.cat([freqs_ref, freqs_ref], dim=-1)
        cos_r, sin_r = emb_ref.cos(), emb_ref.sin()

        # Apply both to random q/k and compare
        torch.manual_seed(0)
        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        q_m, k_m = apply_mrope_4d(q, k, cos_m, sin_m, head_dim)
        q_r, k_r = apply_mrope_4d(q, k, cos_r, sin_r, head_dim)

        diff_q = (q_m - q_r).abs().max().item()
        diff_k = (k_m - k_r).abs().max().item()
        assert diff_q < 1e-5, f"{name}: q identity reduction FAILED ({diff_q})"
        assert diff_k < 1e-5, f"{name}: k identity reduction FAILED ({diff_k})"
        print(f"[ok] HF-path identity reduction ({name:7s}, base={base:>10g}, "
              f"pos_scale={pos_scale:.4f}): max diff q={diff_q:.2e} k={diff_k:.2e}")


# --------------------------------------------------------------------------
# Model-level retrofit helper has moved to mrope_patch.py — see
# patch_medgemma_with_mrope_4d(). This module keeps only the math primitives
# (angle table, apply_mrope_4d, build_position_ids, identity-reduction test).
# --------------------------------------------------------------------------


if __name__ == "__main__":
    _test_identity_reduction()
    _test_identity_reduction_hf_path()

    # Show how visual tokens get distinct rotation
    print("\nDemo: visual tokens with distinct (t, d, h, w):")
    cfg = MRope4DConfig(head_dim=64, rotary_dim=32, axis_channels=(4, 4, 4, 4))
    table = MRope4DAngleTable(cfg)
    visual_pos = torch.tensor([
        [0, 0, 1, 1, 2, 2],    # t: T1, T1, T1CE, T1CE, T2, T2
        [5, 5, 5, 6, 6, 6],    # d (slice)
        [10, 11, 10, 11, 10, 11],  # h
        [20, 20, 21, 21, 22, 22],  # w
    ])
    cos_vis, sin_vis = table(visual_pos)
    print(f"  visual pos shape: {visual_pos.shape}")
    print(f"  cos/sin shape:    {cos_vis.shape}")
    print(f"  channel partition: t={cfg.axis_channels[0]} d={cfg.axis_channels[1]} "
          f"h={cfg.axis_channels[2]} w={cfg.axis_channels[3]} pairs")
