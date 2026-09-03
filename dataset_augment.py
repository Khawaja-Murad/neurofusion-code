"""3D MRI augmentation for Phase 2b multimodal training.

Augmentation is applied per-case after volume loading. The same spatial
transform is applied to both MRI and segmentation; intensity transforms
are MRI-only. Lesion-routing happens AFTER augmentation so the connected-
components routing sees the augmented seg directly.

Design choices:
- All transforms are pure PyTorch ops on the already-cropped/resampled
  volumes (no MONAI dict pipeline). Keeps the call surface trivial:
  `augmented_mri, augmented_seg = transform(mri, seg)`.
- Conservative magnitudes — clinical structures must remain recognizable.
  Brain orientation matters (left vs right), so we DO NOT flip across the
  sagittal axis (which would swap hemispheres and invalidate the GT report's
  "hemisphere": "left/right" field). Only flip along D (axial) and H
  (anterior-posterior) axes.
- All augmentations are off at val/test time.

Probabilities and magnitudes tuned for small (~100-case) cohort training,
where over-augmentation can dominate the signal.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class TrainAugment:
    """Stochastic 3D augmentation: spatial (mri+seg) + intensity (mri only).

    Usage:
        aug = TrainAugment()
        mri_aug, seg_aug = aug(mri, seg)   # mri: [C, D, H, W], seg: [D, H, W]

    Returns tensors of the same shape as input.
    """

    def __init__(
        self,
        p_flip: float = 0.0,         # see _SPATIAL_FLIPS comment below — defaults disabled
        p_rotate: float = 0.5,       # any-axis small rotation
        max_rot_deg: float = 15.0,
        p_scale: float = 0.3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p_translate: float = 0.3,
        max_translate_frac: float = 0.05,
        p_intensity_shift: float = 0.4,
        intensity_shift_range: Tuple[float, float] = (-0.1, 0.1),
        p_intensity_scale: float = 0.4,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        p_gamma: float = 0.3,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        p_noise: float = 0.3,
        noise_std_max: float = 0.05,
    ) -> None:
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.max_rot_rad = math.radians(max_rot_deg)
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_translate = p_translate
        self.max_translate_frac = max_translate_frac
        self.p_intensity_shift = p_intensity_shift
        self.intensity_shift_range = intensity_shift_range
        self.p_intensity_scale = p_intensity_scale
        self.intensity_scale_range = intensity_scale_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_noise = p_noise
        self.noise_std_max = noise_std_max

    def __call__(
        self, mri: torch.Tensor, seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---------- Spatial transforms (mri + seg in lockstep) ----------

        # _SPATIAL_FLIPS — disabled by default (p_flip=0) for a reason.
        #
        # The MRI tensor shape is (C=4, D, H, W) and comes from nibabel via
        # NeuroFusionDataset, which loads BraTS NIfTI in its native on-disk
        # order — for BraTS 2020 that is (X-sagittal, Y-coronal, Z-axial)
        # for the spatial dims. So in the MRI tensor:
        #   dim 1 = X = SAGITTAL  (L–R)        ← flipping swaps hemispheres
        #   dim 2 = Y = CORONAL   (A–P)        ← flipping swaps anterior/posterior labels
        #   dim 3 = Z = AXIAL     (S–I)        ← flipping rotates brain upside-down
        # Same applies to the seg tensor with shape (D, H, W) → dim 0/1/2.
        #
        # An EARLIER version of this code believed dim 1=axial, dim 2=AP,
        # dim 3=sagittal and flipped dim 1 + dim 2 with p=0.5. That silently
        # swapped hemispheres on 50% of training cases — the root cause of
        # the bilateral over-prediction surfaced at inference.
        # See project_phase3_findings_2026-05-16.
        #
        # No anatomical flip is label-preserving for our schema:
        #   - sagittal flip: breaks hemisphere
        #   - coronal flip:  breaks location enum (frontal↔occipital, etc.)
        #   - axial flip:    upside-down brain, semantically nonsense
        # If a future flip is ever re-enabled, it MUST come with corresponding
        # label-flip logic (swap "left"/"right", swap "frontal"/"occipital", …).
        if np.random.rand() < self.p_flip:
            mri = torch.flip(mri, dims=[3])  # Z = axial (intentional, but disabled by default)
            seg = torch.flip(seg, dims=[2])
        if np.random.rand() < self.p_flip:
            mri = torch.flip(mri, dims=[2])  # Y = coronal (intentional, but disabled by default)
            seg = torch.flip(seg, dims=[1])

        # Affine: rotation + scale + translate, applied via a single grid sample.
        do_rot = np.random.rand() < self.p_rotate
        do_scale = np.random.rand() < self.p_scale
        do_trans = np.random.rand() < self.p_translate
        if do_rot or do_scale or do_trans:
            mri, seg = self._affine(mri, seg, do_rot, do_scale, do_trans)

        # ---------- Intensity transforms (mri only) ----------

        if np.random.rand() < self.p_intensity_shift:
            lo, hi = self.intensity_shift_range
            mri = mri + (np.random.rand() * (hi - lo) + lo)
        if np.random.rand() < self.p_intensity_scale:
            lo, hi = self.intensity_scale_range
            mri = mri * (np.random.rand() * (hi - lo) + lo)
        if np.random.rand() < self.p_gamma:
            lo, hi = self.gamma_range
            g = np.random.rand() * (hi - lo) + lo
            mri = mri.sign() * mri.abs().clamp(min=1e-8).pow(g)
        if np.random.rand() < self.p_noise:
            std = np.random.rand() * self.noise_std_max
            mri = mri + torch.randn_like(mri) * std

        return mri.contiguous(), seg.contiguous()

    def _affine(
        self, mri: torch.Tensor, seg: torch.Tensor,
        do_rot: bool, do_scale: bool, do_trans: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotation + scale + translate as a single 3D affine grid_sample.

        Rotation is around the volume center (any-axis small angles).
        """
        device = mri.device
        # Build 3D rotation matrix (small rotations around each axis)
        if do_rot:
            ax, ay, az = (np.random.rand(3) * 2 - 1) * self.max_rot_rad
        else:
            ax = ay = az = 0.0
        Rx = torch.tensor(
            [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]],
            dtype=torch.float32, device=device,
        )
        Ry = torch.tensor(
            [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]],
            dtype=torch.float32, device=device,
        )
        Rz = torch.tensor(
            [[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]],
            dtype=torch.float32, device=device,
        )
        R = Rz @ Ry @ Rx  # [3, 3]

        # Scale (isotropic for simplicity)
        if do_scale:
            lo, hi = self.scale_range
            s = float(np.random.rand() * (hi - lo) + lo)
            R = R * s

        # Translation in normalized coords ([-1, 1] span)
        if do_trans:
            t = torch.tensor(
                (np.random.rand(3) * 2 - 1) * self.max_translate_frac,
                dtype=torch.float32, device=device,
            )
        else:
            t = torch.zeros(3, dtype=torch.float32, device=device)

        theta = torch.cat([R, t.unsqueeze(1)], dim=1).unsqueeze(0)  # [1, 3, 4]

        # mri: [C, D, H, W] -> add batch dim for affine_grid
        mri_b = mri.unsqueeze(0).float()
        seg_b = seg.unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        size = mri_b.shape  # [1, C, D, H, W]
        seg_size = seg_b.shape

        grid_mri = F.affine_grid(theta, size, align_corners=False)
        grid_seg = F.affine_grid(theta, seg_size, align_corners=False)

        mri_out = F.grid_sample(
            mri_b, grid_mri, mode="bilinear", padding_mode="zeros", align_corners=False,
        ).squeeze(0)
        # Use nearest-neighbor for seg to preserve label values
        seg_out = F.grid_sample(
            seg_b, grid_seg, mode="nearest", padding_mode="zeros", align_corners=False,
        ).squeeze(0).squeeze(0).long()

        return mri_out, seg_out


# Functional wrapper for the dataset's `transforms: Callable | None` API.
def make_train_transform() -> TrainAugment:
    """Default Phase 2b training-time augmentation."""
    return TrainAugment()


if __name__ == "__main__":
    # Smoke test: shape and dtype preservation.
    torch.manual_seed(0)
    mri = torch.randn(4, 64, 64, 64)
    seg = torch.randint(0, 4, (64, 64, 64))
    aug = TrainAugment()
    mri_a, seg_a = aug(mri, seg)
    assert mri_a.shape == mri.shape, f"mri shape mismatch: {mri_a.shape} vs {mri.shape}"
    assert seg_a.shape == seg.shape, f"seg shape mismatch: {seg_a.shape} vs {seg.shape}"
    assert seg_a.dtype == seg.dtype, f"seg dtype changed: {seg_a.dtype} vs {seg.dtype}"
    print(f"OK — mri {mri.shape} {mri.dtype} -> {mri_a.shape} {mri_a.dtype}")
    print(f"OK — seg {seg.shape} {seg.dtype} -> {seg_a.shape} {seg_a.dtype}")
    # Verify the augmentation actually changes the data
    assert not torch.allclose(mri, mri_a)
    print("OK — augmentation produces different output (non-identity)")
