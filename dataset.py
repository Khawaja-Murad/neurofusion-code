"""
dataset.py — PyTorch Dataset for NeuroFusion-369.

Reads:
  - neurofusion_369.jsonl   (typed reports from preprocess_reports.py)
  - BraTS 2020 .nii volumes (4 modalities + seg per case)
  - splits.json             (case_id → fold/test/conformal assignment)

Emits, per item:
  mri        : FloatTensor [4, D, H, W]    (T1, T1CE, T2, FLAIR co-registered channels)
  seg        : LongTensor  [D, H, W]       (BraTS labels: 0=bg, 1=NCR, 2=ED, 4=ET; remapped)
  report     : BrainTumorReport            (typed Pydantic object)
  report_json: str                         (canonical JSON string for LM supervision)
  case_id    : str

IMPORTANT: The 4 MRI modalities are CHANNELS of a single 3D volume, co-registered
voxel-to-voxel. The dataset has 369 cases — not 369 × 4 = 1476. This is the standard
BraTS input layout (see: Bakas et al. 2017, Menze et al. 2014).

Usage:
    from dataset import NeuroFusionDataset, get_split_loaders
    train, val = get_split_loaders(fold=0, batch_size=2)
    batch = next(iter(train))
"""
from __future__ import annotations

import os

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from schema import BrainTumorReport

log = logging.getLogger("dataset")

# BraTS 2020 modality file suffixes (standard naming: BraTS20_Training_XXX_<mod>.nii.gz)
_MODALITIES = ["t1", "t1ce", "t2", "flair"]
_SEG_SUFFIX = "seg"

# BraTS-2023 cohorts (RadGenome GLI/MEN weak-sup) name modalities t1n/t1c/t2w/t2f.
# When the 2020 suffix is absent we fall back to the 2023 alias, so ONE --brats-root
# can mix BraTS-2020 (BraTS20_Training_*) and BraTS-2023 (RG_* weak-sup) cases.
# 2020 dirs always resolve the primary suffix first, so their behavior is unchanged.
_MOD_2023_ALIAS = {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"}

# BraTS label remapping → contiguous {0, 1, 2, 3} for CE loss. 2020 uses {0 bg, 1 NCR/NET,
# 2 ED, 4 ET}; 2023 cohorts use {0,1,2,3} natively (ET already = 3). The unified map below
# is correct for BOTH (2020 has no label 3; 2023 has no label 4) and leaves 2020 unchanged.
_BRATS_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}


class NeuroFusionDataset(Dataset):
    """One item = one case (4-channel volume + typed report)."""

    def __init__(
        self,
        jsonl_path: Path,
        brats_root: Path,
        case_ids: list[str] | None = None,
        transforms: Callable | None = None,
        cache_volumes: bool = False,
        normalize: Literal["zscore", "zscore_support", "minmax"] = "zscore",
        target_shape: tuple[int, int, int] | None = (128, 128, 128),
        seg_zero_placeholder: bool = False,
    ) -> None:
        """
        Args:
            jsonl_path: path to neurofusion_369.jsonl
            brats_root: path to BraTS 2020 training data (one subdir per case)
            case_ids:   if provided, restrict to this subset (used with splits.json)
            transforms: optional MONAI-style transform callable applied to (mri, seg)
            cache_volumes: keep loaded volumes in memory (only feasible for small splits)
            normalize:  "zscore" (per-modality, non-zero voxels) or "minmax"
            target_shape: resize/crop to this shape; None = keep native
            seg_zero_placeholder: IMAGE-ONLY roots with NO ground-truth segmentation (the
                triage K9 MR-RATE sidecars). The returned `seg` is an all-zero tensor
                built in memory; NO seg file is read, and the case dir must (a) carry the
                `*_NOT_A_SEG_zero_placeholder.nii.gz` marker written by
                scripts/triage_t8_k9_pregate.py and (b) contain NO `*seg.nii*` file, so
                the flag can never be pointed at a real BraTS root and a real seg can
                never be silently replaced by zeros.
        """
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.brats_root = Path(brats_root)
        self.seg_zero_placeholder = bool(seg_zero_placeholder)
        self.transforms = transforms
        self.cache_volumes = cache_volumes
        self.normalize = normalize
        self.target_shape = target_shape

        # Load all reports
        reports: list[BrainTumorReport] = []
        with self.jsonl_path.open("r") as f:
            for line in f:
                reports.append(BrainTumorReport.model_validate_json(line))

        # Subset by case_ids if provided
        if case_ids is not None:
            id_set = set(case_ids)
            reports = [r for r in reports if r.case_id in id_set]

        # Resolve each case to its BraTS directory
        self.items: list[dict[str, Any]] = []
        for r in reports:
            case_dir = self._find_case_dir(r.case_id)
            if case_dir is None:
                log.warning(f"case {r.case_id}: BraTS directory not found, skipping")
                continue
            self.items.append({"report": r, "case_dir": case_dir})

        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        log.info(f"dataset ready: {len(self.items)} cases")

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _find_case_dir(self, case_id: str) -> Path | None:
        """Find the BraTS subdirectory matching the case_id.

        BraTS 2020 uses ``BraTS20_Training_XXX`` naming. Our report IDs are ``TR<N>``
        (e.g. TR01, TR47, TR331). Mapping: strip the ``TR`` prefix, zero-pad the
        numeric part to 3 digits, prepend ``BraTS20_Training_``.
        """
        # Direct match (in case caller already passed the BraTS-style id)
        direct = self.brats_root / case_id
        if direct.exists():
            return direct

        # Canonical TR → BraTS20_Training_XXX mapping
        if case_id.startswith("TR") and case_id[2:].isdigit():
            num = int(case_id[2:])
            mapped = self.brats_root / f"BraTS20_Training_{num:03d}"
            if mapped.exists():
                return mapped

        # Glob fallback: any subdir containing the case_id stem
        matches = list(self.brats_root.glob(f"*{case_id}*"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            log.warning(f"ambiguous case_id {case_id}: {len(matches)} matches")
            return matches[0]
        return None

    def _find_nii(self, case_dir: Path, suffix: str) -> Path | None:
        """Find the .nii or .nii.gz file in case_dir ending with _<suffix>."""
        for ext in (".nii.gz", ".nii"):
            for p in case_dir.glob(f"*_{suffix}{ext}"):
                return p
            for p in case_dir.glob(f"*{suffix}{ext}"):
                return p
        return None

    # ------------------------------------------------------------------
    # Volume loading + preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_canonical_array(path: Path, dtype) -> np.ndarray:
        """Load a nifti and reorient its data to LAS so the L/R (axis-0)
        direction is CONSISTENT across datasets.

        BraTS-2020 is stored LAS (aff2axcodes ('L','P','S')) and the seg ->
        centroid -> hemisphere mapping was learned in that convention. The
        RadGenome / BraTS-2023 cohorts are stored RAS (('R','A','S')) — opposite
        L/R axis — so without reorientation every cohort case is mirrored and
        NF's hemisphere/laterality is flipped (the 2026-06-07 orientation bug;
        verified 96.5% L/R flip on GLI). Reorienting to LAS is a no-op for
        BraTS-2020 and flips axis-0 for RAS inputs, making the pipeline robust
        to any input orientation (deployability)."""
        import nibabel as nib  # lazy import
        from nibabel.orientations import (
            axcodes2ornt, io_orientation, ornt_transform, apply_orientation,
        )

        img = nib.load(str(path))
        data = img.get_fdata()
        transform = ornt_transform(io_orientation(img.affine),
                                   axcodes2ornt(("L", "P", "S")))
        data = apply_orientation(data, transform)
        return data.astype(dtype)

    def _load_modality(self, case_dir: Path, modality: str) -> np.ndarray:
        """Load one .nii modality volume as float32 numpy array (LAS-canonical)."""
        path = self._find_nii(case_dir, modality)
        if path is None:
            # BraTS-2023 weak-sup cases name modalities t1n/t1c/t2w/t2f.
            alias = _MOD_2023_ALIAS.get(modality)
            if alias is not None:
                path = self._find_nii(case_dir, alias)
        if path is None:
            raise FileNotFoundError(f"modality {modality} not found in {case_dir}")
        return self._load_canonical_array(path, np.float32)

    def _load_seg(self, case_dir: Path) -> np.ndarray:
        path = self._find_nii(case_dir, _SEG_SUFFIX)
        if path is None:
            raise FileNotFoundError(f"seg not found in {case_dir}")
        seg = self._load_canonical_array(path, np.int64)
        # Remap BraTS labels to contiguous {0, 1, 2, 3}
        remapped = np.zeros_like(seg)
        for src, dst in _BRATS_LABEL_REMAP.items():
            remapped[seg == src] = dst
        return remapped

    def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        """Per-modality normalization over non-zero (in-brain) voxels.

        ☠️ "zscore" (the trunk's TRAINING semantics, byte-for-byte unchanged) detects support as
        `vol > 0` — correct for RAW non-negative intensities, and catastrophically wrong for
        SIGNED inputs: on the contract-v3 MR-RATE volumes (z-scores BAKED at ingest) it ERASED
        the negative-z half of the brain — measured 57% of nonzero voxels on the first sidecar
        case (2026-08-31, R-C2-1 idempotence measurement; job 2145613) — before the trunk saw
        them, while the BraTS side kept its full support. A side-correlated pipeline distortion
        is exactly what the criterion-2 source contrast exists to NOT measure.

        "zscore_support" differs ONLY in the support predicate: `vol != 0`. On raw non-negative
        data the two predicates select the SAME voxels (drilled: byte-identical output), so this
        is a support-detection fix, not a normalisation change; on signed data it keeps the whole
        brain, and re-z-scoring a z-scored volume over the true support is a near-identity
        (affine invariance), matching what the serving path feeds the trunk.
        """
        if self.normalize in ("zscore", "zscore_support"):
            mask = (vol != 0) if self.normalize == "zscore_support" else (vol > 0)
            if mask.sum() == 0:
                return vol
            mean = vol[mask].mean()
            std = vol[mask].std()
            if std < 1e-8:
                return np.zeros_like(vol)
            out = np.zeros_like(vol)
            out[mask] = (vol[mask] - mean) / std
            return out
        elif self.normalize == "minmax":
            lo, hi = np.percentile(vol[vol > 0], [1, 99]) if (vol > 0).sum() > 0 else (0.0, 1.0)
            return np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        else:
            raise ValueError(f"unknown normalize mode: {self.normalize}")

    def _resize_to_target(self, arr: np.ndarray, is_label: bool = False) -> np.ndarray:
        """Resize via scipy ndimage zoom to target_shape."""
        if self.target_shape is None:
            return arr
        if arr.shape == self.target_shape:
            return arr
        from scipy.ndimage import zoom

        zoom_factors = tuple(t / s for t, s in zip(self.target_shape, arr.shape))
        order = 0 if is_label else 1  # nearest for labels, linear for images
        return zoom(arr, zoom_factors, order=order).astype(arr.dtype)

    # ☠️ K9 sidecar contract (NF 1.4 north-star §1 item 15): MR-RATE cases have NO voxel
    # annotation, for either class. The old design symlinked `<alias>_seg.nii.gz` to an
    # all-zero file purely to satisfy _load_seg — and that name matched every `*seg.nii*`
    # glob in the codebase, so a stray copy under a training root would have trained
    # MR-RATE positives as "no tumour". The placeholder is now named so that NO seg glob
    # can match it, and this branch never opens it: it is a MARKER, not data.
    SEG_PLACEHOLDER_SUFFIX = "_NOT_A_SEG_zero_placeholder.nii.gz"

    def _zero_seg_no_gt(self, case_dir: Path, shape: tuple[int, ...]) -> np.ndarray:
        marker = list(case_dir.glob(f"*{self.SEG_PLACEHOLDER_SUFFIX}"))
        if not marker:
            raise FileNotFoundError(
                f"seg_zero_placeholder=True but {case_dir} carries no "
                f"*{self.SEG_PLACEHOLDER_SUFFIX} marker — this is not a K9 sidecar case dir "
                f"(or it predates --migrate-sidecar-seg); refusing to fabricate a seg.")
        real = [p for ext in (".nii.gz", ".nii") for p in case_dir.glob(f"*seg{ext}")]
        if real:
            raise RuntimeError(
                f"seg_zero_placeholder=True but {case_dir} ALSO holds a seg-named file "
                f"{[p.name for p in real]} — a real seg must never be replaced by zeros.")
        return np.zeros(tuple(shape), dtype=np.int64)

    def _load_case(self, case_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Load 4-channel MRI + seg tensors for one case."""
        if self.cache_volumes and str(case_dir) in self._cache:
            return self._cache[str(case_dir)]

        channels = []
        for mod in _MODALITIES:
            vol = self._load_modality(case_dir, mod)
            vol = self._normalize_volume(vol)
            vol = self._resize_to_target(vol, is_label=False)
            channels.append(vol)
        mri = np.stack(channels, axis=0)  # [4, D, H, W]
        if self.seg_zero_placeholder:
            seg = self._zero_seg_no_gt(case_dir, mri.shape[1:])
        else:
            seg = self._load_seg(case_dir)
            seg = self._resize_to_target(seg, is_label=True)  # [D, H, W]

        mri_t = torch.from_numpy(mri).float()
        seg_t = torch.from_numpy(seg).long()

        if self.cache_volumes:
            self._cache[str(case_dir)] = (mri_t, seg_t)

        return mri_t, seg_t

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        report: BrainTumorReport = item["report"]
        mri, seg = self._load_case(item["case_dir"])

        if self.transforms is not None:
            mri, seg = self.transforms(mri, seg)

        return {
            "mri": mri,                # [4, D, H, W]
            "seg": seg,                # [D, H, W]
            "report": report,          # Pydantic object
            # NF_STRIP_CASE_ID (2026-07-23): the cohort-tagged case_id is the FIRST field of the
            # LM supervision target, two fields before differential_diagnosis. Trained that way the
            # model learns to COPY the cohort out of the id instead of reading the image: with a
            # deliberately WRONG tag it followed the tag in 19/19 cases (dx 0.0), and with a shuffled
            # tag it scored 0.308 vs 0.333 chance. case_id has no clinical role in a generated
            # report, so strip it from supervision. Default OFF preserves byte-identical
            # reproduction of every existing (tag-confounded) checkpoint.
            "report_json": report.model_dump_json(
                exclude={"case_id"} if os.environ.get("NF_STRIP_CASE_ID") == "1" else None),
            "case_id": report.case_id,
        }


# ======================================================================
# Split loaders — the canonical entry point for training/eval scripts
# ======================================================================


def load_splits(splits_path: Path) -> dict[str, Any]:
    with open(splits_path) as f:
        return json.load(f)


def case_ids_for(splits: dict[str, Any], which: str, fold: int | None = None) -> list[str]:
    """Resolve case_ids for a given split selection.

    which: 'test' | 'conformal_calibration' | 'train' | 'val' | 'all_cv'
    fold: required if which in {'train', 'val'}
    """
    if which == "test":
        return splits["test"]
    if which in ("test_gli", "test_men", "test_met"):
        return splits[which]
    if which == "conformal_calibration":
        return splits["conformal_calibration"]
    if which == "all_cv":
        out: list[str] = []
        for fold_cids in splits["folds"].values():
            out.extend(fold_cids)
        return out
    if which == "val":
        assert fold is not None
        return splits["folds"][str(fold)]
    if which == "train":
        assert fold is not None
        out = []
        for f, cids in splits["folds"].items():
            if int(f) != fold:
                out.extend(cids)
        return out
    raise ValueError(f"unknown split selector: {which}")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate: stacks tensors; keeps report objects as a list.

    Pydantic objects are NOT stacked; LM supervision consumes report_json strings.
    """
    mri = torch.stack([b["mri"] for b in batch])       # [B, 4, D, H, W]
    seg = torch.stack([b["seg"] for b in batch])       # [B, D, H, W]
    return {
        "mri": mri,
        "seg": seg,
        "reports": [b["report"] for b in batch],
        "report_jsons": [b["report_json"] for b in batch],
        "case_ids": [b["case_id"] for b in batch],
    }


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker seed -- MODULE-LEVEL so it pickles under the 'spawn'
    start method. Without distinct seeds, workers inherit the parent numpy RNG
    state and emit identical augmentation streams (a known PyTorch fork trap)."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Cohort-balanced sampling (L4, --cohort-balanced-sampler; default OFF)
# ---------------------------------------------------------------------------
def _cohort_of_case_id(case_id: str) -> str:
    """Cohort label from a case_id prefix (mirrors build_balanced_corpus.infer_cohort's
    prefix rule): RG_MEN_ -> MEN, RG_MET_ -> MET; RG_GLI_ and every BraTS-2020 TR*
    (all gliomas) -> GLI. Used ONLY to weight the optional WeightedRandomSampler; the
    tag never enters the model input or the training target (leak-free)."""
    cid = case_id or ""
    if cid.startswith("RG_MEN_"):
        return "MEN"
    if cid.startswith("RG_MET_"):
        return "MET"
    return "GLI"


def get_split_loaders(
    jsonl_path: Path,
    brats_root: Path,
    splits_path: Path,
    fold: int,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_volumes: bool = False,
    target_shape: tuple[int, int, int] | None = (128, 128, 128),
    augment_train: bool = False,
    cohort_balanced_sampler: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for a given fold, using splits.json assignments.

    augment_train: if True, applies 3D MRI augmentation to the training set
        only (flips on non-sagittal axes, small affine, intensity jitter).
        Val/test/conformal always run without augmentation.
    cohort_balanced_sampler: if True, draw the training set with a
        WeightedRandomSampler (per-case weight = 1/cohort-count) so each cohort
        (GLI/MEN/MET) contributes ~equal mass per epoch, replacing shuffle=True.
        Default False = shuffle=True, no sampler (byte-identical to before).
    """
    splits = load_splits(splits_path)
    train_ids = case_ids_for(splits, "train", fold=fold)
    val_ids = case_ids_for(splits, "val", fold=fold)

    train_transforms = None
    if augment_train:
        from dataset_augment import make_train_transform
        train_transforms = make_train_transform()

    train_ds = NeuroFusionDataset(jsonl_path, brats_root, case_ids=train_ids,
                                   cache_volumes=cache_volumes, target_shape=target_shape,
                                   transforms=train_transforms)
    val_ds = NeuroFusionDataset(jsonl_path, brats_root, case_ids=val_ids,
                                 cache_volumes=cache_volumes, target_shape=target_shape)

    # num_workers>0: use the 'spawn' start method (NOT the default 'fork'). The
    # Mistral 4-bit load initializes CUDA + bitsandbytes worker threads in the
    # MAIN process; forking DataLoader workers after that inherits those threads'
    # held futexes and DEADLOCKS the workers (observed 2026-06-13: pt_data_worker
    # stuck in futex_wait, GPU idle, no first batch). Spawn workers start a fresh
    # interpreter and inherit no CUDA/bnb thread state. persistent_workers
    # amortizes the one-time spawn startup over the run. _seed_worker is now
    # module-level so it pickles for spawn; per-worker seeding still diversifies
    # the augmentation streams.
    dl_kwargs: dict[str, Any] = dict(
        batch_size=batch_size, collate_fn=collate_fn, pin_memory=True,
        num_workers=num_workers, worker_init_fn=_seed_worker,
    )
    if num_workers > 0:
        import multiprocessing as _mp
        dl_kwargs["multiprocessing_context"] = _mp.get_context("spawn")
        dl_kwargs["persistent_workers"] = True

    if cohort_balanced_sampler:
        # L4: oversample minority cohorts so each (GLI/MEN/MET) contributes ~equal
        # mass per epoch. Per-case weight = 1 / (#cases of that case's cohort in the
        # train split); WeightedRandomSampler(replacement=True) then draws
        # len(train_ds) indices per epoch. sampler and shuffle are mutually exclusive
        # in a DataLoader, so we DROP shuffle=True here. Cohort is read from the
        # case_id prefix only (leak-free: it sets sampling frequency, never enters the
        # model input or the teacher-forced target). train_ds.items order matches the
        # dataset index the sampler draws, so weights[i] aligns with index i.
        _cohorts = [_cohort_of_case_id(it["report"].case_id) for it in train_ds.items]
        _counts = Counter(_cohorts)
        _weights = [1.0 / _counts[c] for c in _cohorts]
        sampler = WeightedRandomSampler(
            weights=_weights, num_samples=len(train_ds), replacement=True,
        )
        log.info(
            f"cohort-balanced sampler ENABLED: {len(train_ds)} draws/epoch "
            f"(replacement) over cohort counts {dict(_counts)}"
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader


# ======================================================================
# Sanity check
# ======================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--brats-root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    train, val = get_split_loaders(args.jsonl, args.brats_root, args.splits, fold=args.fold,
                                    batch_size=1, num_workers=0)
    print(f"train cases: {len(train.dataset)}, val cases: {len(val.dataset)}")
    batch = next(iter(train))
    print(f"mri batch shape: {batch['mri'].shape}")
    print(f"seg batch shape: {batch['seg'].shape}")
    print(f"first case_id: {batch['case_ids'][0]}")
    print(f"first report lesions: {len(batch['reports'][0].lesions)}")
