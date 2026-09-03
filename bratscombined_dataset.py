"""
bratscombined_dataset.py — Phase 1 segmentation-only Dataset.

Combines BraTS 2020 + 2021 + 2023 (and any other BraTS-format root) into a single
unpaired segmentation training set for the MedNeXt backbone pretrain stage.

What this is NOT: it does not load paired clinician reports. Phase 1's only
supervision is the BraTS segmentation labels. NeuroFusion-369 (paired data) is
loaded by `dataset.NeuroFusionDataset` for Phase 2b.

Naming convention quirks across BraTS years:

  BraTS 2020:   <root>/BraTS20_Training_NNN/BraTS20_Training_NNN_<mod>.nii.gz
                <root>/BraTS20_Training_NNN/BraTS20_Training_NNN_seg.nii.gz
                modalities: t1, t1ce, t2, flair  (lowercase suffix)

  BraTS 2021:   <root>/BraTS2021_NNNNN/BraTS2021_NNNNN_<mod>.nii.gz
                modalities: t1, t1ce, t2, flair

  BraTS 2023:   <root>/BraTS-GLI-NNNNN-NNN/BraTS-GLI-NNNNN-NNN-<mod>.nii.gz
                modalities: t1n, t1c, t2w, t2f   (renamed from 2020/2021!)
                seg suffix: -seg.nii.gz (hyphen not underscore)

The discovery layer normalizes all of these to a unified internal modality order
(T1, T1CE, T2, FLAIR) and BraTS label set ({0=bg, 1=NCR, 2=ED, 4=ET} → {0,1,2,3}).

BraTS 2023 also redefined some labels — adult-glioma uses {0,1,2,3} natively where
3=ET. The remap below handles both old (1,2,4) and new (1,2,3) label conventions.

Smoke test:
    python bratscombined_dataset.py --roots /data/BraTS2020 /data/BraTS2021 /data/BraTS2023
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("bratscombined")


# ---------------------------------------------------------------------------
# Modality discovery rules per BraTS year
# ---------------------------------------------------------------------------

# Each rule maps year-tag -> (modality canonical -> filename suffix list, in priority order)
_BRATS_MODALITY_SUFFIXES: dict[str, dict[str, list[str]]] = {
    "2020": {
        "t1": ["t1"], "t1ce": ["t1ce"], "t2": ["t2"], "flair": ["flair"],
    },
    "2021": {
        "t1": ["t1"], "t1ce": ["t1ce"], "t2": ["t2"], "flair": ["flair"],
    },
    "2023": {
        # BraTS 2023 renamed: t1 -> t1n, t1ce -> t1c, t2 -> t2w, flair -> t2f
        "t1": ["t1n", "t1"], "t1ce": ["t1c", "t1ce"],
        "t2": ["t2w", "t2"], "flair": ["t2f", "flair"],
    },
    "unknown": {
        # Fallback: try both naming schemes
        "t1": ["t1n", "t1"], "t1ce": ["t1c", "t1ce"],
        "t2": ["t2w", "t2"], "flair": ["t2f", "flair"],
    },
}

# Seg label remapping: handle both BraTS 2020/2021 (1,2,4 with no class 3) and
# BraTS 2023 adult-glioma (1,2,3 contiguous).
_BRATS_LABEL_REMAP_LEGACY = {0: 0, 1: 1, 2: 2, 4: 3}    # 2020/2021
_BRATS_LABEL_REMAP_2023 = {0: 0, 1: 1, 2: 2, 3: 3}      # 2023 (already contiguous)


def _detect_year(root: Path, sample_dir_name: str) -> str:
    """Heuristically detect BraTS year from root path or sample dir name.
    Checks more-specific substrings first to avoid 'brats20' matching 'brats2023'."""
    s = (str(root) + " " + sample_dir_name).lower()
    # BraTS-2023 cohorts all use the t1n/t1c/t2w/t2f + hyphen-seg naming. Match every
    # cohort prefix, not just GLI, so the reorganized MEN/PED/MET roots (and the assembled
    # radgenome_brats_root, whose case dirs carry BraTS-<cohort>-NNNNN-NNN) detect 2023.
    if "2023" in s or "brats-gli" in s or "brats-men" in s or "brats-ped" in s or "brats-met" in s:
        return "2023"
    if "2021" in s:
        return "2021"
    if "2020" in s or "brats20_" in s:
        return "2020"
    return "unknown"


def _find_nii(case_dir: Path, suffix_options: list[str]) -> Path | None:
    """Find the first .nii / .nii.gz file in case_dir matching any of the suffixes.
    Tries _<suffix> first (BraTS 2020/2021), then -<suffix> (BraTS 2023)."""
    for suffix in suffix_options:
        for ext in (".nii.gz", ".nii"):
            for sep in ("_", "-"):
                matches = list(case_dir.glob(f"*{sep}{suffix}{ext}"))
                if matches:
                    return matches[0]
    return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class BraTSCombinedDataset(Dataset):
    """Unpaired BraTS segmentation dataset across multiple years.

    Returns per item:
        mri:  FloatTensor [4, D, H, W]   (T1, T1CE, T2, FLAIR co-registered channels)
        seg:  LongTensor  [D, H, W]      (remapped to {0=bg, 1=NCR, 2=ED, 3=ET})
        case_id: str
        year:    str ('2020' / '2021' / '2023' / 'unknown')
    """

    def __init__(
        self,
        roots: list[Path],
        target_shape: tuple[int, int, int] | None = (128, 128, 128),
        normalize: str = "zscore",  # 'zscore' | 'minmax'
        cache_index: bool = True,
        transforms: Callable | None = None,
        skip_missing_modality: bool = True,
        max_per_root: int | None = None,
        optional_modalities: frozenset[str] | set[str] | None = None,
        canonical_orientation: str | None = None,
        fixed_mask: "np.ndarray | None" = None,
    ) -> None:
        """optional_modalities: channels that may be ABSENT and are zero-filled instead of skipping
        the case. Default None == {} == the historical behaviour (require all 4), so every existing
        caller is byte-identical.

        Measured basis for which channels are droppable (out/m6_research/modality_ablation.json,
        n=91, inference-time zero-fill on the 4-channel trunk):
            T2w  -0.0026 Dice   T1  +0.0092   |   FLAIR -0.4562   T1ce -0.3161
        So {'t1','t2'} is the defensible optional set and FLAIR/T1ce are load-bearing.
        ⚠️ Two caveats that must not be lost: that ablation is MET-only, WT-only, has NO CI, and was
        computed on the leak-#5 trunk; and "droppable at inference" is NOT the same claim as
        "trainable when never present". Re-measure with a CI on the clean trunk before relying on it.
        ⚠️ If a whole cohort is systematically zero-filled on one channel, "that channel is constant"
        becomes a free cohort label inside connector_IN -- the very vector the diagnosis head reads.
        Balance the zero-filling across cohorts and probe for it (plan W3-A / kill criterion K9).
        """
        super().__init__()
        self.target_shape = target_shape
        self.normalize = normalize
        self.transforms = transforms
        self.skip_missing_modality = skip_missing_modality
        self.optional_modalities = frozenset(optional_modalities or ())
        # canonical_orientation: e.g. "LPS". DEFAULT None == current behaviour (raw on-disk axis
        # order, affine ignored) so leak5_repro_phase1_split.py and leak5_effect_size_eval.py keep
        # reproducing bit-exactly. Turning it on CHANGES THE DATA DISTRIBUTION and therefore
        # invalidates any warm-start from a trunk trained without it -- only enable it for a
        # from-scratch run (Phase-1-R).
        self.canonical_orientation = canonical_orientation
        self.n_zero_filled = 0
        self.n_skipped_missing = 0
        # ★ NF 1.4 fixed-support normaliser (OPT-IN; `normalize="fixed_support"`). Contract v3
        # (ruling A, 2026-08-25): statistics from FIXED_MASK ∩ template parenchyma
        # (fixed_support.zscore_stats_region), applied to EVERY mask voxel (the ring is still
        # served), exact zeros outside, a constant channel -> zeros + DECLARED
        # (fixed_support.zscore_in_mask, the ONE implementation shared with the eval bankers
        # and nf_infer). The DEFAULT ("zscore" over vol>0) is byte-identical to before: the
        # 1.3 lineage must not move. The mask is REQUIRED in this mode -- a None mask would
        # silently fall back to the old normaliser, which is the void-behind-a-default trap.
        if normalize == "fixed_support":
            if fixed_mask is None:
                raise ValueError("normalize='fixed_support' requires fixed_mask (bool array on "
                                 "the contract grid, in the SAME axis order as the loaded volumes)")
            fixed_mask = np.asarray(fixed_mask, dtype=bool)
            if fixed_mask.ndim != 3 or not fixed_mask.any():
                raise ValueError(f"fixed_mask must be a non-empty 3-D bool array, got {fixed_mask.shape}")
        elif fixed_mask is not None:
            raise ValueError("fixed_mask given but normalize != 'fixed_support' -- refusing a mask "
                             "that would never be applied")
        self.fixed_mask = fixed_mask
        self.n_declared_constant = 0

        # Build the case index across all roots
        self.items: list[dict[str, Any]] = []
        for root in roots:
            root = Path(root)
            if not root.exists():
                log.warning(f"BraTS root does not exist: {root}")
                continue
            n_added = self._index_root(root, max_per_root)
            log.info(f"indexed {n_added} cases from {root}")
        log.info(f"BraTSCombinedDataset ready: {len(self.items)} total cases across {len(roots)} roots")

    # ------------------------------------------------------------------

    def _index_root(self, root: Path, max_n: int | None) -> int:
        """Walk one BraTS root and add all viable cases to self.items."""
        case_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        added = 0
        for case_dir in case_dirs:
            year = _detect_year(root, case_dir.name)
            mod_suffixes = _BRATS_MODALITY_SUFFIXES[year]

            # Try to resolve all 4 modalities + seg
            paths: dict[str, Path | None] = {}
            for mod in ("t1", "t1ce", "t2", "flair"):
                paths[mod] = _find_nii(case_dir, mod_suffixes[mod])
            paths["seg"] = _find_nii(case_dir, ["seg"])

            missing = [k for k, v in paths.items() if v is None]
            # `seg` is never optional -- there is nothing to supervise without it.
            hard_missing = [m for m in missing if m == "seg" or m not in self.optional_modalities]
            if hard_missing:
                # NOTE (fixed 2026-07-25): `skip_missing_modality=False` used to `continue` as well,
                # so the flag only ever changed the LOG LEVEL and never included the case. Anyone
                # setting it False got a silently smaller dataset and no error. It now raises.
                if not self.skip_missing_modality:
                    raise ValueError(
                        f"{case_dir.name}: missing {hard_missing} and skip_missing_modality=False. "
                        f"To train on incomplete data, pass optional_modalities={{'t1','t2'}} "
                        f"(the measured-droppable set) rather than disabling the check.")
                log.debug(f"  {case_dir.name}: skipping (missing {hard_missing})")
                self.n_skipped_missing += 1
                continue
            if missing:
                # Droppable modality absent -> synthesize a zero channel at __getitem__ time.
                # This is the SAME semantics as the measured inference-time ablation: zero AFTER
                # z-scoring is the post-normalization distribution mean, not an OOD constant.
                self.n_zero_filled += 1
                log.debug(f"  {case_dir.name}: zero-filling {missing}")

            self.items.append({
                "case_id": case_dir.name,
                "year": year,
                "paths": paths,
                "zero_filled": tuple(m for m in missing if m != "seg"),
            })
            added += 1
            if max_n is not None and added >= max_n:
                break
        return added

    # ------------------------------------------------------------------

    def _load_volume(self, path: Path) -> np.ndarray:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        data = img.get_fdata()
        if self.canonical_orientation is not None:
            # Backported from dataset.py:156-178, which has been in production on the Phase-2b path
            # since the 2026-06-07 orientation bug (RAS-stored BraTS-2023 vs LAS-stored BraTS-2020
            # mirrored every cohort case; verified 96.5% L/R flip on GLI). Phase-1 never got the fix.
            # Measured on disk 2026-07-25: BraTS2020_train_minus_protected is LPS, radgenome_brats_root
            # is RAS, and BOTH MET roots are MIXED WITHIN THE ROOT -- so this cannot be fixed per-root,
            # only per-case from the affine. The p=0.5 3-axis flip aug has been masking it for seg,
            # but the Phase-2b head that consumes these features DOES emit laterality.
            from nibabel.orientations import (  # type: ignore
                axcodes2ornt, io_orientation, ornt_transform, apply_orientation,
            )
            tf = ornt_transform(io_orientation(img.affine),
                                axcodes2ornt(tuple(self.canonical_orientation)))
            data = apply_orientation(data, tf)
        return data.astype(np.float32)

    def _geom_of(self, path: Path) -> dict:
        """Voxel spacing and native shape, IN THE CANONICAL AXIS ORDER.

        ☠️ D.6: neither loader has ever returned spacing, so every mm^3, diameter and
        midline-distance downstream is wrong by a per-case factor. On the deployed 128^3
        grid a BraTS voxel is 1.875 x 1.875 x 1.211 mm = 4.257 mm^3, not 1 mm^3 -- a 4.26x
        error in volume, straddling the 10 mm RANO-measurable boundary in both directions.

        ☠️ The spacing MUST be permuted with the orientation transform. `apply_orientation`
        reorders axes, so `header.get_zooms()` is in the ON-DISK order and would be silently
        mismatched to the returned array -- and these roots are mixed LPS/RAS *within* a
        single root, so it differs case by case. nib.load is lazy: this reads the header
        only, no voxel data, so it costs nothing.
        """
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        shape = tuple(int(s) for s in img.shape[:3])
        if self.canonical_orientation is not None:
            from nibabel.orientations import (  # type: ignore
                axcodes2ornt, io_orientation, ornt_transform,
            )
            tf = ornt_transform(io_orientation(img.affine),
                                axcodes2ornt(tuple(self.canonical_orientation)))
            perm = [int(tf[i, 0]) for i in range(3)]
            zooms = tuple(zooms[p] for p in perm)
            shape = tuple(shape[p] for p in perm)
        return {"spacing_native_mm": zooms, "native_shape": shape}

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        if self.normalize == "fixed_support":
            from fixed_support import zscore_in_mask  # scripts/ on sys.path (trainer) or repo
            if vol.shape != self.fixed_mask.shape:
                raise ValueError(
                    f"fixed_support normaliser: volume {vol.shape} is not on the contract grid "
                    f"{self.fixed_mask.shape} -- this root is not an `_fs1` root")
            out, info = zscore_in_mask(vol, self.fixed_mask)
            if info["declared_constant"]:
                self.n_declared_constant += 1
                self._last_declared_constant = True
            return out
        if self.normalize == "zscore":
            mask = vol > 0
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
            if (vol > 0).sum() == 0:
                return vol
            lo, hi = np.percentile(vol[vol > 0], [1, 99])
            return np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        return vol

    def _resize(self, arr: np.ndarray, is_label: bool = False) -> np.ndarray:
        if self.target_shape is None or arr.shape == self.target_shape:
            return arr
        from scipy.ndimage import zoom  # type: ignore
        zoom_factors = tuple(t / s for t, s in zip(self.target_shape, arr.shape))
        order = 0 if is_label else 1
        return zoom(arr, zoom_factors, order=order).astype(arr.dtype)

    def _remap_seg(self, seg: np.ndarray, year: str) -> np.ndarray:
        """Remap BraTS labels to contiguous {0, 1, 2, 3}.
        Detects which convention applies by inspecting the present label set."""
        present = set(int(v) for v in np.unique(seg).tolist())
        if 4 in present and 3 not in present:
            remap = _BRATS_LABEL_REMAP_LEGACY
        else:
            remap = _BRATS_LABEL_REMAP_2023
        # GUARD (added 2026-07-25). Everything not in `remap` falls through to the zeros below, i.e.
        # is SILENTLY CONVERTED TO BACKGROUND. A volume holding BOTH 3 and 4 takes the 2023 branch
        # and loses all enhancing tumour; a stray label >4 vanishes. There was no assertion, no
        # warning, and the loss would keep falling. This project has lost enough headlines to silent
        # data corruption -- fail loudly instead.
        unmapped = present - set(remap)
        if unmapped:
            raise ValueError(
                f"seg carries label value(s) {sorted(unmapped)} that the "
                f"{'LEGACY' if remap is _BRATS_LABEL_REMAP_LEGACY else '2023'} remap does not cover; "
                f"they would be silently written to BACKGROUND. present={sorted(present)}. "
                f"Audit the root with scripts/verify_seg_root.py before training on it.")
        out = np.zeros_like(seg, dtype=np.int64)
        for src, dst in remap.items():
            out[seg == src] = dst
        return out

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        paths = item["paths"]

        # Load + normalize 4 modalities. A modality listed in `optional_modalities` and absent on
        # disk is synthesized as zeros AFTER normalization -- matching the measured ablation, where
        # zero is the post-z-score distribution mean rather than an out-of-distribution constant.
        # The zeros are shaped from a channel that IS present, never from a hardcoded (240,240,155):
        # pretrain_phase1_v5.py passes target_shape=None, so _resize is a no-op and np.stack below
        # would raise on any shape mismatch.
        channels: list[np.ndarray] = []
        zero_slots: list[int] = []
        declared_constant: list[str] = []
        for ci, mod in enumerate(("t1", "t1ce", "t2", "flair")):
            p = paths[mod]
            if p is None:
                channels.append(None)  # type: ignore[arg-type]
                zero_slots.append(ci)
                continue
            v = self._load_volume(p)
            self._last_declared_constant = False
            v = self._normalize(v)
            if getattr(self, "_last_declared_constant", False):
                declared_constant.append(mod)          # fixed_support mode only; never silent
            v = self._resize(v, is_label=False)
            channels.append(v)
        if zero_slots:
            ref = next((c for c in channels if c is not None), None)
            if ref is None:
                raise ValueError(f"{item['case_id']}: every modality absent -- cannot synthesize")
            for ci in zero_slots:
                channels[ci] = np.zeros_like(ref)
        mri = np.stack(channels, axis=0)  # [4, D, H, W]

        # Load + remap seg
        seg = self._load_volume(paths["seg"]).astype(np.int64)
        seg = self._remap_seg(seg, item["year"])
        seg = self._resize(seg, is_label=True)

        mri_t = torch.from_numpy(mri).float()
        seg_t = torch.from_numpy(seg).long()

        if self.transforms is not None:
            mri_t, seg_t = self.transforms(mri_t, seg_t)

        # D.6: geometry travels WITH the volume. `spacing_mm` is the EFFECTIVE spacing of
        # the array actually returned -- native spacing scaled by the resize factor -- so a
        # consumer never has to know whether _resize ran. With target_shape=None (Phase-1)
        # it equals the native spacing; at 128^3 (Phase-2b/deploy) it is the 1.875/1.211 grid.
        _g = self._geom_of(paths["seg"])
        _final = tuple(int(s) for s in seg.shape[:3])
        _rf = tuple(n / f for n, f in zip(_g["native_shape"], _final))
        geom = {
            "spacing_mm": tuple(s * r for s, r in zip(_g["spacing_native_mm"], _rf)),
            "spacing_native_mm": _g["spacing_native_mm"],
            "native_shape": _g["native_shape"],
            "final_shape": _final,
            "resize_factors": _rf,
        }

        out = {
            "mri": mri_t,
            "seg": seg_t,
            "case_id": item["case_id"],
            "year": item["year"],
            "geom": geom,
        }
        if self.normalize == "fixed_support":
            # declared-missing semantics (contract D4): the caller can SEE a constant channel
            out["declared_constant"] = tuple(declared_constant)
        return out


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_seg(batch: list[dict[str, Any]]) -> dict[str, Any]:
    # ☠️ This collate picks keys EXPLICITLY and drops everything else, so adding a key to
    # __getitem__ without adding it here is a silent no-op -- the producer/consumer split
    # that has bitten this codebase before. `geom` and `spacing` are forwarded on purpose;
    # `spacing` is the flat per-case tuple the detection/measurement code consumes.
    out = {
        "mri": torch.stack([b["mri"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "case_ids": [b["case_id"] for b in batch],
        "years": [b["year"] for b in batch],
    }
    if "geom" in batch[0]:
        out["geom"] = [b["geom"] for b in batch]
        out["spacing"] = [b["geom"]["spacing_mm"] for b in batch]
    if "declared_constant" in batch[0]:
        out["declared_constant"] = [b["declared_constant"] for b in batch]
    return out


def get_brats_loader(
    roots: list[Path],
    batch_size: int = 2,
    num_workers: int = 4,
    target_shape: tuple[int, int, int] | None = (128, 128, 128),
    shuffle: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for combined BraTS segmentation pretrain.

    Splits cases stratified by year (so each year is proportionally represented
    in train and val).
    """
    ds = BraTSCombinedDataset(roots, target_shape=target_shape)

    # Stratified train/val split by year
    rng = np.random.default_rng(seed)
    by_year: dict[str, list[int]] = {}
    for i, item in enumerate(ds.items):
        by_year.setdefault(item["year"], []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for year, idxs in by_year.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_seg, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_seg, pin_memory=True,
    )
    log.info(f"split: train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-per-root", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    ds = BraTSCombinedDataset(args.roots, max_per_root=args.max_per_root)
    print(f"\n=== Dataset summary ===")
    print(f"total cases: {len(ds)}")
    by_year: dict[str, int] = {}
    for item in ds.items:
        by_year[item["year"]] = by_year.get(item["year"], 0) + 1
    for y, n in sorted(by_year.items()):
        print(f"  {y}: {n}")

    if len(ds) == 0:
        print("no cases found — check --roots paths")
        exit(1)

    print("\n=== Loading first case ===")
    item = ds[0]
    print(f"case_id: {item['case_id']}  year: {item['year']}")
    print(f"mri shape: {tuple(item['mri'].shape)}  dtype: {item['mri'].dtype}")
    print(f"  per-channel mean/std: ", end="")
    for c in range(4):
        m = item["mri"][c].mean().item()
        s = item["mri"][c].std().item()
        print(f"  c{c}: μ={m:+.3f} σ={s:.3f}", end="")
    print()
    print(f"seg shape: {tuple(item['seg'].shape)}  dtype: {item['seg'].dtype}")
    print(f"  unique labels: {sorted(item['seg'].unique().tolist())}")

    print("\n=== DataLoader smoke ===")
    train_loader, val_loader = get_brats_loader(
        args.roots, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    batch = next(iter(train_loader))
    print(f"batch mri:  {tuple(batch['mri'].shape)}")
    print(f"batch seg:  {tuple(batch['seg'].shape)}")
    print(f"batch ids:  {batch['case_ids']}")
    print(f"batch years: {batch['years']}")
    print("\nOK")
