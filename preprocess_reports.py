"""
preprocess_reports.py — CSV → typed NeuroFusion-369 JSONL

Pipeline stages (matches diagram 04_preprocessing_pipeline.svg):
  1. Load raw CSV (369 rows, 10 columns)
  2. Canonicalize per-field values (casing, spacing, synonyms)
  3. Split multi-lesion reports into lesion arrays
  4. Build BrainTumorReport Pydantic objects (typed validation)
  5. Cross-check lesion count vs. BraTS segmentation (if .nii available)
  6. Partition: 39 held-out test + 5-fold CV on 330 + 50-case conformal calibration subset
  7. Write: neurofusion_369.jsonl · splits.json · review_queue.csv · dataset_stats.md

Usage:
    python preprocess_reports.py \
        --csv path/to/reports.csv \
        --brats-root path/to/BraTS2020_TrainingData \
        --out-dir path/to/output \
        [--openai-key KEY]  # optional, for LLM-assist on ambiguous multi-lesion

Anonymized for double-blind review.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from enum_mappings import (
    AXIS_SHIFT_MAP,
    COMPOSITION_MAP,
    EDEMA_MAP,
    ENHANCEMENT_MAP,
    HEMISPHERE_MAP,
    INVOLVEMENT_KEYWORDS,
    LOCATION_MAP,
    NO_COMPRESSION_KEYWORDS,
)
from schema import (
    BrainTumorReport,
    Involvement,
    Lesion,
    SurroundingEffects,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("preprocess")


# =====================================================================
# Stage 1: Raw CSV loading
# =====================================================================


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load the clinician-annotated CSV. Returns list of row dicts keyed by column header."""
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("Number", "").strip()]
    log.info(f"loaded {len(rows)} rows from {path}")
    return rows


# =====================================================================
# Stage 2: Canonicalization helpers
# =====================================================================


def _norm(s: str | None) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def canonicalize(raw: str | None, mapping: dict[str, str], field_name: str, review_queue: list[dict]) -> str | None:
    """Map raw string → canonical enum value via `mapping`; log unmatched to review_queue."""
    norm = _norm(raw)
    if not norm:
        return None
    if norm in mapping:
        return mapping[norm]
    # fuzzy: try substring containment for a handful of known terms
    for key, val in mapping.items():
        if key in norm or norm in key:
            return val
    review_queue.append({"field": field_name, "raw_value": raw, "normalized": norm})
    return None


# =====================================================================
# Stage 3: Multi-lesion detection and splitting
# =====================================================================


LESION_MARKER_RE = re.compile(r"lesion\s*(\d+)\s*[:\-]", re.IGNORECASE)


def is_multi_lesion(row: dict[str, str]) -> bool:
    """Detect multi-lesion cases by marker presence in any field."""
    fields_to_check = ["Tumor Composition", "Tumor Enhancement Pattern", "Surrounding Effects", "Report"]
    for field in fields_to_check:
        val = row.get(field, "")
        if LESION_MARKER_RE.search(val):
            return True
    return False


def split_lesion_string(s: str) -> dict[int, str]:
    """Parse 'lesion 1: X / lesion 2: Y' → {1: 'X', 2: 'Y'}.

    Returns {0: s} if no markers found (single-lesion case).
    """
    matches = list(LESION_MARKER_RE.finditer(s))
    if not matches:
        return {0: s.strip()}

    result: dict[int, str] = {}
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        content = s[start:end].strip().rstrip("/;,")
        result[idx] = content
    return result


# =====================================================================
# Stage 4: Build typed Lesion + BrainTumorReport objects
# =====================================================================


def parse_surrounding_effects(raw: str, review_queue: list[dict], case_id: str) -> SurroundingEffects | None:
    """Parse bundled edema+mass_effect+midline_shift string.

    Source format example: 'marked edema, mass effect with midline shift'
    """
    norm = _norm(raw)

    # edema severity — find first matching key
    edema_val = None
    for key, val in EDEMA_MAP.items():
        if key in norm:
            edema_val = val
            break
    if edema_val is None:
        edema_val = "none" if "no edema" in norm else None
    if edema_val is None:
        review_queue.append({"case_id": case_id, "field": "edema_severity", "raw_value": raw})
        return None

    # mass effect
    if "no mass effect" in norm:
        me_val = "none"
    elif "mass effect" in norm:
        me_val = "present"
    else:
        me_val = "none"

    # midline shift
    if "no midline" in norm or "no midline shift" in norm:
        ms_val = "none"
    elif "midline shift" in norm or "midline" in norm:
        ms_val = "present"
    else:
        ms_val = "none"

    try:
        return SurroundingEffects(edema_severity=edema_val, mass_effect=me_val, midline_shift=ms_val)
    except ValidationError as e:
        review_queue.append({"case_id": case_id, "field": "surrounding_effects", "raw_value": raw, "error": str(e)})
        return None


def parse_involvement(raw: str, review_queue: list[dict], case_id: str) -> Involvement | None:
    """Parse ventricular/brainstem/midbrain compression presence from a raw string."""
    norm = _norm(raw)

    # Fast path: no compression
    if any(kw in norm for kw in NO_COMPRESSION_KEYWORDS):
        return Involvement(ventricular_compression="none", brainstem_compression="none", midbrain_compression="none")

    fields: dict[str, str] = {}
    for field, keywords in INVOLVEMENT_KEYWORDS.items():
        fields[field] = "present" if any(kw in norm for kw in keywords) else "none"

    try:
        return Involvement(**fields)
    except ValidationError as e:
        review_queue.append({"case_id": case_id, "field": "involvement", "raw_value": raw, "error": str(e)})
        return None


def build_lesion(
    lesion_index: int,
    composition_raw: str,
    enhancement_raw: str,
    surrounding_raw: str,
    involvement_raw: str,
    axis_shift_raw: str,
    location_raw: str,
    review_queue: list[dict],
    case_id: str,
) -> Lesion | None:
    """Build a single Lesion from raw field strings."""
    composition = canonicalize(composition_raw, COMPOSITION_MAP, "composition", review_queue)
    enhancement = canonicalize(enhancement_raw, ENHANCEMENT_MAP, "enhancement_pattern", review_queue)
    location = canonicalize(location_raw, LOCATION_MAP, "location", review_queue)
    axis_shift = canonicalize(axis_shift_raw, AXIS_SHIFT_MAP, "axis_shift", review_queue)
    surrounding = parse_surrounding_effects(surrounding_raw, review_queue, case_id)
    involvement = parse_involvement(involvement_raw, review_queue, case_id)

    if not all([composition, enhancement, location, axis_shift, surrounding, involvement]):
        review_queue.append({"case_id": case_id, "field": "lesion_build", "raw_value": f"lesion {lesion_index} incomplete"})
        return None

    try:
        return Lesion(
            lesion_index=lesion_index,
            location=location,  # type: ignore
            composition=composition,  # type: ignore
            enhancement_pattern=enhancement,  # type: ignore
            surrounding_effects=surrounding,  # type: ignore
            involvement=involvement,  # type: ignore
            axis_shift=axis_shift,  # type: ignore
        )
    except ValidationError as e:
        review_queue.append({"case_id": case_id, "field": "lesion_final", "raw_value": str(e)})
        return None


def build_report(row: dict[str, str], review_queue: list[dict]) -> BrainTumorReport | None:
    """Build a BrainTumorReport from a raw CSV row."""
    case_id = row["Number"].strip()

    hemisphere = canonicalize(row.get("Hemisphere", ""), HEMISPHERE_MAP, "hemisphere", review_queue)
    if not hemisphere:
        return None

    # DDx: split on common separators
    ddx_raw = row.get("Histopathology/Differential Diagnosis", "")
    ddx: list[str] = [d.strip() for d in re.split(r"[,/]", ddx_raw) if d.strip()]
    if not ddx:
        review_queue.append({"case_id": case_id, "field": "ddx", "raw_value": ddx_raw})
        return None

    # Overall mass effect derived from primary lesion's surrounding_effects
    # (set after lesion construction to match the most-severe reported)

    # Lesion enumeration
    if is_multi_lesion(row):
        composition_splits = split_lesion_string(row.get("Tumor Composition", ""))
        enhancement_splits = split_lesion_string(row.get("Tumor Enhancement Pattern", ""))
        surrounding_splits = split_lesion_string(row.get("Surrounding Effects", ""))
        involvement_splits = split_lesion_string(row.get("Ventricular/Brainstem Involvement", ""))
        axis_shift_splits = split_lesion_string(row.get("Axis Shift", ""))

        lesion_indices = sorted(
            set(composition_splits.keys())
            | set(enhancement_splits.keys())
            | set(surrounding_splits.keys())
        )
        lesion_indices = [i for i in lesion_indices if i > 0]

        lesions: list[Lesion] = []
        for i, idx in enumerate(lesion_indices, start=1):
            # Fallback: if a field doesn't have per-lesion split, use the whole-field value
            les = build_lesion(
                lesion_index=i,
                composition_raw=composition_splits.get(idx, row.get("Tumor Composition", "")),
                enhancement_raw=enhancement_splits.get(idx, row.get("Tumor Enhancement Pattern", "")),
                surrounding_raw=surrounding_splits.get(idx, row.get("Surrounding Effects", "")),
                involvement_raw=involvement_splits.get(idx, row.get("Ventricular/Brainstem Involvement", "")),
                axis_shift_raw=axis_shift_splits.get(idx, row.get("Axis Shift", "")),
                location_raw=row.get("Tumor Location", ""),
                review_queue=review_queue,
                case_id=case_id,
            )
            if les is not None:
                lesions.append(les)
        if not lesions:
            review_queue.append({"case_id": case_id, "field": "all_lesions", "raw_value": "multi-lesion parse produced 0 lesions"})
            return None
    else:
        les = build_lesion(
            lesion_index=1,
            composition_raw=row.get("Tumor Composition", ""),
            enhancement_raw=row.get("Tumor Enhancement Pattern", ""),
            surrounding_raw=row.get("Surrounding Effects", ""),
            involvement_raw=row.get("Ventricular/Brainstem Involvement", ""),
            axis_shift_raw=row.get("Axis Shift", ""),
            location_raw=row.get("Tumor Location", ""),
            review_queue=review_queue,
            case_id=case_id,
        )
        if les is None:
            return None
        lesions = [les]

    # Derive overall_mass_effect from most severe lesion
    severity_order = ["none", "mild", "moderate", "marked"]
    max_severity = "none"
    for les in lesions:
        this = les.surrounding_effects.edema_severity
        if severity_order.index(this) > severity_order.index(max_severity):
            max_severity = this

    # Build findings narrative (use the raw Report field; impression derived from DDx)
    findings = row.get("Report", "").strip()
    impression = f"Findings most consistent with {ddx[0]}."

    try:
        return BrainTumorReport(
            case_id=case_id,
            hemisphere=hemisphere,  # type: ignore
            differential_diagnosis=ddx,
            overall_mass_effect=max_severity,  # type: ignore
            lesions=lesions,
            findings=findings,
            impression=impression,
        )
    except ValidationError as e:
        review_queue.append({"case_id": case_id, "field": "top_level_validation", "raw_value": str(e)})
        return None


# =====================================================================
# Stage 5: BraTS segmentation cross-check (optional, if .nii available)
# =====================================================================


def lesion_count_from_seg(seg_path: Path, min_volume_cm3: float = 1.0) -> int:
    """Count connected components in a BraTS seg.nii (label > 0) above volume threshold.

    Returns -1 if nibabel/scipy unavailable or file missing.
    """
    try:
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import label as scipy_label
    except ImportError:
        return -1

    if not seg_path.exists():
        return -1

    img = nib.load(str(seg_path))
    seg = img.get_fdata()
    binary = (seg > 0).astype(np.uint8)
    labeled, n = scipy_label(binary)

    voxel_vol_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    kept = 0
    for i in range(1, n + 1):
        voxels = int((labeled == i).sum())
        if voxels * voxel_vol_cm3 >= min_volume_cm3:
            kept += 1
    return kept


# =====================================================================
# Stage 6: Partitioning (39 test + 5-fold CV + conformal calibration)
# =====================================================================


def partition_dataset(
    reports: list[BrainTumorReport],
    n_test: int = 39,
    n_folds: int = 5,
    n_conformal: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified partition of reports into test / CV folds / conformal calibration subset.

    Stratification is by (hemisphere, DDx-top-1-lowercased-first-word).
    """
    import random

    rng = random.Random(seed)

    def strat_key(r: BrainTumorReport) -> str:
        top_ddx = r.differential_diagnosis[0].lower().split()[0] if r.differential_diagnosis else "unk"
        return f"{r.hemisphere}|{top_ddx}"

    by_stratum: dict[str, list[BrainTumorReport]] = {}
    for r in reports:
        by_stratum.setdefault(strat_key(r), []).append(r)

    # Build test set: proportional sampling from each stratum
    test_ids: set[str] = set()
    test_target = n_test
    for stratum, rs in by_stratum.items():
        rng.shuffle(rs)
    # Fill test from strata proportionally
    total = len(reports)
    remaining_target = test_target
    strata_order = list(by_stratum.keys())
    rng.shuffle(strata_order)
    for stratum in strata_order:
        rs = by_stratum[stratum]
        share = round(test_target * len(rs) / total)
        share = min(share, len(rs), remaining_target)
        for r in rs[:share]:
            test_ids.add(r.case_id)
        remaining_target -= share
        if remaining_target <= 0:
            break
    # top-up if we under-allocated
    for stratum in strata_order:
        if remaining_target <= 0:
            break
        for r in by_stratum[stratum]:
            if r.case_id not in test_ids:
                test_ids.add(r.case_id)
                remaining_target -= 1
                if remaining_target <= 0:
                    break

    # CV on non-test
    cv_reports = [r for r in reports if r.case_id not in test_ids]
    rng.shuffle(cv_reports)

    # Conformal calibration: take first n_conformal from shuffled CV set
    conformal_ids = {r.case_id for r in cv_reports[:n_conformal]}
    cv_for_folds = cv_reports[n_conformal:]

    # Assign folds round-robin
    fold_assignments: dict[str, int] = {}
    for i, r in enumerate(cv_for_folds):
        fold_assignments[r.case_id] = i % n_folds

    splits = {
        "seed": seed,
        "n_test": len(test_ids),
        "n_conformal": len(conformal_ids),
        "n_folds": n_folds,
        "test": sorted(test_ids),
        "conformal_calibration": sorted(conformal_ids),
        "folds": {str(f): sorted([cid for cid, fold in fold_assignments.items() if fold == f]) for f in range(n_folds)},
    }
    return splits


# =====================================================================
# Stage 7: Stats dump
# =====================================================================


def compute_stats(reports: list[BrainTumorReport]) -> str:
    """Return a markdown-formatted stats report for dataset_stats.md."""
    lines = ["# NeuroFusion-369 dataset statistics", ""]
    lines.append(f"- Total reports: **{len(reports)}**")
    lines.append("")

    # Hemisphere
    hemi_counts = Counter(r.hemisphere for r in reports)
    lines.append("## Hemisphere distribution")
    for h, c in hemi_counts.most_common():
        lines.append(f"- {h}: {c}  ({c/len(reports):.1%})")
    lines.append("")

    # Lesion count
    lesion_counts = Counter(len(r.lesions) for r in reports)
    lines.append("## Lesion count per case")
    for n, c in sorted(lesion_counts.items()):
        lines.append(f"- {n} lesion(s): {c}  ({c/len(reports):.1%})")
    lines.append("")

    # Location (top 10)
    loc_counts = Counter(les.location for r in reports for les in r.lesions)
    lines.append("## Location (top 10)")
    for loc, c in loc_counts.most_common(10):
        lines.append(f"- {loc}: {c}")
    lines.append("")

    # Composition
    comp_counts = Counter(les.composition for r in reports for les in r.lesions)
    lines.append("## Composition")
    for comp, c in comp_counts.most_common():
        lines.append(f"- {comp}: {c}")
    lines.append("")

    # Enhancement
    enh_counts = Counter(les.enhancement_pattern for r in reports for les in r.lesions)
    lines.append("## Enhancement pattern")
    for enh, c in enh_counts.most_common():
        lines.append(f"- {enh}: {c}")
    lines.append("")

    # DDx top-1
    ddx_counts = Counter(r.differential_diagnosis[0] for r in reports)
    lines.append("## Primary differential diagnosis (top 10)")
    for ddx, c in ddx_counts.most_common(10):
        lines.append(f"- {ddx}: {c}")
    lines.append("")

    # Narrative length
    findings_lengths = [len(r.findings.split()) for r in reports]
    lines.append("## Findings narrative length (words)")
    lines.append(f"- min / median / max: {min(findings_lengths)} / {sorted(findings_lengths)[len(findings_lengths)//2]} / {max(findings_lengths)}")
    lines.append("")

    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NeuroFusion-369 typed dataset")
    ap.add_argument("--csv", type=Path, required=True, help="path to raw clinician-annotated CSV")
    ap.add_argument("--brats-root", type=Path, default=None, help="optional: BraTS 2020 training data root for seg cross-check")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory")
    ap.add_argument("--n-test", type=int, default=39)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-conformal", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    rows = load_csv(args.csv)

    # Stage 2-4
    review_queue: list[dict] = []
    reports: list[BrainTumorReport] = []
    failed_case_ids: list[str] = []
    for row in rows:
        report = build_report(row, review_queue)
        if report is not None:
            reports.append(report)
        else:
            failed_case_ids.append(row.get("Number", "?"))

    log.info(f"parsed {len(reports)}/{len(rows)} reports successfully")
    log.info(f"{len(review_queue)} entries queued for review")
    if failed_case_ids:
        log.warning(f"failed cases (first 10): {failed_case_ids[:10]}")

    # Stage 5: BraTS seg cross-check
    if args.brats_root is not None:
        log.info("running BraTS segmentation lesion-count cross-check")
        disagreements = 0
        for r in reports:
            # Map TR01 → BraTS20_Training_001 (strip "TR", zero-pad to 3 digits)
            import re as _re
            m = _re.match(r"TR(\d+)$", r.case_id, _re.IGNORECASE)
            brats_num = m.group(1).zfill(3) if m else r.case_id
            brats_name = f"BraTS20_Training_{brats_num}"
            seg_candidates = list(args.brats_root.rglob(f"{brats_name}_seg.nii*"))
            if not seg_candidates:
                continue
            seg_n = lesion_count_from_seg(seg_candidates[0])
            if seg_n >= 0 and seg_n != len(r.lesions):
                disagreements += 1
                review_queue.append({
                    "case_id": r.case_id,
                    "field": "lesion_count_mismatch",
                    "raw_value": f"seg says {seg_n}, parsed {len(r.lesions)}",
                })
        log.info(f"lesion-count disagreements: {disagreements}")

    # Stage 6: partition
    splits = partition_dataset(reports, n_test=args.n_test, n_folds=args.n_folds,
                               n_conformal=args.n_conformal, seed=args.seed)

    # Stage 7: write outputs
    jsonl_path = args.out_dir / "neurofusion_369.jsonl"
    with jsonl_path.open("w") as f:
        for r in reports:
            f.write(json.dumps(r.model_dump(mode="json")) + "\n")
    log.info(f"wrote {jsonl_path}")

    splits_path = args.out_dir / "splits.json"
    with splits_path.open("w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"wrote {splits_path}")

    if review_queue:
        rq_path = args.out_dir / "review_queue.csv"
        with rq_path.open("w", newline="") as f:
            fieldnames = sorted({k for item in review_queue for k in item.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_queue)
        log.info(f"wrote {rq_path}  ({len(review_queue)} items need radiologist review)")

    stats_path = args.out_dir / "dataset_stats.md"
    with stats_path.open("w") as f:
        f.write(compute_stats(reports))
    log.info(f"wrote {stats_path}")

    # Success criterion
    pass_rate = len(reports) / len(rows) if rows else 0.0
    log.info(f"Auto-validation pass rate: {pass_rate:.1%}")
    if pass_rate < 0.98:
        log.warning(
            f"pass rate {pass_rate:.1%} < 98% target — radiologist review required before downstream use"
        )


if __name__ == "__main__":
    main()
