"""
enum_mappings.py — Canonical string → enum value lookup tables.

These dicts are consumed by preprocess_reports.py to canonicalize the raw
clinician-annotated CSV strings into the exact values defined in schema.py.

Maintenance rules:
  - Keys are lowercased, stripped, whitespace-collapsed (the _norm() transform).
  - Values are the exact strings from schema.py enums.
  - Add new keys here first; never silently widen an enum in schema.py.
  - Tested by test_enum_mappings.py (keys must round-trip through _norm).
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Hemisphere  (schema.Hemisphere)
# ─────────────────────────────────────────────────────────────
HEMISPHERE_MAP: dict[str, str] = {
    "left": "left",
    "right": "right",
    "both": "bilateral",
    "bilateral": "bilateral",
    "both hemispheres": "bilateral",
    "both sides": "bilateral",
}

# ─────────────────────────────────────────────────────────────
# Location  (schema.Location)
# Derived from the full set of unique values seen in the 256-row CSV.
# ─────────────────────────────────────────────────────────────
LOCATION_MAP: dict[str, str] = {
    # ── simple lobes ──────────────────────────────────────────
    "frontal lobe": "frontal",
    "frontal": "frontal",
    "temporal lobe": "temporal",
    "temporal": "temporal",
    "parietal lobe": "parietal",
    "parietal": "parietal",
    "posterior parietal lobe": "parietal",
    "occipital lobe": "occipital",
    "occipital": "occipital",
    "insular": "insular",
    "insula": "insular",
    "insular lobe": "insular",
    # ── cerebellar ────────────────────────────────────────────
    "cerebellar hemisphere": "cerebellar",
    "cerebellum": "cerebellar",
    "cerebellar": "cerebellar",
    # ── brainstem / midbrain ──────────────────────────────────
    "brainstem": "brainstem",
    "brain stem": "brainstem",
    "brain-stem": "brainstem",
    "midbrain": "brainstem",      # midbrain is anatomically part of brainstem
    # ── parieto-temporal ──────────────────────────────────────
    "parieto-temporal lobe": "parieto-temporal",
    "parieto-temporal": "parieto-temporal",
    "parietotemporal lobe": "parieto-temporal",
    "parietotemporal": "parieto-temporal",
    # ── temporo-parietal ──────────────────────────────────────
    "temporo-parietal lobe": "temporo-parietal",
    "temporo- parietal lobe": "temporo-parietal",
    "temporo-parietal": "temporo-parietal",
    "temporoparietal lobe": "temporo-parietal",
    "temporo- parietal lobe and basal ganglia": "temporo-parietal",  # primary region
    # ── fronto-parietal ───────────────────────────────────────
    "fronto-parietal lobes": "fronto-parietal",
    "fronto-parietal lobe": "fronto-parietal",
    "fronto-parietal": "fronto-parietal",
    "frontoparietal": "fronto-parietal",
    "fronto-parietal and temporal lobe": "fronto-parietal",
    # ── fronto-temporal ───────────────────────────────────────
    "fronto-temporal lobe": "fronto-temporal",
    "fronto-temporal": "fronto-temporal",
    "frontotemporal": "fronto-temporal",
    # ── temporo-occipital ─────────────────────────────────────
    "temporo-occipital lobe": "temporo-occipital",
    "temporo-occipital": "temporo-occipital",
    # ── parieto-occipital ─────────────────────────────────────
    "parieto-occipital lobe": "parieto-occipital",
    "parieto- occipital lobe": "parieto-occipital",
    "parieto-occipital": "parieto-occipital",
    "occipito-parietal lobe": "parieto-occipital",
    # ── deep structures ───────────────────────────────────────
    "thalamus": "thalamic",
    "thalamic": "thalamic",
    "basal ganglia": "basal-ganglia",
    "corpus callosum": "corpus-callosum",
    # ── catch-all (unusual compound / generics) ───────────────
    "centrum semiovale": "other",
    "cerebral hemisphere": "other",
    "fronto-parieto-temporal lobes": "other",   # not in schema v1.0 — flag
    "all": "other",
    "parietal lobe, brainstem": "other",        # multi-region; manual review
}

# ─────────────────────────────────────────────────────────────
# Composition  (schema.Composition)
# ─────────────────────────────────────────────────────────────
COMPOSITION_MAP: dict[str, str] = {
    "necrotic tumor": "necrotic",
    "necrotic": "necrotic",
    "necrotic/ cystic tumor": "necrotic-cystic",
    "necrotic/cystic tumor": "necrotic-cystic",
    "necrotic cystic tumor": "necrotic-cystic",
    "necrotic-cystic tumor": "necrotic-cystic",
    "necrotic-cystic": "necrotic-cystic",
    "cystic tumor": "cystic",
    "cystic": "cystic",
    "solid tumor": "solid",
    "solid": "solid",
    # 'none' in the CSV means no necrosis/cyst → solid enhancing mass
    "none": "solid",
    "hemorrhagic": "hemorrhagic",
    "hemorrhagic tumor": "hemorrhagic",
    # single-lesion prefix (multi-lesion split handles the full string)
    "lesion 1: necrotic tumor": "necrotic",
}

# ─────────────────────────────────────────────────────────────
# Enhancement pattern  (schema.EnhancementPattern)
# 38 unique raw strings observed → 7 canonical values.
# Mapping strategy:
#   peripheral-only strings with wall/thick → thick-walled-ring
#   peripheral + nodular → peripheral-nodular
#   peripheral + ring (no thick wall) → ring
#   heterogeneous (any modifier) → heterogeneous
#   simple ring → ring
#   homogeneous → homogeneous
#   explicit none → none
# ─────────────────────────────────────────────────────────────
ENHANCEMENT_MAP: dict[str, str] = {
    # ── heterogeneous ─────────────────────────────────────────
    "heterogeneous enhancement": "heterogeneous",
    "heterogenous enhancement": "heterogeneous",
    "heterogeneous peripheral enhancement": "heterogeneous",
    "heterogeneous peripheral irregular enhancement": "heterogeneous",
    "heterogeneous peripheral wall enhancement": "heterogeneous",
    "heterogenous peripheral enhancement": "heterogeneous",
    "heterogenous peripheral irregular enhancement": "heterogeneous",
    "peripheral enhancement": "heterogeneous",
    "peripheral irregular enhancement": "heterogeneous",
    # ── ring ──────────────────────────────────────────────────
    "peripheral ring enhancement": "ring",
    "ring enhancement": "ring",
    "peripheral irregular ring enhancement": "ring",
    "heterogeneous peripheral ring enhancement": "ring",
    "heterogenous peripheral ring enhancement": "ring",
    "uniform wall enhancement": "ring",
    "peripheral uniform wall enhancement": "ring",
    # ── peripheral-nodular ────────────────────────────────────
    "peripheral ring enhancement nodular enhancement": "peripheral-nodular",
    "nodular enhancement": "nodular",
    "nodular": "nodular",
    # ── thick-walled-ring ─────────────────────────────────────
    "irregular wall enhancement": "thick-walled-ring",
    "peripheral wall enhancement": "thick-walled-ring",
    "peripheral irregular wall enhancement": "thick-walled-ring",
    "peripheral irregular wall ring enhancement": "thick-walled-ring",
    "heterogenous thick wall enhancement": "thick-walled-ring",
    "heterogeneous thick wall enhancement": "thick-walled-ring",
    "thick wall enhancement": "thick-walled-ring",
    "irregular ring wall enhancement": "thick-walled-ring",
    "peripheral irregular ring wall enhancement": "thick-walled-ring",
    # ── more heterogeneous variants (2026-05-19 recovery) ─────
    "heterogenous irregular enhancement": "heterogeneous",
    "heterogeneous irregular enhancement": "heterogeneous",
    "heterogenous cystic contrast enhancement": "heterogeneous",
    "heterogeneous cystic contrast enhancement": "heterogeneous",
    "heterogenous cystic enhancement": "heterogeneous",
    "heterogeneous cystic enhancement": "heterogeneous",
    "irregular enhancement": "heterogeneous",
    "irregular contrast enhancement": "heterogeneous",
    "heterogenous contrast enhancement": "heterogeneous",
    "heterogeneous contrast enhancement": "heterogeneous",
    "contrast enhancement": "heterogeneous",
    "gyriform enhancement": "heterogeneous",
    "gyriform": "heterogeneous",
    "thick ring enhancement": "thick-walled-ring",
    "irregular thick ring enhancement": "thick-walled-ring",
    "peripheral irregular thick ring enhancement": "thick-walled-ring",
    # parieto-temporal lobe is a TR93 data-entry error (location text leaked into
    # enhancement column). Map to most-likely enhancement based on Report.
    "parieto-temporal lobe": "thick-walled-ring",
    # ── homogeneous ───────────────────────────────────────────
    "homogeneous enhancement": "homogeneous",
    "homogeneous": "homogeneous",
    "uniform enhancement": "homogeneous",
    "homogenous enhancement": "homogeneous",
    "homogenous": "homogeneous",
    # ── none ──────────────────────────────────────────────────
    "no enhancement": "none",
    "none": "none",
}

# ─────────────────────────────────────────────────────────────
# Edema severity  (schema.EdemaSeverity)
# Used inside parse_surrounding_effects() to pattern-match within the
# bundled "Surrounding Effects" field string.
# Keys are substrings (checked with `key in norm`).
# ─────────────────────────────────────────────────────────────
EDEMA_MAP: dict[str, str] = {
    "marked edema": "marked",
    "moderate edema": "moderate",
    "mild edema": "mild",
    "no edema": "none",
    # bare ordinals (fallback after compound miss)
    "marked": "marked",
    "moderate": "moderate",
    "mild": "mild",
}

# ─────────────────────────────────────────────────────────────
# Axis shift  (schema.AxisShift)
# ─────────────────────────────────────────────────────────────
AXIS_SHIFT_MAP: dict[str, str] = {
    "no axis shift": "none",
    "no shift": "none",
    "no midline shift": "none",
    "both hemispheres involved": "none",    # bilateral → no lateral shift
    "contralateral midline shift": "contralateral",
    "contralateral": "contralateral",
    "midline shift": "contralateral",       # unqualified shift → contralateral by convention
    "ipsilateral": "ipsilateral",
    "ipsilateral shift": "ipsilateral",
    "ipsilateral midline shift": "ipsilateral",
}

# ─────────────────────────────────────────────────────────────
# Involvement — keyword sets for parse_involvement()
# Keys match Involvement field names; values are substring lists.
# ─────────────────────────────────────────────────────────────
INVOLVEMENT_KEYWORDS: dict[str, list[str]] = {
    "ventricular_compression": [
        "ventricle",
        "ventricular",
        "lateral ventricle",
        "third ventricle",
        "fourth ventricle",
        "horn of",          # e.g. "occipital horn of lateral ventricle"
        "occipital horn",
        "anterior horn",
        "corpus callosum",  # callosal compression implies ventricular proximity
    ],
    "brainstem_compression": [
        "brainstem",
        "brain stem",
        "brain-stem",
    ],
    "midbrain_compression": [
        "midbrain",
        "mid-brain",
        "mid brain",
    ],
}

# Strings that indicate NO involvement of any structure.
NO_COMPRESSION_KEYWORDS: list[str] = [
    "no compression",
    "no ventricular compression",
    "no brainstem compression",
    "no involvement",
    "no ventricular",
]
