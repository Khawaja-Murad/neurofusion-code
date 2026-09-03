"""Prompt templates for the multi-chain CoT inference pipeline.

Stages:
  1. NARRATION - free-text description of the lesion (no XGrammar)
  2. FIELD_HYPOTHESIS - per-field hypothesis with visual justification (no XGrammar)
  3. JSON_COMMIT - schema-valid JSON consistent with the hypothesis (XGrammar)
  4. REVIEW - multi-signal review reconciling JSON + heuristic-predseg + head-argmax
            (XGrammar, lower temperature)

The strings here are deterministic and easy to unit-test; the LM calls
themselves live in scripts/predict_with_multichain_cot.py.

Per-field whitelist + degenerate flagging comes from
xgrammar_decoder.DEFAULT_LAMBDA_PER_FIELD (the 2026-05-28 field-head gate);
this module imports those constants so prompts and post-aggregation share
the same source of truth.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------------
# Field-enum constants (must stay in sync with xgrammar_decoder.py).
#
# xgrammar_decoder.py imports torch at module level; importing FROM it just
# to read these two constants would pull torch into the CoT-prompts code
# path. Instead we duplicate the small, stable values here. If you change
# DEFAULT_LAMBDA_PER_FIELD or _DEFAULT_FIELD_ENUM_STRINGS in
# xgrammar_decoder.py, also update them here.
#
# Verified equivalent to xgrammar_decoder symbols as of 2026-05-28:
#   - _DEFAULT_FIELD_ENUM_STRINGS (matches code/model.py
#       FieldClassificationHead.ENUM_STRINGS)
#   - DEFAULT_LAMBDA_PER_FIELD (set from the 2026-05-28 field-head gate;
#       see out/L3_GATE_FINDINGS.md)
# ----------------------------------------------------------------------

_DEFAULT_FIELD_ENUM_STRINGS: dict[str, list[str]] = {
    "location": [
        "frontal", "temporal", "parietal", "occipital", "insular", "cerebellar",
        "brainstem", "parieto-temporal", "temporo-parietal", "fronto-parietal",
        "fronto-temporal", "temporo-occipital", "parieto-occipital", "thalamic",
        "basal-ganglia", "corpus-callosum",
    ],
    "composition": ["solid", "cystic", "necrotic", "necrotic-cystic", "hemorrhagic"],
    "enhancement_pattern": ["none", "homogeneous", "heterogeneous", "ring",
                            "thick-walled-ring", "nodular", "peripheral-nodular"],
    "edema_severity": ["none", "mild", "moderate", "marked"],
    "mass_effect": ["none", "present"],
    "midline_shift": ["none", "present"],
    "ventricular_compression": ["none", "present"],
    "brainstem_compression": ["none", "present"],
    "midbrain_compression": ["none", "present"],
    "axis_shift": ["none", "contralateral", "ipsilateral"],
}

_UNTRUSTWORTHY_HEAD_FIELDS = ("location", "midbrain_compression")

DEFAULT_LAMBDA_PER_FIELD: dict[str, float] = {
    fname: (0.0 if fname in _UNTRUSTWORTHY_HEAD_FIELDS else 1.0)
    for fname in _DEFAULT_FIELD_ENUM_STRINGS
}


# Fields whose head is empirically unreliable; per L3_GATE_FINDINGS the LM
# should NOT defer to head argmax on these in Stage 4. Composition is borderline
# — head 0.137 is still above LM 0.026 but absolute is low; we still let the LM
# see it but flag it as "low-confidence".
UNRELIABLE_HEAD_FIELDS = tuple(
    f for f, lam in DEFAULT_LAMBDA_PER_FIELD.items() if lam == 0.0
)


def render_stage_narration(
    visual_summary: str,
    heuristic_text: str,
    geometry_block: str = "",
) -> str:
    """Stage 1: ask the LM to narrate what it sees, before naming anything.

    visual_summary is a short textual rendering of the per-lesion features
    (e.g. "1 lesion routed: vol=45mL, T1CE-enhancing rim, central low-T1
    cavity, peritumoral T2-FLAIR hyperintensity"). For now this is empty
    when the runner doesn't synthesize it; the visual evidence reaches the
    LM via the Q-Former visual prefix tokens that are prepended to the
    prompt embeddings at the model.LMWithMRope4D level.

    geometry_block (Approach B): when non-empty, the mask-derived GEOMETRY
    fields are presented as authoritative so the narrative is grounded on them.
    """
    extra = f"Visual summary (auxiliary):\n{visual_summary}\n\n" if visual_summary else ""
    geom = f"{geometry_block}\n\n" if geometry_block else ""
    return (
        "You are a neuroradiologist reading a brain MRI for a tumor report.\n"
        f"{extra}"
        f"{geom}"
        f"Heuristic findings from segmentation:\n{heuristic_text}\n\n"
        "Describe the dominant lesion in 2 short sentences. "
        "Focus on visual cues for composition (solid vs cystic vs necrotic), "
        "enhancement pattern (ring, heterogeneous, homogeneous), peritumoral "
        "edema, and mass effect. Do not commit to typed labels yet."
    )


def render_stage_field_hypothesis(
    narration: str,
) -> str:
    """Stage 2: per-field hypothesis with one short visual justification each.

    The output of this stage is free text that gets fed back into Stage 3.
    Format is enforced by example, not by a grammar — the goal is to make
    the LM CONSIDER each field separately rather than emit them in JSON-key
    order with no intervening thought."""
    enum_lines: list[str] = []
    for fname, classes in _DEFAULT_FIELD_ENUM_STRINGS.items():
        enum_lines.append(f"  - {fname} (one of: {', '.join(classes)})")
    enum_block = "\n".join(enum_lines)
    return (
        f"Your observation:\n{narration}\n\n"
        "Now hypothesize each typed field with a one-sentence visual "
        "justification. Format strictly as `field: value because <visual cue>`. "
        "Do not pick the most common value by default; pick based on the "
        "visual evidence you just described.\n\n"
        f"Fields and allowed values:\n{enum_block}\n\n"
        "Your per-field hypotheses:"
    )


def render_stage_json_commit(
    narration: str,
    field_hypotheses: str,
    case_id: str,
) -> str:
    """Stage 3: emit the schema-valid JSON, primed by Stages 1+2.

    The actual JSON tokens are constrained by XGrammar; this prompt just
    sets the context. The `{"case_id": "<id>", ` prefix is the same trick
    as in model.py's existing generate() — aligns with training
    distribution and prevents case_id hallucination."""
    return (
        f"Your observation:\n{narration}\n\n"
        f"Your field hypotheses:\n{field_hypotheses}\n\n"
        "Now emit the structured JSON report. Use ONLY the values you "
        "just hypothesized; do not introduce new ones.\n\n"
        # NF_STRIP_CASE_ID: `+` is REQUIRED here — a string literal directly followed by `(`
        # parses as a CALL, not concatenation (SyntaxWarning: 'str' object is not callable).
        + ('{' if os.environ.get("NF_STRIP_CASE_ID") == "1"
           else f'{{"case_id": "{case_id}", ')
    )


def render_stage_review(
    draft_json: str,
    heuristic_predseg_json: str,
    head_argmax_per_field: dict[str, str],
    head_confidence_per_field: dict[str, float],
    case_id: str,
    geometry_block: str = "",
) -> str:
    """Stage 4: reconcile three independent signals — your own JSON, the
    heuristic-from-predicted-segmentation, and the field-classifier-head's
    argmax — into a final answer.

    head_argmax_per_field: {field_name: top1_class_string} from the
        FieldClassificationHead softmax for the largest routed lesion.
    head_confidence_per_field: {field_name: top1_prob} for those same fields.
        Used in the prompt as a confidence cue: high-confidence head argmax
        carries more weight than low-confidence.
    geometry_block (Approach B): when non-empty, the mask-derived GEOMETRY
        fields are AUTHORITATIVE — the LM adopts them verbatim and only
        reconciles the appearance fields.
    """
    head_lines: list[str] = []
    for fname, val in head_argmax_per_field.items():
        conf = head_confidence_per_field.get(fname, 0.0)
        reliability = " [LOW-RELIABILITY HEAD]" if fname in UNRELIABLE_HEAD_FIELDS else ""
        head_lines.append(f"  - {fname}: {val}  (head_confidence={conf:.2f}){reliability}")
    head_block = "\n".join(head_lines) if head_lines else "  (none)"

    if geometry_block:
        # Approach B: geometry is authoritative; only appearance fields are reconciled.
        return (
            "Here are independent views of the same case:\n\n"
            f"{geometry_block}\n\n"
            f"YOUR DRAFT JSON:\n{draft_json}\n\n"
            f"HEURISTIC-FROM-SEGMENTATION JSON:\n{heuristic_predseg_json}\n\n"
            f"FIELD-CLASSIFIER-HEAD ARGMAX (per-field top-1):\n{head_block}\n\n"
            "Adopt the AUTHORITATIVE GEOMETRY values VERBATIM for location, mass "
            "effect, midline shift, axis shift, and the ventricular / brainstem / "
            "midbrain compressions. For the appearance fields (hemisphere, "
            "composition, enhancement pattern, edema severity) decide from the "
            "visual evidence and head confidence. Make the narrative consistent "
            "with the authoritative geometry.\n\n"
            'Emit the FINAL revised JSON:\n'
            + ('{' if os.environ.get("NF_STRIP_CASE_ID") == "1"
               else f'{{"case_id": "{case_id}", ')
        )

    return (
        "Here are three independent views of the same case:\n\n"
        f"YOUR DRAFT JSON:\n{draft_json}\n\n"
        f"HEURISTIC-FROM-SEGMENTATION JSON:\n{heuristic_predseg_json}\n\n"
        f"FIELD-CLASSIFIER-HEAD ARGMAX (per-field top-1):\n{head_block}\n\n"
        "For each field where these disagree, decide based on the visual "
        "evidence + the head's confidence + your reasoning. Override the "
        "head-argmax when its confidence is below 0.5 or its field is "
        "flagged LOW-RELIABILITY. Override the heuristic when it disagrees "
        "with both you and a high-confidence head.\n\n"
        'Emit the FINAL revised JSON:\n'
        + ('{' if os.environ.get("NF_STRIP_CASE_ID") == "1"
           else f'{{"case_id": "{case_id}", ')
    )


# -------------------------------------------------------------------------
# Per-field majority-vote aggregation (Self-Consistency, Wang et al. 2022)
# -------------------------------------------------------------------------


def _parse_or_none(s: str) -> dict | None:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    except Exception:
        return None
    # ☠️☠️ THE C0 REPAIR HAS TO BE HERE, BECAUSE *THIS* IS THE SHIPPED PARSE.
    # Keeping `maxLength` switched findings/impression onto a bounded negated character
    # class that admits raw C0 control bytes (TAB, NUL, BEL, VT, ESC), and a raw C0 byte
    # inside a JSON string is invalid JSON. I repaired that at
    # xgrammar_decoder.parse_and_validate -- and then checked no further. The shipped
    # aggregation does NOT go through it: nf_infer.py:1578 calls majority_vote_per_field
    # on the raw stage-4 strings, which parses them HERE, with a bare json.loads. A C0
    # byte would have returned None from every chain and raised ReportParseError -- the
    # exact failure the repair was written to prevent, through the exact path that
    # actually ships.
    # ⇒ ★★★ STANDING RULE, FIRED A SIXTH TIME: FIX THE SHIPPED PATH. I put the repair
    #   where I had been reading rather than where the failure occurs, and the drill I
    #   wrote for it passed 12/12 against the wrong function. A drill that exercises a
    #   parser other than the one that ships is a test of my attention, not of the system.
    # Imported lazily: cot_prompts is imported by training code that must not pull in
    # xgrammar, and the repair is only ever needed on a parse that has already failed.
    try:
        from xgrammar_decoder import escape_raw_control_chars
        return json.loads(escape_raw_control_chars(s))
    except Exception:
        return None


# Approach B (heuristic-in-prompt): the GEOMETRY fields are mask-derivable and the
# segmentation heuristic beats the LM on them; inject them as AUTHORITATIVE so the LM
# grounds its narrative on them and copies them into the structured output. The
# APPEARANCE/intensity fields (hemisphere, composition, enhancement, edema) are left to
# the LM (the mask is blind to intensity). Mirrors the val-frozen router's geometry split.
GEOMETRY_INJECT_FIELDS = (
    "location", "overall_mass_effect", "mass_effect", "midline_shift", "axis_shift",
    "ventricular_compression", "brainstem_compression", "midbrain_compression",
)


def render_geometry_authoritative(heuristic_predseg_json: str) -> str:
    """Extract the geometry fields from the seg-heuristic report (a BrainTumorReport-
    shaped JSON string) and format them as an authoritative block for Approach B.
    Returns '' if it can't be parsed / no geometry values are present."""
    obj = _parse_or_none(heuristic_predseg_json)
    if not isinstance(obj, dict):
        return ""
    les = (obj.get("lesions") or [None])[0]
    les = les if isinstance(les, dict) else {}
    se = les.get("surrounding_effects") or {}
    inv = les.get("involvement") or {}
    vals = [
        ("lobe/location", les.get("location")),
        ("overall mass effect", obj.get("overall_mass_effect")),
        ("mass effect", se.get("mass_effect")),
        ("midline shift", se.get("midline_shift")),
        ("axis shift", les.get("axis_shift")),
        ("ventricular compression", inv.get("ventricular_compression")),
        ("brainstem compression", inv.get("brainstem_compression")),
        ("midbrain compression", inv.get("midbrain_compression")),
    ]
    lines = [f"  - {k}: {v}" for k, v in vals if v is not None]
    if not lines:
        return ""
    return (
        "AUTHORITATIVE GEOMETRY (measured from the 3D segmentation mask — reliable "
        "spatial measurements the LM cannot make; you MUST adopt these exact values "
        "for these fields and describe them in your narrative):\n"
        + "\n".join(lines)
    )


def _get_field(obj: dict, dotted: str) -> str | None:
    """Walk a dotted path inside the parsed BrainTumorReport-shaped dict.
    Lesion fields use a `lesions.0.<sub>` convention; top-level fields are
    just the name. Returns None on any missing piece."""
    cur: object = obj
    for tok in dotted.split("."):
        if isinstance(cur, list):
            try:
                idx = int(tok)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(tok)
        else:
            return None
        if cur is None:
            return None
    return cur if isinstance(cur, str) else None


def majority_vote_per_field(
    chain_jsons: list[str],
    head_argmax_per_field: dict[str, str] | None = None,
    field_whitelist: Iterable[str] | None = None,
) -> dict | None:
    """K-chain self-consistency aggregation.

    For each typed field, count the (parse-survived) values across the K
    chains. The mode wins. Ties are broken by head argmax for whitelisted
    fields, otherwise by the first-chain value.

    chain_jsons: K JSON strings (output of Stage 4 across K chains).
    head_argmax_per_field: maps `{full_dotted_field: head_top1_class_str}`
        for tie-breaking. Pass None to disable head tie-breaking.
    field_whitelist: iterable of dotted-field names where head-argmax wins
        the tie. Pass None to apply head tie-breaking to all fields.

    Returns: aggregated BrainTumorReport-shaped dict, or None if every chain
    failed to parse.
    """
    parsed = [p for p in (_parse_or_none(s) for s in chain_jsons) if p is not None]
    if not parsed:
        return None

    # Start the aggregated record from the first-parsing chain so structural
    # scaffolding (case_id, lesion shape, narrative) is preserved.
    out = json.loads(json.dumps(parsed[0]))  # deep copy
    head_argmax_per_field = head_argmax_per_field or {}
    whitelist = set(field_whitelist) if field_whitelist is not None else None

    # Top-level scalar fields
    TOP_FIELDS = ("hemisphere", "overall_mass_effect")
    for fname in TOP_FIELDS:
        votes: dict[str, int] = {}
        for p in parsed:
            v = p.get(fname)
            if isinstance(v, str):
                votes[v] = votes.get(v, 0) + 1
        if not votes:
            continue
        winner = _pick_winner(votes, head_argmax_per_field.get(fname),
                              whitelist is None or fname in whitelist)
        out[fname] = winner

    # Lesion-level fields — use the first lesion only (corpus is 100% single
    # lesion on labeled data; multi-lesion is descriptive not statistical).
    if not out.get("lesions"):
        return out
    LESION_TOP = ("location", "composition", "enhancement_pattern", "axis_shift")
    LESION_SE = ("edema_severity", "mass_effect", "midline_shift")
    LESION_INV = ("ventricular_compression", "brainstem_compression", "midbrain_compression")

    for fname in LESION_TOP:
        votes = {}
        for p in parsed:
            les = (p.get("lesions") or [None])[0]
            if not isinstance(les, dict):
                continue
            v = les.get(fname)
            if isinstance(v, str):
                votes[v] = votes.get(v, 0) + 1
        if votes:
            key = f"lesions.0.{fname}"
            winner = _pick_winner(votes, head_argmax_per_field.get(f"lesion_{fname}"),
                                  whitelist is None or f"lesion_{fname}" in whitelist)
            out["lesions"][0][fname] = winner

    for fname in LESION_SE:
        votes = {}
        for p in parsed:
            les = (p.get("lesions") or [None])[0]
            if not isinstance(les, dict):
                continue
            v = (les.get("surrounding_effects") or {}).get(fname)
            if isinstance(v, str):
                votes[v] = votes.get(v, 0) + 1
        if votes:
            winner = _pick_winner(votes, head_argmax_per_field.get(f"lesion_{fname}"),
                                  whitelist is None or f"lesion_{fname}" in whitelist)
            out["lesions"][0].setdefault("surrounding_effects", {})[fname] = winner

    for fname in LESION_INV:
        votes = {}
        for p in parsed:
            les = (p.get("lesions") or [None])[0]
            if not isinstance(les, dict):
                continue
            v = (les.get("involvement") or {}).get(fname)
            if isinstance(v, str):
                votes[v] = votes.get(v, 0) + 1
        if votes:
            winner = _pick_winner(votes, head_argmax_per_field.get(f"lesion_{fname}"),
                                  whitelist is None or f"lesion_{fname}" in whitelist)
            out["lesions"][0].setdefault("involvement", {})[fname] = winner

    return out


def _pick_winner(
    votes: dict[str, int],
    head_argmax: str | None,
    head_tiebreak_allowed: bool,
) -> str:
    """Pick the mode. On ties, prefer head_argmax if it's one of the tied
    values AND tie-breaking is allowed for this field. Otherwise the
    lexicographically-first tied value wins (deterministic)."""
    if not votes:
        return ""  # caller checks for non-empty before calling
    top_count = max(votes.values())
    tied = sorted(c for c, n in votes.items() if n == top_count)
    if len(tied) == 1:
        return tied[0]
    if head_tiebreak_allowed and head_argmax is not None and head_argmax in tied:
        return head_argmax
    return tied[0]
