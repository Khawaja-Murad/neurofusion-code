"""
eval.py — NeuroFusion v4 evaluation pipeline.

Computes the metric panel from methodology v4 §5:

    Faithfulness (PRIMARY for EMNLP):
      - RadGraph-F1            (Yu et al. NEJM AI 2023) via stanford radgraph
      - RaTEScore              (Zhao et al. EMNLP 2024) via PyPI rate_score
      - GREEN                  (Ostmeier et al. Findings of EMNLP 2024) — LLM-judge
      - RadFact                (Bannur et al. 2024 / microsoft/RadFact) — LLM-judge
      - Neuro-entity F1        (our contribution, optional)

    Structure (PRIMARY):
      - Schema adherence       (% predictions parse as valid BrainTumorReport)
      - Section completeness   (% required fields populated per case)
      - Per-field macro-F1     (HEADLINE structured metric)
      - Lesion-count accuracy  (exact, ±1)
      - Hungarian per-lesion field-F1 (multi-lesion)

    NLG (SECONDARY):
      - BLEU 1..4 + corpus-BLEU
      - ROUGE-L
      - METEOR
      - CIDEr (skipped — coco_caption dependency is heavy; left as TODO)
      - BERTScore

    Statistical hygiene:
      - Bootstrap 95% CIs (n=1000) on every metric
      - Per-fold mean ± std for 5-fold CV reporting

Inputs:
    --pred-jsonl    one line per case: {"case_id": str, "pred_json": str, "pred_findings": str?, "pred_impression": str?}
    --gt-jsonl      one line per case: BrainTumorReport JSON (matches schema.py)
    --out-dir       writes metrics.json + report.md + per-case CSV

Quick start (mock judge for development; replace with real Llama-3-70B in production):
    python eval.py --pred-jsonl runs/fold0/predictions.jsonl --gt-jsonl /data/neurofusion_369.jsonl \
                   --out-dir runs/fold0/eval --judge mock

For real RadFact/GREEN evaluation:
    --judge openai      requires OPENAI_API_KEY  (uses gpt-4 by default)
    --judge anthropic   requires ANTHROPIC_API_KEY  (uses claude-sonnet-4 by default)
    --judge hf          requires --judge-model meta-llama/Llama-3.3-70B-Instruct (local HF)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

# Restore transformers.AdamW (removed in 4.42+) so radgraph's allennlp dependency
# imports cleanly. The class was functionally identical to torch.optim.AdamW.
# Must run before any radgraph import. Made lazy 2026-05-28 so the collapse-
# metric helpers in this module can be imported in a CI environment that has
# numpy + pydantic but not torch/transformers (tests/test_collapse_metric.py).
def _ensure_adamw_shim() -> None:
    import torch.optim
    import transformers
    if not hasattr(transformers, "AdamW"):
        transformers.AdamW = torch.optim.AdamW


from schema import BrainTumorReport, Lesion, strict_content

log = logging.getLogger("neurofusion.eval")


# ===========================================================================
# JUDGE PROTOCOL — swappable LLM backend for RadFact / GREEN
# ===========================================================================


class JudgeFn(Protocol):
    """A judge takes a prompt string and returns the judge's response string.
    Used for RadFact and GREEN, both of which prompt an LLM to score generated reports.
    """
    def __call__(self, prompt: str, max_tokens: int = 1024) -> str: ...


def make_mock_judge(seed: int = 0) -> JudgeFn:
    """Deterministic mock judge for dev. Returns plausible-looking scores so
    the pipeline can be tested end-to-end without an LLM endpoint."""
    rng = np.random.default_rng(seed)

    def judge(prompt: str, max_tokens: int = 1024) -> str:
        h = abs(hash(prompt)) % (2**32)
        local = np.random.default_rng(h)  # per-prompt determinism
        # Emit a JSON-like response with reasonable scores
        n_findings = max(1, prompt.count("FINDING") or prompt.count("statement"))
        scores = local.choice(["TRUE", "FALSE"], size=n_findings, p=[0.7, 0.3])
        green_score = float(local.uniform(0.4, 0.85))
        return json.dumps({
            "logical_precision": float((scores == "TRUE").mean()),
            "logical_recall": float(local.uniform(0.4, 0.8)),
            "green_overall": green_score,
            "errors": int(local.integers(0, 4)),
            "_per_finding": [s for s in scores.tolist()],
        })

    return judge


def make_openai_judge(model: str = "gpt-4-turbo", api_key: str | None = None) -> JudgeFn:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("pip install openai required for --judge openai")
    client = OpenAI(api_key=api_key)

    def judge(prompt: str, max_tokens: int = 1024) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful radiology evaluator. Always reply with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or "{}"

    return judge


def make_anthropic_judge(model: str = "claude-sonnet-4-20250514", api_key: str | None = None) -> JudgeFn:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("pip install anthropic required for --judge anthropic")
    client = anthropic.Anthropic(api_key=api_key)

    def judge(prompt: str, max_tokens: int = 1024) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system="You are a careful radiology evaluator. Always reply with valid JSON.",
        )
        return msg.content[0].text  # type: ignore

    return judge


def make_hf_judge(model: str = "meta-llama/Llama-3.3-70B-Instruct", device: str = "cuda") -> JudgeFn:
    """Local HuggingFace judge. Loads a Llama-3-70B-class model into memory.
    For 70B you'll need 2x80GB or 8-bit quantization. Use this on Narval / Cedar."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        import torch  # type: ignore
    except ImportError:
        raise RuntimeError("pip install transformers torch required for --judge hf")
    log.info(f"loading HF judge: {model}")
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=True)
    mdl.eval()

    def judge(prompt: str, max_tokens: int = 1024) -> str:
        msgs = [
            {"role": "system", "content": "You are a careful radiology evaluator. Always reply with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        ids = tok.apply_chat_template(msgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return judge


# ===========================================================================
# SCHEMA-ADHERENCE + STRUCTURE METRICS
# ===========================================================================


def schema_adherence(predictions: list[str]) -> dict[str, float]:
    """Three-tier validity metric.

    - json_validity:       parses as JSON
    - structural_validity: parses as BrainTumorReport with content validators OFF
                           (XGrammar can enforce structure but not string-content
                           rules like "findings must mention enhancement")
    - schema_adherence:    full Pydantic validation including content validators
                           (kept as the legacy headline metric; same value as
                           full_validity).
    - full_validity:       alias of schema_adherence for clarity in reports
    """
    n_total = len(predictions)
    n_json = 0
    n_structural = 0
    n_full = 0
    for p in predictions:
        try:
            json.loads(p)
        except json.JSONDecodeError:
            continue
        n_json += 1
        with strict_content(False):
            try:
                BrainTumorReport.model_validate_json(p)
                n_structural += 1
            except Exception:
                pass
        try:
            BrainTumorReport.model_validate_json(p)
            n_full += 1
        except Exception:
            pass
    return {
        "schema_adherence": n_full / max(n_total, 1),
        "full_validity": n_full / max(n_total, 1),
        "structural_validity": n_structural / max(n_total, 1),
        "json_validity": n_json / max(n_total, 1),
        "n_total": n_total,
        "n_valid": n_full,
    }


_REQUIRED_FIELDS = ["hemisphere", "differential_diagnosis", "overall_mass_effect", "lesions", "findings", "impression"]
_REQUIRED_LESION_FIELDS = ["location", "composition", "enhancement_pattern", "surrounding_effects", "involvement", "axis_shift"]


def section_completeness(predictions: list[str]) -> dict[str, float]:
    """Fraction of required top-level + per-lesion fields populated.
    Operates on the raw JSON dict (does NOT require schema validation), so partially
    valid outputs still get measured."""
    field_present_counts: dict[str, int] = {f: 0 for f in _REQUIRED_FIELDS + _REQUIRED_LESION_FIELDS}
    total_with_lesions = 0
    total = 0
    for p in predictions:
        try:
            d = json.loads(p)
        except json.JSONDecodeError:
            total += 1
            continue
        total += 1
        for f in _REQUIRED_FIELDS:
            if f in d and d[f] not in (None, "", []):
                field_present_counts[f] += 1
        if isinstance(d.get("lesions"), list) and len(d["lesions"]) > 0:
            total_with_lesions += 1
            lesion = d["lesions"][0]
            for f in _REQUIRED_LESION_FIELDS:
                if f in lesion and lesion[f] not in (None, "", []):
                    field_present_counts[f] += 1
    metrics = {f"completeness/{f}": (field_present_counts[f] / max(total, 1)) for f in _REQUIRED_FIELDS}
    for f in _REQUIRED_LESION_FIELDS:
        metrics[f"completeness/lesion_{f}"] = field_present_counts[f] / max(total_with_lesions, 1)
    metrics["completeness/overall"] = float(np.mean(list(metrics.values())))
    return metrics


# ===========================================================================
# PER-FIELD MACRO-F1 (HEADLINE STRUCTURED METRIC)
# ===========================================================================


_FIELD_ACCESSORS: dict[str, Callable[[BrainTumorReport, int], str | None]] = {
    "hemisphere": lambda r, k: r.hemisphere,
    "overall_mass_effect": lambda r, k: r.overall_mass_effect,
    "lesion_location": lambda r, k: r.lesions[k].location if k < len(r.lesions) else None,
    "lesion_composition": lambda r, k: r.lesions[k].composition if k < len(r.lesions) else None,
    "lesion_enhancement_pattern": lambda r, k: r.lesions[k].enhancement_pattern if k < len(r.lesions) else None,
    "lesion_edema_severity": lambda r, k: r.lesions[k].surrounding_effects.edema_severity if k < len(r.lesions) else None,
    "lesion_mass_effect": lambda r, k: r.lesions[k].surrounding_effects.mass_effect if k < len(r.lesions) else None,
    "lesion_midline_shift": lambda r, k: r.lesions[k].surrounding_effects.midline_shift if k < len(r.lesions) else None,
    "lesion_ventricular_compression": lambda r, k: r.lesions[k].involvement.ventricular_compression if k < len(r.lesions) else None,
    "lesion_brainstem_compression": lambda r, k: r.lesions[k].involvement.brainstem_compression if k < len(r.lesions) else None,
    "lesion_midbrain_compression": lambda r, k: r.lesions[k].involvement.midbrain_compression if k < len(r.lesions) else None,
    "lesion_axis_shift": lambda r, k: r.lesions[k].axis_shift if k < len(r.lesions) else None,
}


# ---------------------------------------------------------------------------
# SCORING-BASIS OPTIONS (opt-in; defaults reproduce every published number)
#
# ☠️ READ BEFORE TOUCHING. Every number this project has published was computed
# on the DEFAULT basis below. Both knobs here default to the historical
# behaviour and MUST stay that way; they exist so a caller can *explicitly*
# ask for the corrected basis and get a number that is labelled as such.
# tests/test_eval_fixes.py::test_default_path_is_bit_identical_to_golden guards this.
#
# (A) exclude_degenerate — two lesion fields (brainstem/midbrain compression)
#     are SINGLE-CLASS ('none') on the in-domain n=39 test split, and
#     scripts/heuristic_pseudo_label.py hardcodes 'none' for both, yet each
#     scores ~0.93-1.00 macro-F1. Averaging them into the unweighted 12-field
#     mean adds a near-constant ~+0.09..+0.12 to EVERY system and adds MORE to
#     weaker systems, compressing the gaps we are trying to measure.
#     The canonical exclusion list lives in out/field_subset_robustness.json
#     under "excluded_degenerate" — we READ it rather than re-deriving a rule.
#
# (B) penalize_unmatched — the over-prediction penalty was calibrated on the
#     in-domain split where all 39/39 cases have EXACTLY ONE GT lesion
#     (verified). On multi-lesion cohorts (BraTS-MET) a system that correctly
#     enumerates lesions the GT *report* does not list eats one false-positive
#     row per field per unmatched lesion. Turning it off is the right basis for
#     a cohort whose GT report under-enumerates relative to the GT mask.
# ---------------------------------------------------------------------------

#: Fallback used only if out/field_subset_robustness.json cannot be read.
_DEGENERATE_FIELDS_FALLBACK: tuple[str, ...] = (
    "lesion_brainstem_compression",
    "lesion_midbrain_compression",
)

#: Canonical banked artifact carrying the "excluded_degenerate" list.
FIELD_SUBSET_ROBUSTNESS_JSON = Path(__file__).resolve().parent / "out" / "field_subset_robustness.json"

#: Env-var names. UNSET == historical/default behaviour, always.
ENV_EXCLUDE_DEGENERATE = "NF_EVAL_EXCLUDE_DEGENERATE"
ENV_PENALIZE_UNMATCHED = "NF_EVAL_PENALIZE_UNMATCHED"


def load_degenerate_fields(path: str | Path | None = None) -> tuple[str, ...]:
    """Return the canonical degenerate-field exclusion list.

    Source of truth is out/field_subset_robustness.json's `excluded_degenerate`
    key (which already scores this correctly); we reuse it rather than
    inventing a second rule that could drift. Falls back to
    _DEGENERATE_FIELDS_FALLBACK with a warning if the file is unreadable."""
    p = Path(path) if path is not None else FIELD_SUBSET_ROBUSTNESS_JSON
    try:
        d = json.loads(p.read_text())
        fields = d.get("excluded_degenerate")
        if isinstance(fields, list) and fields and all(isinstance(x, str) for x in fields):
            return tuple(fields)
        log.warning(f"{p} has no usable 'excluded_degenerate' list; using fallback")
    except Exception as e:  # missing file, bad JSON, permissions…
        log.warning(f"could not read {p} ({e}); using fallback degenerate-field list")
    return _DEGENERATE_FIELDS_FALLBACK


#: Arming token for the env channel. See _env_flag's SEV-1 note.
ENV_BASIS_OPTIN = "NF_EVAL_BASIS_OPTIN"


def _parse_bool(name: str, raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


def _env_flag(name: str, default: bool) -> bool:
    """Read a basis env var, FAILING CLOSED unless the process armed it.

    ☠️ SEV-1, found by the critic pass 2026-08-05 and fixed here. `scoring_basis`
    provenance is written ONLY by evaluate_all, but ~25 call sites bypass
    evaluate_all and write their own artifacts — including scripts/eval_per_field.py,
    which generated out/per_field_test_table6*.json (PAPER TABLE 6). Demonstrated:
    exporting NF_EVAL_EXCLUDE_DEGENERATE=1 moved that artifact 0.305141 -> 0.280455
    with a BYTE-IDENTICAL key set and no record of why. `--export=ALL` is the sbatch
    default AND is passed explicitly by ~18 launchers, so a single login-shell export
    reaches every job. Under HEAD this was impossible; the option channel introduced it.

    ⇒ A basis var set WITHOUT the arming token is a hard error, not a silent
    re-scoring. Failing closed converts an invisible number change into an
    unmissable crash. The CLI flags and EvalConfig remain the ordinary route and
    are unaffected."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    armed_raw = os.environ.get(ENV_BASIS_OPTIN)
    armed = bool(armed_raw) and armed_raw.strip() != "" and _parse_bool(ENV_BASIS_OPTIN, armed_raw)
    if not armed:
        raise RuntimeError(
            f"{name} is set in the environment but {ENV_BASIS_OPTIN} is not. "
            f"A scoring-basis env var changes published numbers and most eval "
            f"scripts record no basis provenance, so it will not be applied "
            f"implicitly. Either unset {name} (the published basis), or set "
            f"{ENV_BASIS_OPTIN}=1 in THIS process to state that a non-published "
            f"basis is intended. Prefer the explicit CLI flags."
        )
    return _parse_bool(name, raw)


def resolve_exclude_degenerate(
    arg: bool | Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalise the exclude_degenerate option to a concrete field tuple.

    arg=None   -> read ENV_EXCLUDE_DEGENERATE, which DEFAULTS TO OFF -> ()
    arg=False  -> ()                      (DEFAULT / published basis)
    arg=True   -> load_degenerate_fields()
    arg=list   -> exactly those field names
    An empty tuple means "aggregate over all fields", i.e. today's behaviour."""
    if arg is None:
        arg = _env_flag(ENV_EXCLUDE_DEGENERATE, False)
        if arg:
            log.warning(
                f"NON-DEFAULT SCORING BASIS: {ENV_EXCLUDE_DEGENERATE} is set — the "
                "aggregate EXCLUDES degenerate fields and is NOT comparable to any "
                "previously published NeuroFusion number."
            )
    if arg is False:
        return ()
    if arg is True:
        return load_degenerate_fields()
    return tuple(arg)


def resolve_penalize_unmatched(arg: bool | None = None) -> bool:
    """Normalise the penalize_unmatched option.

    arg=None -> read ENV_PENALIZE_UNMATCHED, which DEFAULTS TO True (published
    basis). arg=True/False -> used verbatim."""
    if arg is None:
        val = _env_flag(ENV_PENALIZE_UNMATCHED, True)
        if val is not True:
            log.warning(
                f"NON-DEFAULT SCORING BASIS: {ENV_PENALIZE_UNMATCHED} disables the "
                "over-prediction penalty — NOT comparable to any previously "
                "published NeuroFusion number."
            )
        return val
    return bool(arg)


def _safe_parse(p: str) -> BrainTumorReport | None:
    try:
        return BrainTumorReport.model_validate_json(p)
    except Exception:
        return None


def _collect_per_field_pairs(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    use_hungarian: bool = True,
    penalize_unmatched: bool | None = True,
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Build the {field: [(pred, gt), ...]} dict used by per_field_macro_f1 and
    by compute_collapse_metrics. Factored out so both metrics see the same pairs.
    Predicted lesions left UNMATCHED by Hungarian (over-prediction) are emitted
    with `gt == _UNMATCHED_GT` so downstream consumers can decide whether to
    penalize them (per_field_macro_f1 does) or drop them (collapse-metric does,
    since they have no GT to be a head-vs-tail class of).

    penalize_unmatched=None defers to resolve_penalize_unmatched() (env-var
    override, default True = published basis)."""
    penalize_unmatched = resolve_penalize_unmatched(penalize_unmatched)
    by_field: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)

    for p_str, gt in zip(predictions, ground_truths):
        p = _safe_parse(p_str)
        by_field["hemisphere"].append((getattr(p, "hemisphere", None), gt.hemisphere))
        by_field["overall_mass_effect"].append((
            getattr(p, "overall_mass_effect", None), gt.overall_mass_effect))

        gt_lesions = gt.lesions
        pred_lesions = p.lesions if p is not None else []

        if use_hungarian and len(gt_lesions) > 0 and len(pred_lesions) > 0:
            pairs = _hungarian_lesion_pairing(pred_lesions, gt_lesions)
        else:
            pairs = [(i, i) for i in range(min(len(pred_lesions), len(gt_lesions)))]

        for pi, gi in pairs:
            for fname in _FIELD_ACCESSORS:
                if not fname.startswith("lesion_"):
                    continue
                pl = pred_lesions[pi] if pi < len(pred_lesions) else None
                gl = gt_lesions[gi]
                pred_val = _lesion_field(pl, fname.removeprefix("lesion_"))
                gt_val = _lesion_field(gl, fname.removeprefix("lesion_"))
                by_field[fname].append((pred_val, gt_val))

        # Over-prediction penalty: predicted lesions Hungarian left UNMATCHED have
        # no GT counterpart. Their field values are pure false positives — emit a
        # sentinel-GT row so _accuracy_and_macro_f1 dents the predicted class's
        # precision (instead of dropping them for free, which favoured the model
        # that over-segments on this single-lesion test set).
        if penalize_unmatched:
            matched_pred = {pi for pi, _ in pairs}
            for pi in range(len(pred_lesions)):
                if pi in matched_pred:
                    continue
                for fname in _FIELD_ACCESSORS:
                    if not fname.startswith("lesion_"):
                        continue
                    pred_val = _lesion_field(pred_lesions[pi], fname.removeprefix("lesion_"))
                    if pred_val is not None:
                        by_field[fname].append((pred_val, _UNMATCHED_GT))

    return by_field


def per_field_macro_f1(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    use_hungarian: bool = True,
    union_classes: bool = True,
    penalize_unmatched: bool | None = None,
    exclude_degenerate: bool | Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-field macro-F1 across the 9-field schema.
    For lesion-level fields: uses Hungarian assignment to pair predicted lesions
    with GT lesions before scoring (handles enumeration errors). Predicted lesions
    left UNMATCHED by Hungarian (over-prediction) are counted as false positives
    per lesion field when penalize_unmatched=True (default) instead of silently
    dropped. union_classes/penalize_unmatched are forwarded to _accuracy_and_macro_f1;
    set both False to reproduce the legacy (pre-2026-05-26) locked numbers.

    penalize_unmatched (default None -> True): see resolve_penalize_unmatched.
      Pass False on multi-lesion cohorts (BraTS-MET), where the in-domain
      single-lesion calibration of this penalty does not hold.

    exclude_degenerate (default None -> off): when truthy, the named fields are
      still scored and still returned per-field, but are DROPPED FROM THE
      "_overall" AGGREGATE. True loads the canonical list from
      out/field_subset_robustness.json. OFF by default so every published
      number is reproduced bit-for-bit.

    Returns per-field {accuracy, macro_f1, n_pairs}. "_overall" carries
    {macro_f1, n_fields}, plus {excluded_fields, n_excluded, macro_f1_all_fields}
    ONLY when exclude_degenerate is active (so the default output dict is
    byte-identical to the historical one).
    """
    assert len(predictions) == len(ground_truths)
    penalize_unmatched = resolve_penalize_unmatched(penalize_unmatched)
    excluded = resolve_exclude_degenerate(exclude_degenerate)

    by_field = _collect_per_field_pairs(
        predictions, ground_truths,
        use_hungarian=use_hungarian, penalize_unmatched=penalize_unmatched,
    )

    # Compute accuracy + macro-F1 per field
    result: dict[str, dict[str, float]] = {}
    for fname, pairs in by_field.items():
        result[fname] = _accuracy_and_macro_f1(
            pairs, union_classes=union_classes, penalize_unmatched=penalize_unmatched)

    # Aggregate macro-F1 across all fields
    macro_f1s = [v["macro_f1"] for v in result.values() if v["n_pairs"] > 0]
    if not excluded:
        result["_overall"] = {
            "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
            "n_fields": len(macro_f1s),
        }
        return result

    kept = [v["macro_f1"] for k, v in result.items()
            if v["n_pairs"] > 0 and k not in excluded]
    result["_overall"] = {
        "macro_f1": float(np.mean(kept)) if kept else 0.0,
        "n_fields": len(kept),
        "macro_f1_all_fields": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "n_fields_all": len(macro_f1s),
        "excluded_fields": list(excluded),
        "n_excluded": len(macro_f1s) - len(kept),
    }
    return result


def _lesion_field(lesion: Lesion | None, fname: str) -> str | None:
    if lesion is None:
        return None
    if fname == "location":
        return lesion.location
    if fname == "composition":
        return lesion.composition
    if fname == "enhancement_pattern":
        return lesion.enhancement_pattern
    if fname == "edema_severity":
        return lesion.surrounding_effects.edema_severity
    if fname == "mass_effect":
        return lesion.surrounding_effects.mass_effect
    if fname == "midline_shift":
        return lesion.surrounding_effects.midline_shift
    if fname == "ventricular_compression":
        return lesion.involvement.ventricular_compression
    if fname == "brainstem_compression":
        return lesion.involvement.brainstem_compression
    if fname == "midbrain_compression":
        return lesion.involvement.midbrain_compression
    if fname == "axis_shift":
        return lesion.axis_shift
    return None


# Sentinel GT value marking an UNMATCHED predicted lesion (over-prediction):
# a predicted field value with no GT counterpart. It is a pure false positive —
# it dents the precision of whatever class it predicted, but is never itself a
# scored class and never contributes a false negative. See per_field_macro_f1.
_UNMATCHED_GT = "\x00__UNMATCHED_PRED__\x00"


def _accuracy_and_macro_f1(
    pairs: list[tuple[str | None, str | None]],
    union_classes: bool = True,
    penalize_unmatched: bool = True,
) -> dict[str, float]:
    """Compute accuracy + macro-F1 from a list of (pred, gt) string pairs.

    None-valued preds are treated as misses; None-valued GTs are skipped (filtered).

    union_classes (default True): average F1 over classes present in EITHER GT or
      predictions. With False (legacy) the class set is GT-only, which silently
      forgives spurious off-GT predictions (e.g. predicting "bilateral" when GT
      never has it) — they dent the true class's recall but their own precision=0
      is never averaged in. The legacy path also already fixed the symmetric
      never-predicted-GT-class skip (see project_phase3_findings_2026-05-16).

    penalize_unmatched (default True): rows whose GT is the _UNMATCHED_GT sentinel
      are over-predicted lesions (Hungarian left them unpaired). They count as a
      false positive for their predicted class but are not a GT class and add no
      false negatives. With False they are dropped (legacy behaviour)."""
    if penalize_unmatched:
        real = [(p, g) for p, g in pairs if g is not None and g != _UNMATCHED_GT]
        fp_only = [p for p, g in pairs if g == _UNMATCHED_GT and p is not None]
    else:
        real = [(p, g) for p, g in pairs if g is not None and g != _UNMATCHED_GT]
        fp_only = []
    if not real and not fp_only:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n_pairs": 0}

    n_correct = sum(1 for p, g in real if p == g)
    denom = len(real) + len(fp_only)  # unmatched preds are wrong by construction
    accuracy = n_correct / denom if denom else 0.0

    # Per-class precision/recall/F1.
    # Class set: GT classes (so a never-predicted GT class still scores F1=0,
    # matching sklearn zero_division=0), plus — when union_classes — predicted
    # classes (so a spurious off-GT class scores its own precision=0). Legacy
    # GT-only iteration forgave over-prediction; see docstring.
    gt_classes = {g for _, g in real}
    pred_classes = {p for p, _ in real if p is not None} | set(fp_only)
    classes = sorted(gt_classes | pred_classes) if union_classes else sorted(gt_classes)
    f1_per_class: list[float] = []
    for c in classes:
        tp = sum(1 for p, g in real if p == c and g == c)
        fp = sum(1 for p, g in real if p == c and g != c) + sum(1 for p in fp_only if p == c)
        fn = sum(1 for p, g in real if p != c and g == c)
        if tp + fp == 0 and tp + fn == 0:
            # Class doesn't appear in either preds or GT — truly N/A, skip.
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1_per_class.append(2 * prec * rec / (prec + rec))
        else:
            f1_per_class.append(0.0)  # both prec and rec are 0 → F1=0
    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "n_pairs": len(real)}


# ===========================================================================
# GENERATIVE-COLLAPSE METRIC
#
# Two-part decomposition that separates "looks fluent" from "actually competent."
# A model that always emits the genre-frequent answer ("heterogeneous", "ring",
# "parietal", "mild edema") will score well on BLEU/BERTScore and even tie a
# rule heuristic on macro-F1 because head-class predictions cancel head-class
# GTs by sheer prior. The collapse metric exposes this by scoring head and tail
# classes separately, then quantifying how close the predicted class distribution
# is to the training-prior distribution (high KL → not collapsed; low KL → just
# regurgitates the prior).
#
# Defined in §Methodology of the ACL plan. Three numbers per field:
#   R_head   = mean recall over GT classes with prior >= HEAD_TAIL_PRIOR_THRESHOLD
#   R_tail   = mean recall over GT classes with prior <  HEAD_TAIL_PRIOR_THRESHOLD
#   KL_prior = KL(p_predicted_class_dist || p_gt_prior)   (smoothed)
#   composite = R_tail - clip(KL_prior, 0, +inf)
#
# Sanity invariants verified in tests/test_collapse_metric.py:
#   majority-class predictor  ->  R_tail = 0, low KL  -> low composite
#   oracle (always GT)        ->  R_tail = 1.0       -> high composite
#   uniform-random            ->  somewhere between, with high KL
# ===========================================================================


HEAD_TAIL_PRIOR_THRESHOLD = 0.20  # classes with prior >= this are "head"


def _enumerate_gt_field_values(gt: BrainTumorReport) -> Iterable[tuple[str, str | None]]:
    """Yield (field_name, value) for every typed field in a GT report,
    expanding lesion-level fields across all lesions."""
    yield "hemisphere", gt.hemisphere
    yield "overall_mass_effect", gt.overall_mass_effect
    for L in gt.lesions:
        for sub in ("location", "composition", "enhancement_pattern", "axis_shift"):
            yield f"lesion_{sub}", getattr(L, sub)
        for sub in ("edema_severity", "mass_effect", "midline_shift"):
            yield f"lesion_{sub}", getattr(L.surrounding_effects, sub)
        for sub in ("ventricular_compression", "brainstem_compression", "midbrain_compression"):
            yield f"lesion_{sub}", getattr(L.involvement, sub)


def compute_gt_field_prior(
    gt_jsonl_path: str | Path,
) -> dict[str, dict[str, float]]:
    """Compute per-field class probability distribution over a GT jsonl.

    Used as the reference distribution for the collapse metric's KL term.
    Returns {field_name: {class_label: probability}}.
    Skip rules: None values are dropped (treated as un-annotated, not as a class).

    Content validators (e.g. "findings must mention enhancement") are disabled
    during validation — the prior is a structural property of the typed enums
    across the corpus, not a narrative-quality filter.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    with Path(gt_jsonl_path).open() as f, strict_content(False):
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                gt = BrainTumorReport.model_validate(rec)
            except Exception:
                continue
            for fname, val in _enumerate_gt_field_values(gt):
                if val is not None:
                    counts[fname][val] += 1
    out: dict[str, dict[str, float]] = {}
    for fname, d in counts.items():
        tot = sum(d.values())
        if tot:
            out[fname] = {c: n / tot for c, n in d.items()}
    return out


def _kl_pred_to_prior(
    pairs_real: list[tuple[str | None, str | None]],
    fp_only: list[str],
    prior: dict[str, float],
    eps: float = 1e-3,
) -> float:
    """KL(p_pred || p_prior) with epsilon smoothing over the union class set."""
    pred_counter: Counter = Counter()
    for p, _ in pairs_real:
        if p is not None:
            pred_counter[p] += 1
    for p in fp_only:
        pred_counter[p] += 1
    classes = set(prior) | set(pred_counter)
    if not classes or not pred_counter:
        return 0.0
    n_pred_total = sum(pred_counter.values())
    n_prior_total = sum(prior.values()) or 1.0
    kl = 0.0
    n_cls = len(classes)
    for c in classes:
        p_hat = (pred_counter.get(c, 0) + eps) / (n_pred_total + eps * n_cls)
        q = (prior.get(c, 0.0) + eps) / (n_prior_total + eps * n_cls)
        if p_hat > 0 and q > 0:
            kl += p_hat * float(np.log(p_hat / q))
    return float(kl)


def compute_collapse_metrics(
    by_field_pairs: dict[str, list[tuple[str | None, str | None]]],
    field_prior: dict[str, dict[str, float]],
    head_tail_threshold: float = HEAD_TAIL_PRIOR_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Per-field generative-collapse decomposition.

    Inputs:
      by_field_pairs: as produced by _collect_per_field_pairs.
      field_prior:    {field: {class: prob_in_gt_training_set}}.
    Returns per-field {R_head, R_tail, KL_prior, composite, n_head_classes,
    n_tail_classes, n_pairs} plus a macro aggregate at key "_macro".
    """
    result: dict[str, dict[str, float]] = {}
    for fname, pairs in by_field_pairs.items():
        prior = field_prior.get(fname, {})
        real = [(p, g) for p, g in pairs if g is not None and g != _UNMATCHED_GT]
        fp_only = [p for p, g in pairs if g == _UNMATCHED_GT and p is not None]
        if not real or not prior:
            result[fname] = {"R_head": 0.0, "R_tail": 0.0, "KL_prior": 0.0,
                             "composite": 0.0, "n_head_classes": 0,
                             "n_tail_classes": 0, "n_pairs": len(real)}
            continue

        head_classes = {c for c, p in prior.items() if p >= head_tail_threshold}
        tail_classes = {c for c, p in prior.items() if p < head_tail_threshold}

        def class_recall(c: str) -> float | None:
            tp = sum(1 for p, g in real if g == c and p == c)
            fn = sum(1 for p, g in real if g == c and p != c)
            if tp + fn == 0:
                return None
            return tp / (tp + fn)

        head_recs = [r for c in head_classes if (r := class_recall(c)) is not None]
        tail_recs = [r for c in tail_classes if (r := class_recall(c)) is not None]
        R_head = float(np.mean(head_recs)) if head_recs else 0.0
        R_tail = float(np.mean(tail_recs)) if tail_recs else 0.0

        kl = _kl_pred_to_prior(real, fp_only, prior)
        composite = R_tail - max(kl, 0.0)

        result[fname] = {
            "R_head": R_head,
            "R_tail": R_tail,
            "KL_prior": kl,
            "composite": float(composite),
            "n_head_classes": len(head_recs),
            "n_tail_classes": len(tail_recs),
            "n_pairs": len(real),
        }

    items = [v for k, v in result.items() if not k.startswith("_") and v["n_pairs"] > 0]
    if items:
        result["_macro"] = {
            "R_head": float(np.mean([v["R_head"] for v in items])),
            "R_tail": float(np.mean([v["R_tail"] for v in items])),
            "KL_prior": float(np.mean([v["KL_prior"] for v in items])),
            "composite": float(np.mean([v["composite"] for v in items])),
            "n_fields": len(items),
        }
    else:
        result["_macro"] = {"R_head": 0.0, "R_tail": 0.0, "KL_prior": 0.0,
                            "composite": 0.0, "n_fields": 0}
    return result


def per_field_collapse(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    field_prior: dict[str, dict[str, float]],
    use_hungarian: bool = True,
    head_tail_threshold: float = HEAD_TAIL_PRIOR_THRESHOLD,
    penalize_unmatched: bool | None = None,
) -> dict[str, dict[str, float]]:
    """Convenience: collect pairs + compute the collapse metric in one call.
    Mirrors per_field_macro_f1's signature so callers can stack the two metrics
    against the same predictions list.

    penalize_unmatched was HARDCODED True here until 2026-08-05; it is now an
    explicit option resolving to True by default (unchanged behaviour)."""
    # Don't penalize unmatched in collapse — unmatched preds have no GT class to
    # contribute as a head-or-tail recall denominator; they only show up in the
    # KL term where pred-class frequency matters.
    pairs = _collect_per_field_pairs(
        predictions, ground_truths,
        use_hungarian=use_hungarian,
        penalize_unmatched=resolve_penalize_unmatched(penalize_unmatched),
    )
    return compute_collapse_metrics(pairs, field_prior, head_tail_threshold=head_tail_threshold)


# ===========================================================================
# HUNGARIAN PER-LESION MATCHING
# ===========================================================================


def _hungarian_lesion_pairing(
    pred_lesions: list[Lesion],
    gt_lesions: list[Lesion],
) -> list[tuple[int, int]]:
    """Optimal pairing of predicted to GT lesions via Hungarian algorithm.
    Cost = number of disagreeing fields (lower is better).
    Returns list of (pred_idx, gt_idx) pairs; unpaired predictions/GTs are dropped.
    """
    from scipy.optimize import linear_sum_assignment

    np_pred, np_gt = len(pred_lesions), len(gt_lesions)
    n = max(np_pred, np_gt)
    if n == 0:
        return []
    # Cost matrix; pad to square with high cost
    HIGH = 100.0
    cost = np.full((n, n), HIGH)
    for i in range(np_pred):
        for j in range(np_gt):
            cost[i, j] = _lesion_disagreement(pred_lesions[i], gt_lesions[j])
    row_idx, col_idx = linear_sum_assignment(cost)
    # Filter to real pairs
    return [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if r < np_pred and c < np_gt]


def _lesion_disagreement(p: Lesion, g: Lesion) -> int:
    """Count of disagreeing fields between two lesions. 0 = perfect match."""
    diffs = 0
    for f in _REQUIRED_LESION_FIELDS:
        pv, gv = _lesion_field(p, f), _lesion_field(g, f)
        if f in ("surrounding_effects", "involvement"):
            continue  # already covered by per-subfield accessors
        if pv != gv:
            diffs += 1
    # Sub-fields
    for sub in ("edema_severity", "mass_effect", "midline_shift"):
        if getattr(p.surrounding_effects, sub) != getattr(g.surrounding_effects, sub):
            diffs += 1
    for sub in ("ventricular_compression", "brainstem_compression", "midbrain_compression"):
        if getattr(p.involvement, sub) != getattr(g.involvement, sub):
            diffs += 1
    return diffs


def lesion_count_metrics(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
) -> dict[str, float]:
    """Lesion-count accuracy: exact, ±1, MAE, distribution match."""
    exact = 0
    plus_minus_1 = 0
    abs_err = []
    for p_str, gt in zip(predictions, ground_truths):
        p = _safe_parse(p_str)
        n_pred = len(p.lesions) if p is not None else 0
        n_gt = len(gt.lesions)
        if n_pred == n_gt:
            exact += 1
        if abs(n_pred - n_gt) <= 1:
            plus_minus_1 += 1
        abs_err.append(abs(n_pred - n_gt))
    return {
        "lesion_count/exact_acc": exact / max(len(predictions), 1),
        "lesion_count/within_1_acc": plus_minus_1 / max(len(predictions), 1),
        "lesion_count/mae": float(np.mean(abs_err)),
    }


# ===========================================================================
# NLG METRICS (BLEU / ROUGE / METEOR / BERTScore)
# ===========================================================================


def _extract_narrative(p_str: str, gt: BrainTumorReport) -> tuple[str, str]:
    """Pull the (findings + impression) narrative from prediction and GT."""
    p = _safe_parse(p_str)
    if p is not None:
        pred_text = (p.findings or "") + " " + (p.impression or "")
    else:
        # Best-effort: try to parse partial JSON
        try:
            d = json.loads(p_str)
            pred_text = (d.get("findings") or "") + " " + (d.get("impression") or "")
        except json.JSONDecodeError:
            pred_text = p_str  # last resort: use raw output
    gt_text = (gt.findings or "") + " " + (gt.impression or "")
    return pred_text.strip(), gt_text.strip()


def bleu_rouge_meteor(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
) -> dict[str, float]:
    """Standard NLG metrics. SECONDARY for EMNLP — present alongside Yu et al. 2023 caveat."""
    pred_texts = []
    gt_texts = []
    for p_str, gt in zip(predictions, ground_truths):
        pt, gtt = _extract_narrative(p_str, gt)
        pred_texts.append(pt)
        gt_texts.append(gtt)

    metrics: dict[str, float] = {}

    # BLEU
    try:
        import sacrebleu  # type: ignore
        bleu = sacrebleu.corpus_bleu(pred_texts, [gt_texts])
        metrics["bleu/corpus"] = float(bleu.score) / 100.0  # normalize 0-1
        for n in (1, 2, 3, 4):
            bleu_n = sacrebleu.corpus_bleu(pred_texts, [gt_texts], smooth_method="exp")
            # sacrebleu doesn't expose ngram-specific easily; compute via nltk fallback
        # Per-ngram via nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
        smooth = SmoothingFunction().method1
        for n in (1, 2, 3, 4):
            scores = []
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            for p, g in zip(pred_texts, gt_texts):
                if not p.strip() or not g.strip():
                    continue
                scores.append(sentence_bleu([g.split()], p.split(), weights=weights, smoothing_function=smooth))
            metrics[f"bleu/{n}"] = float(np.mean(scores)) if scores else 0.0
    except ImportError:
        log.warning("sacrebleu / nltk not installed; skipping BLEU. pip install sacrebleu nltk")

    # ROUGE
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, g in zip(pred_texts, gt_texts):
            if not p.strip() or not g.strip():
                continue
            s = scorer.score(g, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        metrics["rouge/1"] = float(np.mean(r1)) if r1 else 0.0
        metrics["rouge/2"] = float(np.mean(r2)) if r2 else 0.0
        metrics["rouge/L"] = float(np.mean(rl)) if rl else 0.0
    except ImportError:
        log.warning("rouge_score not installed; skipping ROUGE. pip install rouge-score")

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore
        import nltk  # type: ignore
        try:
            nltk.data.find("wordnet.zip")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        scores = []
        for p, g in zip(pred_texts, gt_texts):
            if not p.strip() or not g.strip():
                continue
            scores.append(meteor_score([g.split()], p.split()))
        metrics["meteor"] = float(np.mean(scores)) if scores else 0.0
    except (ImportError, LookupError):
        log.warning("nltk / wordnet not installed; skipping METEOR. pip install nltk")

    return metrics


def bert_score(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    model_type: str = "roberta-large",
) -> dict[str, float]:
    """BERTScore — semantic similarity. Requires bert-score package + GPU."""
    try:
        from bert_score import score  # type: ignore
    except ImportError:
        log.warning("bert_score not installed; skipping BERTScore. pip install bert-score")
        return {}

    pred_texts, gt_texts = [], []
    for p_str, gt in zip(predictions, ground_truths):
        pt, gtt = _extract_narrative(p_str, gt)
        if pt.strip() and gtt.strip():
            pred_texts.append(pt)
            gt_texts.append(gtt)
    if not pred_texts:
        return {"bertscore/p": 0.0, "bertscore/r": 0.0, "bertscore/f1": 0.0}
    P, R, F = score(pred_texts, gt_texts, lang="en", model_type=model_type, verbose=False)
    return {
        "bertscore/p": float(P.mean().item()),
        "bertscore/r": float(R.mean().item()),
        "bertscore/f1": float(F.mean().item()),
    }


# ===========================================================================
# RADGRAPH-F1 (Yu et al. NEJM AI 2023)
# ===========================================================================


def radgraph_f1(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
) -> dict[str, float]:
    """RadGraph entity + relation F1 via stanford radgraph package.

    NOTE: RadGraph's NER is chest-X-ray-trained. Apply with the chest-mismatch caveat.
    pip install radgraph
    """
    _ensure_adamw_shim()  # transformers.AdamW restore for radgraph's allennlp dep
    try:
        from radgraph import RadGraph  # type: ignore
    except ImportError:
        log.warning("radgraph not installed; skipping RadGraph-F1. pip install radgraph")
        return {}

    radgraph = RadGraph(reward_level="all")  # 'all' returns entity + relation F1
    pred_texts, gt_texts = [], []
    for p_str, gt in zip(predictions, ground_truths):
        pt, gtt = _extract_narrative(p_str, gt)
        pred_texts.append(pt)
        gt_texts.append(gtt)

    try:
        scores = radgraph(refs=gt_texts, hyps=pred_texts)
        return {
            "radgraph/entity_f1": float(scores.get("entity_f1", 0.0)),
            "radgraph/relation_f1": float(scores.get("relation_f1", 0.0)),
            "radgraph/overall_f1": float(scores.get("overall_f1", 0.0)),
        }
    except Exception as e:
        log.warning(f"RadGraph evaluation failed: {e}")
        return {}


# ===========================================================================
# RATESCORE (Zhao et al. EMNLP 2024) — PRIMARY FAITHFULNESS NUMBER
# ===========================================================================


def ratescore(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
) -> dict[str, float]:
    """RaTEScore — radiology entity + temporal-aware similarity. PRIMARY metric.

    pip install RaTEScore
    """
    try:
        from RaTEScore import RaTEScore  # type: ignore
    except ImportError:
        log.warning("RaTEScore not installed; skipping. pip install RaTEScore")
        return {}

    rate = RaTEScore()
    pred_texts, gt_texts = [], []
    for p_str, gt in zip(predictions, ground_truths):
        pt, gtt = _extract_narrative(p_str, gt)
        pred_texts.append(pt)
        gt_texts.append(gtt)

    try:
        scores = rate.compute_score(pred_texts, gt_texts)
        return {
            "ratescore/mean": float(np.mean(scores)),
            "ratescore/median": float(np.median(scores)),
            "ratescore/std": float(np.std(scores)),
        }
    except Exception as e:
        log.warning(f"RaTEScore evaluation failed: {e}")
        return {}


# ===========================================================================
# RADFACT (Bannur et al. 2024) — LLM-judge logical entailment
# ===========================================================================

_RADFACT_PROMPT_PRECISION = """You are evaluating a generated radiology report against a ground-truth report.
For each statement in the GENERATED report, decide whether it is logically entailed by the GROUND-TRUTH report.
A statement is TRUE if it is supported by or consistent with the GT report. A statement is FALSE otherwise.

GROUND-TRUTH REPORT:
{gt}

GENERATED REPORT:
{pred}

For each statement in the GENERATED report, output a JSON object:
{{"per_finding": ["TRUE" | "FALSE", ...]}}

Only output the JSON, no other text."""

_RADFACT_PROMPT_RECALL = """You are evaluating a generated radiology report against a ground-truth report.
For each statement in the GROUND-TRUTH report, decide whether it is logically entailed by the GENERATED report.

GROUND-TRUTH REPORT:
{gt}

GENERATED REPORT:
{pred}

For each statement in the GROUND-TRUTH report, output:
{{"per_finding": ["TRUE" | "FALSE", ...]}}

Only output the JSON, no other text."""


def _split_statements(text: str) -> list[str]:
    """Split a report into individual statements by sentence boundaries."""
    # Simple regex split; for production use spacy or nltk.sent_tokenize
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s.split()) >= 3]


def _parse_judge_response(resp: str) -> list[str]:
    """Extract per_finding TRUE/FALSE list from judge response. Robust to noise."""
    # Find the first {...} block
    m = re.search(r"\{[^{}]*\"per_finding\"[^{}]*\}", resp, re.DOTALL)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
        return [str(x).upper() for x in d.get("per_finding", [])]
    except json.JSONDecodeError:
        return []


def radfact(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    judge: JudgeFn,
    max_cases: int | None = None,
) -> dict[str, float]:
    """RadFact: LLM-judge logical precision + recall. Slow (one LLM call per case per direction).
    Set max_cases for dev runs."""
    items = list(zip(predictions, ground_truths))
    if max_cases is not None:
        items = items[:max_cases]
    if not items:
        return {}

    log_precs, log_recs = [], []
    t0 = time.time()
    for i, (p_str, gt) in enumerate(items):
        pt, gtt = _extract_narrative(p_str, gt)
        if not pt.strip() or not gtt.strip():
            continue
        # Logical precision: each pred statement entailed by GT?
        prec_resp = judge(_RADFACT_PROMPT_PRECISION.format(gt=gtt, pred=pt), max_tokens=512)
        prec_flags = _parse_judge_response(prec_resp)
        if prec_flags:
            log_precs.append(sum(1 for f in prec_flags if f == "TRUE") / len(prec_flags))
        # Logical recall: each GT statement entailed by pred?
        rec_resp = judge(_RADFACT_PROMPT_RECALL.format(gt=gtt, pred=pt), max_tokens=512)
        rec_flags = _parse_judge_response(rec_resp)
        if rec_flags:
            log_recs.append(sum(1 for f in rec_flags if f == "TRUE") / len(rec_flags))
        if (i + 1) % 10 == 0:
            log.info(f"  RadFact: {i+1}/{len(items)} cases in {time.time()-t0:.1f}s")

    if not log_precs and not log_recs:
        return {}
    metrics = {}
    if log_precs:
        metrics["radfact/logical_precision"] = float(np.mean(log_precs))
    if log_recs:
        metrics["radfact/logical_recall"] = float(np.mean(log_recs))
    if log_precs and log_recs:
        p, r = metrics["radfact/logical_precision"], metrics["radfact/logical_recall"]
        if p + r > 0:
            metrics["radfact/logical_f1"] = 2 * p * r / (p + r)
    return metrics


# ===========================================================================
# GREEN (Ostmeier et al. Findings of EMNLP 2024) — LLM-judge clinical errors
# ===========================================================================

_GREEN_PROMPT = """You are a careful radiologist evaluating a generated radiology report against the ground truth.
Score the GENERATED report on a 0.0-1.0 scale considering:
- Accuracy: are findings correct?
- Completeness: are all GT findings present?
- No invented findings: any phantom lesions or fabricated details?
- Correct localization: are anatomical locations accurate?
- Correct severity: are severity assessments accurate?

Also count clinical errors of these types:
- false_finding (invented)
- missed_critical (missed important finding)
- wrong_localization
- wrong_severity

GROUND-TRUTH REPORT:
{gt}

GENERATED REPORT:
{pred}

Output JSON:
{{"score": <0.0-1.0>, "errors": {{"false_finding": <int>, "missed_critical": <int>, "wrong_localization": <int>, "wrong_severity": <int>}}}}

Only output the JSON, no other text."""


def green(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    judge: JudgeFn,
    max_cases: int | None = None,
) -> dict[str, float]:
    """GREEN: LLM-judge overall score + error taxonomy."""
    items = list(zip(predictions, ground_truths))
    if max_cases is not None:
        items = items[:max_cases]
    if not items:
        return {}

    scores = []
    err_counts: dict[str, list[int]] = {
        "false_finding": [], "missed_critical": [],
        "wrong_localization": [], "wrong_severity": [],
    }
    t0 = time.time()
    for i, (p_str, gt) in enumerate(items):
        pt, gtt = _extract_narrative(p_str, gt)
        if not pt.strip() or not gtt.strip():
            continue
        resp = judge(_GREEN_PROMPT.format(gt=gtt, pred=pt), max_tokens=400)
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
            scores.append(float(d.get("score", 0.0)))
            for k in err_counts:
                err_counts[k].append(int(d.get("errors", {}).get(k, 0)))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if (i + 1) % 10 == 0:
            log.info(f"  GREEN: {i+1}/{len(items)} cases in {time.time()-t0:.1f}s")

    metrics = {}
    if scores:
        metrics["green/overall"] = float(np.mean(scores))
        metrics["green/median"] = float(np.median(scores))
    for k, v in err_counts.items():
        if v:
            metrics[f"green/errors_per_case/{k}"] = float(np.mean(v))
    return metrics


# ===========================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ===========================================================================


def bootstrap_ci(
    metric_fn: Callable[[list[Any], list[Any]], dict[str, float]],
    predictions: list[Any],
    ground_truths: list[Any],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Run metric_fn on n_bootstrap resamples; return per-metric mean + CI.

    metric_fn must return a dict[str, float] of metric name -> scalar value.
    Returns dict[metric_name -> {mean, lower, upper, std}].
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    bootstrap_results: dict[str, list[float]] = defaultdict(list)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sub_preds = [predictions[i] for i in idx]
        sub_gts = [ground_truths[i] for i in idx]
        try:
            metrics = metric_fn(sub_preds, sub_gts)
        except Exception as e:
            log.debug(f"bootstrap iter {b} failed: {e}")
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bootstrap_results[k].append(float(v))

    alpha = (1 - confidence) / 2
    ci: dict[str, dict[str, float]] = {}
    for k, vals in bootstrap_results.items():
        if not vals:
            continue
        arr = np.array(vals)
        ci[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1 - alpha)),
            "n_bootstrap": len(vals),
        }
    return ci


# ===========================================================================
# SEMANTIC ENTROPY + CONFORMAL ABSTENTION (50 held-out cal cases)
# ===========================================================================


def _canonical_for_clustering(s: str) -> str:
    """Canonical form for semantic-equivalence clustering.

    Strategy: parse JSON, drop free-text narrative fields (findings/impression),
    sort keys, dump → exact-match clusters samples that agree on structured fields.
    Falls back to lowercased text on parse failure.
    """
    try:
        d = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s.strip().lower()
    if isinstance(d, dict):
        d = {k: v for k, v in d.items() if k not in ("findings", "impression")}
    return json.dumps(d, sort_keys=True, default=str)


def semantic_cluster(samples: list[str]) -> list[int]:
    """Cluster K samples by structured-field equivalence. Returns cluster id per sample.

    Cheap proxy following Farquhar et al. 2024: equal canonical form ⇒ same meaning class.
    For paper-grade, swap _canonical_for_clustering with a bidirectional NLI judge."""
    clusters: dict[str, int] = {}
    out: list[int] = []
    for s in samples:
        c = _canonical_for_clustering(s)
        out.append(clusters.setdefault(c, len(clusters)))
    return out


def semantic_entropy(samples: list[str]) -> float:
    """Shannon entropy over semantic clusters of K samples (Farquhar et al. 2024).
    H(S) = -Σ p_c log p_c with p_c = |cluster c|/K. Returns 0.0 if samples empty."""
    if not samples:
        return 0.0
    ids = semantic_cluster(samples)
    K = len(ids)
    counts = Counter(ids)
    probs = np.array([c / K for c in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def conformal_calibrate(scores: list[float], alpha: float = 0.1) -> float:
    """Split-conformal abstention threshold (marginal coverage guarantee).

    Higher score = more uncertain (e.g., semantic entropy). On a calibration set
    of size n, τ = empirical quantile at level ⌈(n+1)(1-α)⌉/n. Test cases with
    score > τ abstain; expected abstention rate is approximately α
    (Angelopoulos & Bates 2023, eq. 2.2)."""
    if not scores:
        raise ValueError("conformal_calibrate: empty scores")
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.asarray(scores, dtype=np.float64), q, method="higher"))


def conformal_risk_calibrate(
    scores: list[float],
    losses: list[float],
    alpha: float = 0.1,
) -> float:
    """Risk-controlling conformal threshold (Angelopoulos et al. 2024 / Quach et al. 2024).

    Returns the largest τ such that empirical mean loss among kept items
    (score ≤ τ) is ≤ α. Provides E[loss | non-abstain] ≤ α on iid test data.

    scores: nonconformity (higher = less confident; e.g., semantic entropy)
    losses: per-case loss in [0,1] (e.g., 1 - per-field accuracy)"""
    assert len(scores) == len(losses), "scores and losses must align"
    if not scores:
        raise ValueError("conformal_risk_calibrate: empty inputs")
    # NB: walk the FULL sorted list. Previous version broke on the first
    # violation, which made τ_risk = -inf whenever the very first (lowest-
    # score) case happened to be a high-loss outlier — even if later cases
    # would have brought the running mean back under α. That artifact
    # explained the "τ_risk finite only at α=0.9" symptom in Phase 2b.
    pairs = sorted(zip(scores, losses), key=lambda x: x[0])
    cum_loss = 0.0
    best_tau = float("-inf")
    for i, (s, l) in enumerate(pairs, start=1):
        cum_loss += l
        if cum_loss / i <= alpha:
            best_tau = s
    return best_tau


def conformal_evaluate(
    scores: list[float],
    threshold: float,
    losses: list[float] | None = None,
) -> dict[str, float]:
    """Apply a calibrated conformal threshold to a test set.

    Returns abstention rate, retained-set size, and (if losses given) empirical
    risk on retained set."""
    n = len(scores)
    if n == 0:
        return {}
    arr = np.asarray(scores, dtype=np.float64)
    kept = arr <= threshold
    n_kept = int(kept.sum())
    out: dict[str, float] = {
        "conformal/threshold": float(threshold),
        "conformal/abstention_rate": float(1.0 - n_kept / n),
        "conformal/n_kept": float(n_kept),
        "conformal/n_total": float(n),
    }
    if losses is not None and n_kept > 0:
        out["conformal/risk_on_kept"] = float(np.asarray(losses)[kept].mean())
    return out


def _per_case_accuracy(p_str: str, gt: BrainTumorReport) -> float:
    """Per-case scalar correctness in [0,1]: mean over the 12 structured fields
    of (pred_val == gt_val), using Hungarian pairing for lesion fields.
    Cases that fail to parse score 0.0 (worst-case loss)."""
    p = _safe_parse(p_str)
    correct = 0
    n = 0
    for name, accessor in _FIELD_ACCESSORS.items():
        if name.startswith("lesion_"):
            continue
        try:
            pv = accessor(p, 0) if p is not None else None
            gv = accessor(gt, 0)
        except Exception:
            continue
        n += 1
        if pv == gv:
            correct += 1
    if p is not None and gt.lesions:
        pairs = _hungarian_lesion_pairing(p.lesions, gt.lesions) if p.lesions else []
        for pi, gi in pairs:
            for name, accessor in _FIELD_ACCESSORS.items():
                if not name.startswith("lesion_"):
                    continue
                try:
                    pv = accessor(p, pi)
                    gv = accessor(gt, gi)
                except Exception:
                    continue
                n += 1
                if pv == gv:
                    correct += 1
    return correct / max(n, 1)


# ===========================================================================
# BraTS-STANDARD ET / TC / WT DICE  (Task 6)
# ===========================================================================


def compute_brats_dice(pred: "np.ndarray", target: "np.ndarray") -> dict[str, float]:
    """BraTS-standard union-mask Dice on remapped labels {0,1,2,3}.

    Definitions (after BraTS {0,1,2,4}→{0,1,2,3} remap; see dataset.py):
      - ET (Enhancing Tumor): seg==3
      - TC (Tumor Core):       (seg==1) | (seg==3)
      - WT (Whole Tumor):      (seg==1) | (seg==2) | (seg==3)

    pred, target: integer-label arrays of identical shape with values in {0,1,2,3}.
    Empty-target classes return Dice=1.0 if pred is also empty, else 0.0.
    """
    p = np.asarray(pred)
    t = np.asarray(target)
    assert p.shape == t.shape, f"shape mismatch: pred {p.shape} vs target {t.shape}"

    def _dice(pm: "np.ndarray", tm: "np.ndarray") -> float:
        pm = pm.astype(bool)
        tm = tm.astype(bool)
        if not pm.any() and not tm.any():
            return 1.0
        inter = float((pm & tm).sum())
        denom = float(pm.sum() + tm.sum())
        return (2.0 * inter / denom) if denom > 0 else 0.0

    et = _dice(p == 3, t == 3)
    tc = _dice((p == 1) | (p == 3), (t == 1) | (t == 3))
    wt = _dice((p == 1) | (p == 2) | (p == 3), (t == 1) | (t == 2) | (t == 3))
    return {
        "brats/ET": et,
        "brats/TC": tc,
        "brats/WT": wt,
        "brats/mean": float(np.mean([et, tc, wt])),
    }


# ===========================================================================
# EVALUATOR — orchestrates the full panel
# ===========================================================================


@dataclass
class EvalConfig:
    bootstrap: bool = True
    n_bootstrap: int = 1000
    judge_max_cases: int | None = None  # None = all cases
    skip_radfact: bool = False
    skip_green: bool = False
    skip_bertscore: bool = False
    skip_radgraph: bool = False
    skip_ratescore: bool = False
    judge: JudgeFn | None = None
    # Conformal abstention (active only when per-case samples are provided)
    conformal_alpha: float = 0.1
    conformal_calibration_ids: set[str] | None = None  # case_ids designated as cal set
    # --- SCORING BASIS (both default to the published basis; see the
    #     "SCORING-BASIS OPTIONS" block near _safe_parse) ---
    exclude_degenerate: bool | tuple[str, ...] | None = None  # None -> env -> off
    penalize_unmatched: bool | None = None                    # None -> env -> True


def evaluate_all(
    predictions: list[str],
    ground_truths: list[BrainTumorReport],
    cfg: EvalConfig,
    case_ids: list[str] | None = None,
    samples_per_case: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Run the full metric panel. Returns nested metrics dict.

    If samples_per_case is provided (K samples per case) AND cfg.conformal_calibration_ids
    is non-empty, also runs semantic-entropy + split-conformal abstention calibration
    on the calibration subset and reports test-set abstention rate + risk-on-kept."""
    assert len(predictions) == len(ground_truths)
    log.info(f"=== NeuroFusion eval pipeline on {len(predictions)} cases ===")

    out: dict[str, Any] = {"n_cases": len(predictions)}

    # 1. Schema + structure
    log.info("[1/8] schema adherence + section completeness")
    out["schema"] = schema_adherence(predictions)
    out["completeness"] = section_completeness(predictions)

    # 2. Per-field macro-F1 (PRIMARY)
    log.info("[2/8] per-field macro-F1 with Hungarian lesion pairing")
    _excluded = resolve_exclude_degenerate(cfg.exclude_degenerate)
    _penalize = resolve_penalize_unmatched(cfg.penalize_unmatched)
    out["scoring_basis"] = {
        "excluded_degenerate_fields": list(_excluded),
        "penalize_unmatched": _penalize,
        "is_published_basis": (not _excluded) and _penalize,
    }
    if not out["scoring_basis"]["is_published_basis"]:
        log.warning("scoring basis is NON-DEFAULT: %s", out["scoring_basis"])
    out["per_field"] = per_field_macro_f1(
        predictions, ground_truths, use_hungarian=True,
        penalize_unmatched=_penalize, exclude_degenerate=_excluded or False,
    )

    # 3. Lesion-count
    log.info("[3/8] lesion-count metrics")
    out["lesion_count"] = lesion_count_metrics(predictions, ground_truths)

    # 4. NLG metrics (SECONDARY)
    log.info("[4/8] BLEU / ROUGE / METEOR")
    out["nlg"] = bleu_rouge_meteor(predictions, ground_truths)

    if not cfg.skip_bertscore:
        log.info("[5/8] BERTScore")
        out["bertscore"] = bert_score(predictions, ground_truths)
    else:
        log.info("[5/8] BERTScore SKIPPED")

    # 5. Faithfulness — RadGraph + RaTEScore
    if not cfg.skip_radgraph:
        log.info("[6/8] RadGraph-F1")
        out["radgraph"] = radgraph_f1(predictions, ground_truths)
    if not cfg.skip_ratescore:
        log.info("[7/8] RaTEScore")
        out["ratescore"] = ratescore(predictions, ground_truths)

    # 6. LLM-judge metrics (slow!)
    if not cfg.skip_radfact and cfg.judge is not None:
        log.info("[8/8] RadFact (LLM-judge)")
        out["radfact"] = radfact(predictions, ground_truths, cfg.judge, cfg.judge_max_cases)
    if not cfg.skip_green and cfg.judge is not None:
        log.info("    GREEN (LLM-judge)")
        out["green"] = green(predictions, ground_truths, cfg.judge, cfg.judge_max_cases)

    # 7. Semantic entropy + split-conformal abstention (if K samples per case provided)
    if samples_per_case is not None and case_ids is not None:
        assert len(samples_per_case) == len(predictions)
        entropies = [semantic_entropy(s) for s in samples_per_case]
        losses = [1.0 - _per_case_accuracy(p, g) for p, g in zip(predictions, ground_truths)]
        cal_ids = cfg.conformal_calibration_ids or set()
        cal_idx = [i for i, cid in enumerate(case_ids) if cid in cal_ids]
        test_idx = [i for i, cid in enumerate(case_ids) if cid not in cal_ids]
        if cal_idx and test_idx:
            cal_scores = [entropies[i] for i in cal_idx]
            cal_losses = [losses[i] for i in cal_idx]
            test_scores = [entropies[i] for i in test_idx]
            test_losses = [losses[i] for i in test_idx]
            tau_marginal = conformal_calibrate(cal_scores, alpha=cfg.conformal_alpha)
            tau_risk = conformal_risk_calibrate(cal_scores, cal_losses, alpha=cfg.conformal_alpha)
            out["conformal_marginal"] = conformal_evaluate(test_scores, tau_marginal, test_losses)
            out["conformal_risk"] = conformal_evaluate(test_scores, tau_risk, test_losses)
            out["conformal_meta"] = {
                "alpha": cfg.conformal_alpha,
                "n_calibration": len(cal_idx),
                "n_test": len(test_idx),
                "cal_entropy_mean": float(np.mean(cal_scores)),
                "test_entropy_mean": float(np.mean(test_scores)),
            }
            # Stash per-case for CSV writer
            out["_per_case"] = {
                "case_ids": case_ids,
                "entropies": entropies,
                "losses": losses,
                "is_calibration": [cid in cal_ids for cid in case_ids],
                "kept_marginal": [bool(e <= tau_marginal) for e in entropies],
                "kept_risk": [bool(e <= tau_risk) for e in entropies],
            }
            log.info(
                f"conformal: τ_marginal={tau_marginal:.4f} → abstention {out['conformal_marginal']['conformal/abstention_rate']:.3f}; "
                f"τ_risk={tau_risk:.4f} → risk_on_kept {out['conformal_risk'].get('conformal/risk_on_kept', float('nan')):.4f}"
            )
        else:
            log.warning("conformal: missing calibration_ids or test set; skipping")

    # 8. Bootstrap CIs on the headline metrics
    if cfg.bootstrap:
        log.info(f"computing bootstrap 95% CIs (n={cfg.n_bootstrap})")
        # Run the cheap metrics under bootstrap
        def cheap_panel(p, g):
            m: dict[str, float] = {}
            sa = schema_adherence(p)
            m["schema/adherence"] = sa["schema_adherence"]
            ff = per_field_macro_f1(
                p, g, use_hungarian=True,
                penalize_unmatched=_penalize, exclude_degenerate=_excluded or False,
            )
            m["per_field/macro_f1"] = ff["_overall"]["macro_f1"]
            lc = lesion_count_metrics(p, g)
            m.update(lc)
            return m

        out["bootstrap_ci"] = bootstrap_ci(
            cheap_panel, predictions, ground_truths,
            n_bootstrap=cfg.n_bootstrap, confidence=0.95,
        )

    # Per-case scalars merged into the conformal per-case dict (if present).
    # Previously this assignment unconditionally OVERWROTE out["_per_case"]
    # with a DIFFERENT schema (no `is_calibration`/`kept_marginal` keys),
    # which broke the downstream CSV writer that reads pc["is_calibration"].
    # Audit M-H: merge instead of clobber.
    per_case_scalars = {
        "case_ids": case_ids if case_ids is not None else [str(i) for i in range(len(predictions))],
        "schema_valid": [_safe_parse(p) is not None for p in predictions],
        "n_pred_lesions": [len(_safe_parse(p).lesions) if _safe_parse(p) else 0 for p in predictions],
        "n_gt_lesions": [len(gt.lesions) for gt in ground_truths],
        "per_case_accuracy": [_per_case_accuracy(p, gt) for p, gt in zip(predictions, ground_truths)],
    }
    if "_per_case" in out and isinstance(out["_per_case"], dict):
        # Merge — preserve conformal keys (entropies, kept_marginal, etc.)
        # while adding the scalar diagnostics.
        out["_per_case"] = {**out["_per_case"], **per_case_scalars}
    else:
        out["_per_case"] = per_case_scalars

    return out


# ===========================================================================
# REPORT WRITER
# ===========================================================================


def write_metrics_report(metrics: dict[str, Any], out_dir: Path) -> None:
    """Write metrics.json + report.md to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # Markdown report
    lines: list[str] = ["# NeuroFusion v4 — Evaluation Report", ""]
    lines.append(f"**Cases evaluated:** {metrics.get('n_cases', '?')}")
    lines.append("")

    # Headline numbers
    lines.append("## Headline numbers (for EMNLP abstract table)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    headline = [
        ("Schema adherence", metrics.get("schema", {}).get("schema_adherence")),
        ("Per-field macro-F1 (PRIMARY structured)", metrics.get("per_field", {}).get("_overall", {}).get("macro_f1")),
        ("RaTEScore mean (PRIMARY faithfulness)", metrics.get("ratescore", {}).get("ratescore/mean")),
        ("RadFact logical precision (PRIMARY faithfulness)", metrics.get("radfact", {}).get("radfact/logical_precision")),
        ("RadFact logical recall", metrics.get("radfact", {}).get("radfact/logical_recall")),
        ("RadGraph overall F1", metrics.get("radgraph", {}).get("radgraph/overall_f1")),
        ("GREEN overall (LLM-judge)", metrics.get("green", {}).get("green/overall")),
        ("Lesion-count exact accuracy", metrics.get("lesion_count", {}).get("lesion_count/exact_acc")),
    ]
    for name, val in headline:
        v = f"{val:.4f}" if isinstance(val, float) else "n/a"
        lines.append(f"| {name} | {v} |")
    lines.append("")

    # Bootstrap CIs
    if "bootstrap_ci" in metrics:
        lines.append("## Bootstrap 95% CIs")
        lines.append("")
        lines.append("| Metric | Mean | 95% CI | Std |")
        lines.append("|---|---|---|---|")
        for k, ci in metrics["bootstrap_ci"].items():
            lines.append(f"| {k} | {ci['mean']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {ci['std']:.4f} |")
        lines.append("")

    # Per-field breakdown
    if "per_field" in metrics:
        lines.append("## Per-field accuracy + macro-F1")
        lines.append("")
        lines.append("| Field | Accuracy | Macro-F1 | n_pairs |")
        lines.append("|---|---|---|---|")
        for fname, vals in sorted(metrics["per_field"].items()):
            if fname == "_overall":
                continue
            acc = vals.get("accuracy", 0.0)
            f1 = vals.get("macro_f1", 0.0)
            n = vals.get("n_pairs", 0)
            lines.append(f"| {fname} | {acc:.4f} | {f1:.4f} | {n} |")
        lines.append("")

    # NLG
    if "nlg" in metrics and metrics["nlg"]:
        lines.append("## Standard NLG metrics (SECONDARY — present with Yu et al. 2023 caveat)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in metrics["nlg"].items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    # Conformal abstention block
    if "conformal_marginal" in metrics or "conformal_risk" in metrics:
        meta = metrics.get("conformal_meta", {})
        lines.append("## Conformal abstention (50 held-out calibration)")
        lines.append("")
        lines.append(f"alpha = {meta.get('alpha', '?')}, n_cal = {meta.get('n_calibration', '?')}, n_test = {meta.get('n_test', '?')}")
        lines.append("")
        lines.append("| Mode | τ | Abstention rate | Risk on kept |")
        lines.append("|---|---|---|---|")
        for mode_key, mode_name in (("conformal_marginal", "marginal"), ("conformal_risk", "risk-controlling")):
            m = metrics.get(mode_key, {})
            if not m:
                continue
            tau = m.get("conformal/threshold", float("nan"))
            ar = m.get("conformal/abstention_rate", float("nan"))
            rk = m.get("conformal/risk_on_kept", float("nan"))
            lines.append(f"| {mode_name} | {tau:.4f} | {ar:.4f} | {rk:.4f} |")
        lines.append("")

    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))

    # Per-case CSV (claimed in module docstring; populated when conformal ran)
    pc = metrics.get("_per_case")
    if pc:
        with open(out_dir / "per_case.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "is_calibration", "semantic_entropy", "per_case_loss", "kept_marginal", "kept_risk"])
            for cid, ic, e, l, km, kr in zip(
                pc["case_ids"], pc["is_calibration"], pc["entropies"],
                pc["losses"], pc["kept_marginal"], pc["kept_risk"],
            ):
                w.writerow([cid, int(ic), f"{e:.6f}", f"{l:.6f}", int(km), int(kr)])
        log.info(f"wrote per_case.csv ({len(pc['case_ids'])} rows)")

    log.info(f"wrote metrics.json + report.md to {out_dir}")


# ===========================================================================
# MAIN
# ===========================================================================


def load_predictions_jsonl(path: Path) -> tuple[list[str], list[str], list[list[str]] | None]:
    """Load predictions. Returns (case_ids, pred_jsons, samples_per_case_or_None).

    Each line: {"case_id": str, "pred_json": str, "samples": list[str] (optional)}.
    If any line carries a non-empty `samples` list, returns the full list (aligned
    with case_ids; entries without samples become []). Otherwise returns None.
    """
    case_ids: list[str] = []
    pred_jsons: list[str] = []
    samples: list[list[str]] = []
    any_samples = False
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            case_ids.append(d["case_id"])
            pred_jsons.append(d["pred_json"])
            s = d.get("samples") or []
            if s:
                any_samples = True
            samples.append(list(s))
    return case_ids, pred_jsons, (samples if any_samples else None)


def load_ground_truth_jsonl(path: Path, case_ids_filter: set[str] | None = None) -> dict[str, BrainTumorReport]:
    """Load GT reports, optionally filtered to a subset of case_ids."""
    out: dict[str, BrainTumorReport] = {}
    with open(path) as f:
        for line in f:
            r = BrainTumorReport.model_validate_json(line)
            if case_ids_filter is None or r.case_id in case_ids_filter:
                out[r.case_id] = r
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-jsonl", type=Path, required=True, help="predictions: {case_id, pred_json}")
    p.add_argument("--gt-jsonl", type=Path, required=True, help="ground truth: BrainTumorReport JSON per line")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--judge", choices=["mock", "openai", "anthropic", "hf", "none"], default="mock")
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--judge-max-cases", type=int, default=None)
    p.add_argument("--skip-radfact", action="store_true")
    p.add_argument("--skip-green", action="store_true")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-radgraph", action="store_true")
    p.add_argument("--skip-ratescore", action="store_true")
    p.add_argument("--splits-json", type=Path, default=None,
                   help="out/splits.json — used to find the 50 conformal_calibration case_ids")
    p.add_argument("--conformal-alpha", type=float, default=0.1)
    # --- SCORING BASIS (opt-in; omitting both reproduces every published number) ---
    p.add_argument("--exclude-degenerate-fields", action="store_true",
                   help="DROP single-class degenerate fields (from "
                        "out/field_subset_robustness.json:excluded_degenerate) from the "
                        "per-field aggregate. NON-DEFAULT: not comparable to published numbers.")
    p.add_argument("--no-penalize-unmatched", action="store_true",
                   help="Do NOT count Hungarian-unmatched predicted lesions as per-field "
                        "false positives. Appropriate for multi-lesion cohorts (BraTS-MET); "
                        "the penalty was calibrated on the 1-lesion-per-case in-domain split. "
                        "NON-DEFAULT: not comparable to published numbers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_dir / "eval.log")])

    case_ids, pred_jsons, samples_per_case = load_predictions_jsonl(args.pred_jsonl)
    gt_by_id = load_ground_truth_jsonl(args.gt_jsonl, case_ids_filter=set(case_ids))
    # Align by case_id
    gts_aligned, preds_aligned, ids_aligned = [], [], []
    samples_aligned: list[list[str]] | None = [] if samples_per_case is not None else None
    for i, (cid, p) in enumerate(zip(case_ids, pred_jsons)):
        if cid in gt_by_id:
            gts_aligned.append(gt_by_id[cid])
            preds_aligned.append(p)
            ids_aligned.append(cid)
            if samples_aligned is not None:
                samples_aligned.append(samples_per_case[i])
    log.info(f"aligned {len(preds_aligned)} predictions with GT")

    # Optional: conformal calibration ids from splits.json
    conformal_cal_ids: set[str] | None = None
    if args.splits_json is not None:
        sp = json.loads(args.splits_json.read_text())
        conformal_cal_ids = set(sp.get("conformal_calibration", []))
        log.info(f"conformal calibration set: {len(conformal_cal_ids)} case_ids from {args.splits_json}")

    # Build judge
    judge: JudgeFn | None = None
    if args.judge == "mock":
        judge = make_mock_judge()
    elif args.judge == "openai":
        judge = make_openai_judge(args.judge_model or "gpt-4-turbo")
    elif args.judge == "anthropic":
        judge = make_anthropic_judge(args.judge_model or "claude-sonnet-4-20250514")
    elif args.judge == "hf":
        judge = make_hf_judge(args.judge_model or "meta-llama/Llama-3.3-70B-Instruct")

    cfg = EvalConfig(
        bootstrap=not args.no_bootstrap,
        n_bootstrap=args.bootstrap,
        judge=judge,
        judge_max_cases=args.judge_max_cases,
        skip_radfact=args.skip_radfact or judge is None,
        skip_green=args.skip_green or judge is None,
        skip_bertscore=args.skip_bertscore,
        skip_radgraph=args.skip_radgraph,
        skip_ratescore=args.skip_ratescore,
        conformal_alpha=args.conformal_alpha,
        conformal_calibration_ids=conformal_cal_ids,
        # None -> env -> published default. A CLI flag is an EXPLICIT opt-in.
        exclude_degenerate=True if args.exclude_degenerate_fields else None,
        penalize_unmatched=False if args.no_penalize_unmatched else None,
    )

    metrics = evaluate_all(
        preds_aligned, gts_aligned, cfg,
        case_ids=ids_aligned,
        samples_per_case=samples_aligned,
    )
    write_metrics_report(metrics, args.out_dir)


if __name__ == "__main__":
    main()
