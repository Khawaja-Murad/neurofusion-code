#!/usr/bin/env python
"""Score narrative report faithfulness with RaTEScore (Zhao et al. 2024).

Replaces / augments the BERTScore-only narrative evaluation for the EACL paper.
RaTEScore is an entity-aware metric: it extracts medical entities (anatomy,
abnormality, disease, ...) from candidate and reference reports via a fine-tuned
RaTE-NER-Deberta model, embeds them with BioLORD-2023-C, and computes a
synonym-robust, negation-sensitive precision/recall F1.

We score the *narrative* text (findings + impression) of each system against the
GT human-authored narrative, restricted to the common n=39 test case_ids.

Leak-free: candidates are scored only against GT narrative. No GT-seg, no oracle
selection (we deterministically take the first parse-valid prediction).

GREEN was the first-choice metric but its only PyPI distribution (green_score)
requires Python==3.12.1; the venv is 3.11, so it cannot install.
RadGraph-F1 imports a vendored AllenNLP that references transformers.AdamW,
removed in transformers>=5 (the version the pipeline pins) -> hard API break.
RaTEScore installs cleanly against the venv's transformers 5.8.0 / spacy 3.8.2
(its only missing dep was medspacy + a couple of pure-python transitives, added
with --no-deps so nothing in the pipeline was upgraded/downgraded).

Usage (login node, network on for first-run HF model download; cached after):
  source neuro_venv/bin/activate
  python scripts/eval_narrative_rate.py
"""
import argparse
import json
import os
import sys

# Quiet PyRuSH/medspacy DEBUG spam before RaTEScore import triggers medspacy.load
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="WARNING")
except Exception:
    pass

import numpy as np

# Repo root and prediction directory are environment-driven so this script runs
# outside the original cluster. NF_REPO follows the project convention; NF_PRED_DIR
# points at wherever the per-system prediction JSONLs were written.
REPO = os.environ.get(
    "NF_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.environ.get("NF_PRED_DIR", os.path.join(REPO, "predictions"))
GT_JSONL = os.path.join(REPO, "out", "neurofusion_369.jsonl")
SPLITS = os.path.join(REPO, "out", "splits.json")
OUT_JSON = os.path.join(REPO, "out", "narrative_ratescore.json")

SYSTEMS = {
    "nf_cot": os.path.join(PRED_DIR, "multichain_cot", "predictions_cot_test_k2.jsonl"),
    "nf_single": os.path.join(PRED_DIR, "fold4_k8", "predictions_test.jsonl"),
    "heuristic": os.path.join(REPO, "out", "heuristic_predseg_test.jsonl"),
    "m3d": os.path.join(PRED_DIR, "baselines", "m3d_narrative", "predictions_test.jsonl"),
    "llava": os.path.join(PRED_DIR, "baselines", "llava_med_narrative", "predictions_test.jsonl"),
}


def _findings_impression_from_json(obj):
    """Build narrative text from a parsed report-JSON dict."""
    f = (obj.get("findings") or "").strip()
    i = (obj.get("impression") or "").strip()
    return (f + " " + i).strip()


def _first_valid_pred_text(predictions):
    """Return narrative text for one case.

    predictions is a list. Each element is either a JSON-string report
    (NeuroFusion / heuristic) -> parse and take findings+impression, or raw
    prose (VLM baselines) -> use directly. We take the FIRST element that
    yields usable text (deterministic, no oracle selection).
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if not isinstance(predictions, list):
        return ""
    for p in predictions:
        if not isinstance(p, str):
            continue
        p_str = p.strip()
        if not p_str:
            continue
        # Try structured JSON report first.
        try:
            obj = json.loads(p_str)
            if isinstance(obj, dict) and ("findings" in obj or "impression" in obj):
                txt = _findings_impression_from_json(obj)
                if txt:
                    return txt
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to raw prose (strip a leading FINDINGS:/IMPRESSION: is fine
        # to keep -- the NER model handles the section headers as plain tokens).
        return p_str
    return ""


def load_system(path):
    """case_id -> narrative text."""
    out = {}
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            out[d["case_id"]] = _first_valid_pred_text(d.get("predictions", []))
    return out


def load_gt():
    out = {}
    with open(GT_JSONL) as fh:
        for ln in fh:
            d = json.loads(ln)
            out[d["case_id"]] = _findings_impression_from_json(d)
    return out


def paired_bootstrap_ci(a, b, n_boot=2000, seed=0):
    """Bootstrap CI on mean(a) - mean(b), paired by case index."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float((a.mean() - b.mean())), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-gpu", action="store_true", help="run RaTE models on CUDA")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    test_ids = json.load(open(SPLITS))["test"]
    gt = load_gt()

    sys_texts = {name: load_system(p) for name, p in SYSTEMS.items()}

    # Common case_ids present in GT and EVERY system.
    common = [c for c in test_ids
              if c in gt and all(c in sys_texts[s] for s in SYSTEMS)]
    common.sort()
    print(f"[info] test ids: {len(test_ids)}  common across all systems+GT: {len(common)}")

    from RaTEScore import RaTEScore
    scorer = RaTEScore(use_gpu=args.use_gpu, batch_size=1)

    refs = [gt[c] for c in common]
    per_system_scores = {}
    results = {}
    for name in SYSTEMS:
        cands = [sys_texts[name][c] for c in common]
        scores = scorer.compute_score(cands, refs)  # list[float], one per case
        per_system_scores[name] = scores
        results[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_cases": len(scores),
            "per_case": {c: float(s) for c, s in zip(common, scores)},
        }
        print(f"[scored] {name}: mean={results[name]['mean']:.4f}")

    # Paired bootstrap on NeuroFusion-CoT vs heuristic delta.
    delta_ci = None
    if "nf_cot" in per_system_scores and "heuristic" in per_system_scores:
        d, lo, hi = paired_bootstrap_ci(
            per_system_scores["nf_cot"], per_system_scores["heuristic"],
            n_boot=args.n_boot)
        delta_ci = {"delta_nf_cot_minus_heuristic": d, "ci95_lo": lo, "ci95_hi": hi,
                    "n_boot": args.n_boot}

    out = {
        "metric": "RaTEScore",
        "metric_version": "0.6.0",
        "ner_model": "Angelakeke/RaTE-NER-Deberta",
        "embed_model": "FremyCompany/BioLORD-2023-C",
        "text_field": "findings + impression",
        "n_common_cases": len(common),
        "common_case_ids": common,
        "systems": {k: {kk: vv for kk, vv in v.items() if kk != "per_case"}
                    for k, v in results.items()},
        "per_case": {k: v["per_case"] for k, v in results.items()},
        "paired_bootstrap": delta_ci,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    # Table.
    print("\n=== RaTEScore narrative faithfulness (findings + impression) ===")
    print(f"{'system':<12} {'RaTEScore':>10} {'std':>7} {'n':>4}")
    for name in SYSTEMS:
        r = results[name]
        print(f"{name:<12} {r['mean']:>10.4f} {r['std']:>7.4f} {r['n_cases']:>4}")
    if delta_ci:
        print(f"\nNF-CoT - heuristic delta = {delta_ci['delta_nf_cot_minus_heuristic']:+.4f} "
              f"(95% CI [{delta_ci['ci95_lo']:+.4f}, {delta_ci['ci95_hi']:+.4f}], "
              f"{delta_ci['n_boot']} boot)")
    print(f"\n[written] {OUT_JSON}")


if __name__ == "__main__":
    main()
