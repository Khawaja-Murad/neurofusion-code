"""Narrative-RRG evaluation for the Track B fair cross-model comparison (paper Table 5).

Extracts findings + impression from each model's output and computes
ROUGE-1, ROUGE-L, METEOR, BLEU-4, BERTScore-F1 vs the GT narrative.

Why this exists:
  The headline "+97.4pp structural validity vs M3D" is a strawman — M3D
  and LLaVA-Med were never trained to emit our JSON schema. Comparing on
  their native task (free-text radiology reports) using narrative
  metrics is the fair Track B comparison. NeuroFusion's JSON output is
  parsed for its findings/impression fields; the baselines' free-text
  outputs are used as-is (with FINDINGS:/IMPRESSION: section splitting
  if the model produced them).

Usage (single model):
  python scripts/eval_narrative.py \\
      --pred ~/scratch/predictions/fold4_k8/predictions_test.jsonl \\
      --model-name NeuroFusion --mode json \\
      --gt out/neurofusion_369.jsonl

Usage (multi-model comparison — produces Table 5):
  python scripts/eval_narrative.py \\
      --gt out/neurofusion_369.jsonl \\
      --out out/narrative_eval_test.json \\
      --entry NeuroFusion json ~/scratch/predictions/fold4_k8/predictions_test.jsonl \\
      --entry M3D-LaMed  narrative ~/scratch/predictions_baselines/m3d_narrative/predictions_test.jsonl \\
      --entry LLaVA-Med  narrative ~/scratch/predictions_baselines/llava_med_narrative/predictions_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("eval_narrative")


# ===========================================================================
# Narrative extraction
# ===========================================================================


_FINDINGS_RE = re.compile(r"(?is)\bfindings?\s*[:\-]\s*(.+?)(?=\bimpression\s*[:\-]|$)")
_IMPRESSION_RE = re.compile(r"(?is)\bimpression\s*[:\-]\s*(.+?)$")


def extract_narrative_from_json(pred: str) -> str | None:
    """Parse pred as JSON and return findings + impression. Returns None on parse failure."""
    try:
        obj = json.loads(pred)
    except (json.JSONDecodeError, ValueError):
        return None
    findings = obj.get("findings", "") or ""
    impression = obj.get("impression", "") or ""
    parts = [s.strip() for s in (findings, impression) if s and s.strip()]
    return " ".join(parts) if parts else None


def extract_narrative_from_text(pred: str) -> str:
    """Pull FINDINGS: / IMPRESSION: sections from free-text. Falls back to the raw string."""
    if not pred:
        return ""
    f_match = _FINDINGS_RE.search(pred)
    i_match = _IMPRESSION_RE.search(pred)
    if f_match or i_match:
        chunks = []
        if f_match:
            chunks.append(f_match.group(1).strip())
        if i_match:
            chunks.append(i_match.group(1).strip())
        return " ".join(chunks)
    return pred.strip()


def extract_narrative(pred: str, mode: str) -> str:
    """Extract findings+impression text from a prediction string.

    mode:
      - "json": parse JSON; return "" on failure (failed JSON has no narrative).
      - "narrative": split on FINDINGS:/IMPRESSION: keywords, else raw string.
      - "auto": try JSON; fall back to narrative if parse fails.
    """
    if mode == "json":
        return extract_narrative_from_json(pred) or ""
    if mode == "narrative":
        return extract_narrative_from_text(pred)
    if mode == "auto":
        out = extract_narrative_from_json(pred)
        if out is not None:
            return out
        return extract_narrative_from_text(pred)
    raise ValueError(f"unknown mode: {mode!r}")


# ===========================================================================
# Metrics
# ===========================================================================


def compute_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Compute corpus + per-example aggregated narrative metrics.

    refs / hyps are aligned. Empty hypothesis strings score 0 on all token
    metrics (zero tokens vs reference). BERTScore handles empty strings
    internally (returns 0 contribution).

    Returns mean+std per metric and a `n` count.
    """
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import bert_score

    # Full battery: ROUGE-1/2/3/4/L, BLEU-1/2/3/4, METEOR, BERTScore-P/R/F1.
    # rouge_score supports arbitrary rougeN; fall back gracefully if a build rejects 3/4.
    R_KEYS = ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL"]
    try:
        scorer = rouge_scorer.RougeScorer(R_KEYS, use_stemmer=True)
    except Exception:
        R_KEYS = ["rouge1", "rouge2", "rougeL"]
        scorer = rouge_scorer.RougeScorer(R_KEYS, use_stemmer=True)
    smooth = SmoothingFunction().method1
    BLEU_W = {"bleu1": (1, 0, 0, 0), "bleu2": (0.5, 0.5, 0, 0),
              "bleu3": (1/3, 1/3, 1/3, 0), "bleu4": (0.25, 0.25, 0.25, 0.25)}

    rouge = {k: [] for k in R_KEYS}
    bleu = {k: [] for k in BLEU_W}
    meteor = []
    for ref, hyp in zip(refs, hyps):
        ref_str = (ref or "").strip()
        hyp_str = (hyp or "").strip()
        # ROUGE-1/2/3/4/L
        rs = scorer.score(ref_str, hyp_str)
        for k in R_KEYS:
            rouge[k].append(rs[k].fmeasure)
        # METEOR (token-level)
        ref_tok = word_tokenize(ref_str.lower())
        hyp_tok = word_tokenize(hyp_str.lower())
        meteor.append(meteor_score([ref_tok], hyp_tok) if hyp_tok else 0.0)
        # BLEU-1..4 sentence-level with smoothing (per-example mean)
        for bk, w in BLEU_W.items():
            bleu[bk].append(
                sentence_bleu([ref_tok], hyp_tok, weights=w, smoothing_function=smooth)
                if hyp_tok else 0.0
            )

    # BERTScore — compute once for the full batch (heavy GPU model). Handle
    # empties by replacing with a single space; bert_score then returns ~0.
    safe_refs = [r if r and r.strip() else " " for r in refs]
    safe_hyps = [h if h and h.strip() else " " for h in hyps]
    # transformers 5.x removed `build_inputs_with_special_tokens` from slow
    # tokenizers; bert_score's sent_encode hits this only on the empty-string
    # branch (counts special-token overhead). Patch sent_encode locally so it
    # uses encode("", add_special_tokens=True) — same return value, future-proof.
    import bert_score.utils as _bsu
    _orig_sent_encode = _bsu.sent_encode
    def _sent_encode_compat(tokenizer, sent):
        if sent.strip() == "":
            return tokenizer.encode("", add_special_tokens=True)
        return _orig_sent_encode(tokenizer, sent)
    _bsu.sent_encode = _sent_encode_compat
    log.info(f"computing BERTScore over {len(safe_hyps)} pairs (this loads roberta-large)...")
    try:
        P, R, F1 = bert_score.score(
            safe_hyps, safe_refs,
            lang="en", verbose=False, rescale_with_baseline=False,
        )
    finally:
        _bsu.sent_encode = _orig_sent_encode
    bert_p = P.cpu().numpy().tolist()
    bert_r = R.cpu().numpy().tolist()
    bert_f1 = F1.cpu().numpy().tolist()

    def _agg(xs: list[float]) -> dict[str, float]:
        arr = np.asarray(xs, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    out: dict[str, Any] = {"n": len(refs)}
    percase: dict[str, list] = {}
    for k in R_KEYS:                       # rouge1/2/3/4/L
        out[k] = _agg(rouge[k]); percase[k] = rouge[k]
    for k in BLEU_W:                       # bleu1/2/3/4
        out[k] = _agg(bleu[k]); percase[k] = bleu[k]
    out["meteor"] = _agg(meteor); percase["meteor"] = meteor
    out["bertscore_precision"] = _agg(bert_p); percase["bertscore_precision"] = bert_p
    out["bertscore_recall"] = _agg(bert_r);    percase["bertscore_recall"] = bert_r
    out["bertscore_f1"] = _agg(bert_f1);       percase["bertscore_f1"] = bert_f1
    out["_per_case"] = percase             # per-example lists for paired-bootstrap CIs
    return out


# ===========================================================================
# IO
# ===========================================================================


def load_gt_narratives(gt_path: Path) -> dict[str, str]:
    """Load GT findings + impression keyed by case_id."""
    out: dict[str, str] = {}
    with gt_path.open() as f:
        for line in f:
            r = json.loads(line)
            findings = r.get("findings", "") or ""
            impression = r.get("impression", "") or ""
            out[r["case_id"]] = " ".join(s.strip() for s in (findings, impression) if s.strip())
    return out


def load_predictions(pred_path: Path, mode: str, sample_idx: int | None = None) -> dict[str, str]:
    """Load predictions JSONL → {case_id: extracted_narrative}.

    If `sample_idx` is None (default), scans all K samples per case and picks
    the first one whose extracted narrative is non-empty. This mirrors the
    best_of_k selection eval_folds.evaluate_fold uses for structural metrics:
    when K>1, picking only sample[0] systematically under-counts the model
    because invalid/truncated samples drag the score down.
    If `sample_idx` is an int, that index is used directly (legacy / debugging).
    """
    out: dict[str, str] = {}
    with pred_path.open() as f:
        for line in f:
            r = json.loads(line)
            preds = r.get("predictions") or []
            if not preds:
                out[r["case_id"]] = ""
                continue
            if sample_idx is not None:
                idx = sample_idx if sample_idx < len(preds) else 0
                out[r["case_id"]] = extract_narrative(preds[idx], mode)
                continue
            # best-of-K: first non-empty extraction wins
            chosen = ""
            for p in preds:
                t = extract_narrative(p, mode)
                if t and t.strip():
                    chosen = t
                    break
            out[r["case_id"]] = chosen
    return out


# ===========================================================================
# Driver
# ===========================================================================


def evaluate_model(model_name: str, pred_path: Path, mode: str,
                   gt: dict[str, str], sample_idx: int = 0) -> dict[str, Any]:
    log.info(f"=== {model_name}  mode={mode}  preds={pred_path} ===")
    preds = load_predictions(pred_path, mode=mode, sample_idx=sample_idx)
    # Align on the intersection of case_ids; preserve eval-set order via GT.
    case_ids = [cid for cid in gt if cid in preds]
    missing = [cid for cid in gt if cid not in preds]
    if missing:
        log.warning(f"  {model_name}: {len(missing)} GT cases missing from preds: {missing[:5]}...")
    refs = [gt[cid] for cid in case_ids]
    hyps = [preds[cid] for cid in case_ids]
    empty = sum(1 for h in hyps if not h.strip())
    log.info(f"  {model_name}: {len(case_ids)} aligned; {empty} have empty extracted narrative")
    log.info(f"  example hyp[0]: {hyps[0][:120]!r}")
    log.info(f"  example ref[0]: {refs[0][:120]!r}")
    m = compute_metrics(refs, hyps)
    m["case_ids"] = case_ids
    m["model"] = model_name
    m["mode"] = mode
    m["pred_path"] = str(pred_path)
    m["empty_count"] = empty
    return m


def print_table(results: list[dict[str, Any]]) -> None:
    cols = ["rouge1", "rougeL", "meteor", "bleu4", "bertscore_f1"]
    width = 16
    header = f"{'Model':<24}{'n':>5}{'empty':>7}" + "".join(f"{c:>{width}}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        row = f"{r['model']:<24}{r['n']:>5}{r['empty_count']:>7}"
        for c in cols:
            row += f"{r[c]['mean']:>{width-4}.3f}±{r[c]['std']:.3f}".rjust(width)
        print(row)
    print("=" * len(header) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Narrative-RRG evaluation (paper Table 5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gt", required=True, type=Path, help="GT JSONL (e.g. out/neurofusion_369.jsonl)")
    p.add_argument("--pred", type=Path, help="single prediction JSONL (with --model-name and --mode)")
    p.add_argument("--model-name", help="label for --pred")
    p.add_argument("--mode", choices=["json", "narrative", "auto"], default="auto",
                   help="extraction mode for --pred")
    p.add_argument("--sample-idx", type=int, default=None,
                   help="force a specific sample index when K>1. Default (None): "
                        "best-of-K = first sample whose extraction is non-empty.")
    p.add_argument("--entry", nargs=3, action="append", metavar=("NAME", "MODE", "PATH"),
                   help="add a model entry: --entry NeuroFusion json /path/to/preds.jsonl. Repeatable.")
    p.add_argument("--out", type=Path, help="optional JSON dump of full results")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    gt = load_gt_narratives(args.gt)
    log.info(f"loaded {len(gt)} GT narratives from {args.gt}")

    entries: list[tuple[str, str, Path]] = []
    if args.pred:
        assert args.model_name, "--model-name required when --pred is set"
        entries.append((args.model_name, args.mode, args.pred))
    if args.entry:
        for name, mode, path in args.entry:
            entries.append((name, mode, Path(path)))
    if not entries:
        p.error("provide --pred (+ --model-name) or --entry")

    results: list[dict[str, Any]] = []
    for name, mode, path in entries:
        results.append(evaluate_model(name, path, mode, gt, sample_idx=args.sample_idx))

    print_table(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Drop per_case lists from JSON dump? keep them — useful for stat tests.
        with args.out.open("w") as f:
            json.dump(results, f, indent=2, default=float)
        log.info(f"wrote full results to {args.out}")


if __name__ == "__main__":
    main()
