"""Aggregate non-judge eval across 5 folds.

Reads each fold's predictions_val_fold{N}.jsonl, computes all non-LLM-judge metrics,
prints per-fold and 5-fold mean ± std table, writes a single results JSON.

Skipped (require LLM judge): radfact, green.

Usage:
    python eval_folds.py \
        --pred-root ~/scratch/predictions/val \
        --out ~/scratch/predictions/val/metrics_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from eval import (
    bert_score,
    bleu_rouge_meteor,
    lesion_count_metrics,
    per_field_macro_f1,
    radgraph_f1,
    ratescore,
    schema_adherence,
    section_completeness,
)
from schema import BrainTumorReport, strict_content
from xgrammar_decoder import parse_constrained_output

log = logging.getLogger("neurofusion.eval_folds")


def _sample_score(p: str) -> int:
    """Higher = better. Rank tier:
    3 = passes full schema validation
    2 = passes structural validation (content validators OFF)
    1 = parses as JSON
    0 = otherwise
    Within a tier ties go to whichever sample is encountered first.
    """
    if not p:
        return 0
    # Try JSON parse with bracket/quote repair (covers truncation edge cases).
    parsed = parse_constrained_output(p, BrainTumorReport)
    if parsed is not None:
        return 3
    # Structural-only via temporary content-validator bypass.
    try:
        with strict_content(False):
            BrainTumorReport.model_validate_json(p)
        return 2
    except Exception:
        pass
    # Bare JSON parse (no schema check)
    try:
        import json as _json
        _json.loads(p)
        return 1
    except Exception:
        return 0


def _secondary_score(p: str) -> tuple[int, int, int]:
    """Within-tier tiebreaker for best_of_k.

    Returns a (n_lesions, n_keys, output_len) tuple — larger = better. The
    rationale:
      - n_lesions: tier-3 samples with MORE lesions detected encode richer
        clinical content (assuming they're correctly schema-valid).
      - n_keys: a fuller JSON with more required fields filled beats a
        minimal one.
      - output_len: tiebreaker — longer outputs usually have more findings/
        impression detail.
    """
    if not p:
        return (0, 0, 0)
    try:
        import json as _json
        obj = _json.loads(p)
        n_lesions = len(obj.get("lesions", [])) if isinstance(obj, dict) else 0
        n_keys = len(obj) if isinstance(obj, dict) else 0
        return (n_lesions, n_keys, len(p))
    except Exception:
        return (0, 0, len(p))


def best_of_k(samples: list[str]) -> str:
    """Pick the highest-tier sample (with bracket/quote repair).

    Iterates ALL samples — does not break on first tier-3. Within a tier,
    uses a secondary score (n_lesions, n_keys, output_len) so we pick the
    richest sample, not the first one encountered. Previous "first of K
    that validates" behaviour systematically biased toward earlier (often
    lower-temperature) samples and under-reported what K-sampling could
    achieve. Audit M6.

    Returns the *repaired* JSON string when a tier-3 sample is found via
    parse_constrained_output; otherwise the raw best sample.
    """
    if not samples:
        return ""
    best_p = samples[0]
    best_score = _sample_score(best_p)
    best_secondary = _secondary_score(best_p)
    for p in samples[1:]:
        s = _sample_score(p)
        if s > best_score:
            best_p, best_score, best_secondary = p, s, _secondary_score(p)
        elif s == best_score:
            sec = _secondary_score(p)
            if sec > best_secondary:
                best_p, best_secondary = p, sec
    # If tier-3, return the repaired form so downstream eval gets valid JSON.
    if best_score == 3:
        parsed = parse_constrained_output(best_p, BrainTumorReport)
        if parsed is not None:
            return parsed.model_dump_json()
    return best_p


def load_predictions(path: Path, selection: str = "best_of_k") -> tuple[list[str], list[BrainTumorReport]]:
    """Read predictions JSONL.

    selection:
        "best_of_k"   — pick highest-tier sample per case (repair-aware)
        "first_of_k"  — legacy: take samples[0]
    """
    preds: list[str] = []
    gts: list[BrainTumorReport] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            samples = rec.get("predictions", [])
            if selection == "first_of_k":
                preds.append(samples[0] if samples else "")
            else:
                preds.append(best_of_k(samples))
            gts.append(BrainTumorReport.model_validate_json(rec["gt_report_json"]))
    return preds, gts


def evaluate_fold(pred_path: Path, run_heavy: bool = True, selection: str = "best_of_k") -> dict[str, float]:
    preds, gts = load_predictions(pred_path, selection=selection)
    log.info(f"  {pred_path.name}: {len(preds)} cases (selection={selection})")

    metrics: dict[str, float] = {"n_cases": float(len(preds))}

    metrics.update(schema_adherence(preds))
    metrics.update(section_completeness(preds))
    metrics.update(per_field_macro_f1(preds, gts))
    metrics.update(lesion_count_metrics(preds, gts))
    metrics.update(bleu_rouge_meteor(preds, gts))

    if run_heavy:
        try:
            metrics.update(bert_score(preds, gts))
        except Exception as e:
            log.warning(f"  bert_score skipped: {e}")
        try:
            metrics.update(radgraph_f1(preds, gts))
        except Exception as e:
            log.warning(f"  radgraph_f1 skipped: {e}")
        try:
            metrics.update(ratescore(preds, gts))
        except Exception as e:
            log.warning(f"  ratescore skipped: {e}")

    return metrics


def aggregate(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return {metric: {mean, std, per_fold[]}}."""
    keys = set()
    for d in per_fold:
        keys |= set(d.keys())
    out: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        if k == "n_cases":
            continue
        vals = [d.get(k, float("nan")) for d in per_fold]
        clean = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        if not clean:
            continue
        out[k] = {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "per_fold": [float(v) if isinstance(v, (int, float)) else None for v in vals],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", required=True,
                   help="dir containing fold{N}/predictions_val_fold{N}.jsonl")
    p.add_argument("--out", required=True,
                   help="output JSON with per-fold + aggregate metrics")
    p.add_argument("--skip-heavy", action="store_true",
                   help="skip BERTScore / RadGraph / RaTEScore (CPU-friendly run)")
    p.add_argument("--selection", choices=["best_of_k", "first_of_k"], default="best_of_k",
                   help="how to pick canonical pred from K samples (default: best_of_k)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_root = Path(args.pred_root)
    per_fold: list[dict[str, float]] = []
    for fold in range(5):
        path = pred_root / f"fold{fold}" / f"predictions_val_fold{fold}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            log.warning(f"skipping empty/missing: {path}")
            continue
        log.info(f"=== fold {fold} ===")
        m = evaluate_fold(path, run_heavy=not args.skip_heavy, selection=args.selection)
        if m.get("n_cases", 0) == 0:
            log.warning(f"  fold {fold} has 0 cases, skipping")
            continue
        per_fold.append(m)

    agg = aggregate(per_fold)

    summary = {
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "aggregate": agg,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(f"wrote {out_path}")

    print("\n" + "=" * 78)
    print(f"{'METRIC':<40} {'MEAN':>10} {'STD':>10}   PER-FOLD")
    print("-" * 78)
    for k, v in agg.items():
        per = "  ".join(f"{x:.3f}" if isinstance(x, float) else "—" for x in v["per_fold"])
        print(f"{k:<40} {v['mean']:>10.4f} {v['std']:>10.4f}   {per}")
    print("=" * 78)


if __name__ == "__main__":
    main()
