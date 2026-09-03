"""Paired-bootstrap significance test for a candidate NeuroFusion prediction
file vs locked baseline prediction files. Uses the canonical Hungarian
per-field macro-F1 (eval.per_field_macro_f1), matching paper Tables 5+6.

Usage:
  python scripts/bootstrap_significance.py \
      --pred ~/scratch/phase_e_results/hybrid_combined_smart_test_polished.jsonl \
      --label "NF Phase E (smart polished)" \
      --baseline-pred "M3D-LaMed=PATH_TO_M3D_PRED.jsonl" \
      --baseline-pred "LLaVA-Med=PATH_TO_LLAVA_PRED.jsonl" \
      --baseline-pred "Heuristic=out/heuristic_baseline_test.jsonl" \
      --baseline-pred "NF locked=out/hybrid_test_locked.jsonl" \
      --n-boot 10000 --seed 0 \
      --out out/bootstrap_phase_e.json

Bootstrap: resample the cases with replacement B times; for each resample
recompute Hungarian per-field macro_F1 for the candidate and each baseline.
Report the distribution of (F1_NF − F1_baseline) and a two-sided p-value
(2× tail probability that the difference flips sign from the observed point
estimate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval import per_field_macro_f1
from schema import BrainTumorReport, strict_content


def load_paired(pred_jsonl: Path, ref_case_ids: list[str] | None = None):
    """Return (case_ids, pred_strs, gt_objs). If ref_case_ids supplied, align
    output ordering to it (drop unknown cases, raise on missing)."""
    by_id: dict[str, tuple[str, BrainTumorReport]] = {}
    with pred_jsonl.open() as f, strict_content(False):
        for line in f:
            rec = json.loads(line)
            cid = rec.get("case_id")
            gt_str = rec.get("gt_report_json")
            p_str = (rec.get("predictions") or [None])[0]
            if not gt_str or not cid:
                continue
            try:
                gt = BrainTumorReport.model_validate_json(gt_str)
            except Exception:
                continue
            by_id[cid] = (p_str or "{}", gt)

    if ref_case_ids is None:
        items = sorted(by_id.items())
        return [c for c, _ in items], [v[0] for _, v in items], [v[1] for _, v in items]

    cids, preds, gts = [], [], []
    for c in ref_case_ids:
        if c in by_id:
            cids.append(c)
            preds.append(by_id[c][0])
            gts.append(by_id[c][1])
    return cids, preds, gts


def macro_f1_indices(preds: list[str], gts: list[BrainTumorReport], idx: np.ndarray) -> float:
    """Compute Hungarian per-field macro-F1 on a resampled index list.
    Uses strict_content(False) so non-NF model outputs (which may fail strict
    narrative content validators) are still parsed for their structured fields."""
    sub_p = [preds[i] for i in idx]
    sub_g = [gts[i] for i in idx]
    with strict_content(False):
        r = per_field_macro_f1(sub_p, sub_g, use_hungarian=True)
    return float(r["_overall"]["macro_f1"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="NF candidate predictions JSONL")
    p.add_argument("--label", default="NF candidate", help="Display label")
    p.add_argument("--baseline-pred", action="append", default=[],
                   help="Repeated: name=PATH for each baseline JSONL to bootstrap against")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load NF candidate, use its case_ids as the canonical ordering.
    cids_nf, preds_nf, gts_nf = load_paired(Path(args.pred))
    n = len(cids_nf)
    print(f"NF candidate: {n} cases from {args.pred}")
    summary = {
        "candidate": {"label": args.label, "pred": str(args.pred), "n_cases": n},
        "comparisons": [],
    }

    # Single-model bootstrap on candidate
    rng = np.random.default_rng(args.seed)
    pt_nf = macro_f1_indices(preds_nf, gts_nf, np.arange(n))
    boots_nf = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        boots_nf.append(macro_f1_indices(preds_nf, gts_nf, idx))
    ci_nf = (float(np.percentile(boots_nf, 2.5)), float(np.percentile(boots_nf, 97.5)))
    summary["candidate"]["bootstrap"] = {
        "f1": pt_nf, "f1_ci": ci_nf, "n_boot": args.n_boot,
    }
    print(f"  macro_F1 = {pt_nf:.4f}  95% CI [{ci_nf[0]:.4f}, {ci_nf[1]:.4f}]")

    # Paired comparisons (use same resamples across all baselines for consistency)
    rng_all = np.random.default_rng(args.seed + 1)
    resamples = [rng_all.integers(0, n, size=n) for _ in range(args.n_boot)]

    for spec in args.baseline_pred:
        if "=" not in spec:
            raise ValueError(f"--baseline-pred expects name=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        cids_b, preds_b, gts_b = load_paired(Path(path), ref_case_ids=cids_nf)
        if len(cids_b) != n:
            print(f"  SKIP {name}: aligned {len(cids_b)}/{n} case_ids")
            continue
        pt_b = macro_f1_indices(preds_b, gts_b, np.arange(n))
        # Paired bootstrap
        f1_nf_b, f1_b_b, delta_b = [], [], []
        for idx in resamples:
            a = macro_f1_indices(preds_nf, gts_nf, idx)
            b = macro_f1_indices(preds_b, gts_b, idx)
            f1_nf_b.append(a); f1_b_b.append(b); delta_b.append(a - b)
        delta_arr = np.asarray(delta_b)
        obs = pt_nf - pt_b
        if obs > 0:
            p_two = float((delta_arr <= 0).mean()) * 2
        elif obs < 0:
            p_two = float((delta_arr >= 0).mean()) * 2
        else:
            p_two = 1.0
        p_two = min(1.0, p_two)
        res = {
            "baseline_label": name,
            "baseline_pred": path,
            "n_cases": n,
            "f1_nf": pt_nf,
            "f1_baseline": pt_b,
            "delta": obs,
            "delta_mean": float(delta_arr.mean()),
            "delta_std": float(delta_arr.std()),
            "delta_ci": (float(np.percentile(delta_arr, 2.5)),
                         float(np.percentile(delta_arr, 97.5))),
            "p_two_sided": p_two,
        }
        summary["comparisons"].append(res)
        sig = "**" if p_two < 0.01 else ("*" if p_two < 0.05 else "n.s.")
        print(f"  vs {name}: NF={pt_nf:.4f} {name}={pt_b:.4f}  "
              f"Δ={obs:+.4f} 95% CI [{res['delta_ci'][0]:+.4f}, {res['delta_ci'][1]:+.4f}]  "
              f"p={p_two:.4f} {sig}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
