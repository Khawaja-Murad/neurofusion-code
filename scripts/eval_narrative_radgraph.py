#!/usr/bin/env python
"""RadGraph-F1 narrative faithfulness eval (mirrors eval_narrative_rate_v2.py).

RadGraph-F1 (Jain et al. 2021) is the field-standard entity+relation graph metric
for radiology report generation. It was previously BLOCKED in this env: radgraph
0.1.0 vendors an AllenNLP whose optimizers.py references `transformers.AdamW`, an
alias modern transformers removed -> ImportError at module load. That alias was
just torch.optim.AdamW and is only touched at class-definition (never used for
inference scoring), so we shim it before importing radgraph. The RadGraph model
weights (radgraph.tar.gz) download on first use (login node has internet).

Methodology is identical to the RaTEScore v2 re-scorer so the two narrative
metrics are directly comparable:
  * per-case scoring (one hyp/ref pair at a time) -> a drop can never misalign;
  * an empty/failed candidate scores 0.0 explicitly (NOT dropped);
  * paired bootstrap CIs for nf_cot vs {heuristic, nf_single, m3d, llava} on the
    SAME aligned common set;
  * leak-free, no GT-seg, no oracle selection (deterministic first parse-valid pred).

If the RadGraph model cannot be fetched/instantiated, this exits non-zero WITHOUT
writing partial output -- we never fabricate a metric.

Usage:
  source neuro_venv/bin/activate
  python scripts/eval_narrative_radgraph.py
"""
import argparse
import json
import os
import sys

import numpy as np

# --- shim: modern transformers removed the AdamW alias that radgraph's vendored
#     AllenNLP references at import time. It was torch.optim.AdamW. ---
import torch
import transformers
if not hasattr(transformers, "AdamW"):
    transformers.AdamW = torch.optim.AdamW

# --- shim 2: radgraph's vendored AllenNLP calls
#     AutoTokenizer.from_pretrained(model_name, add_special_tokens=False, ...).
#     Modern transformers rejects add_special_tokens as an init kwarg
#     ("conflicts with the method add_special_tokens in BertTokenizer"). It only
#     governs encode-time behaviour, which AllenNLP sets explicitly per-call
#     anyway, so dropping it at init is safe. ---
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained.__func__
def _patched_from_pretrained(cls, *a, **kw):
    kw.pop("add_special_tokens", None)
    return _orig_from_pretrained(cls, *a, **kw)
transformers.AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_narrative_rate import (  # noqa: E402
    SPLITS, SYSTEMS, load_gt, load_system,
)

OUT_JSON = os.path.join(
    os.environ.get("NF_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "out", "narrative_radgraph.json")


def paired_bootstrap_ci(a, b, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), f"unpaired arrays {len(a)} vs {len(b)}"
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float(min(2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean()), 1.0))
    return float(a.mean() - b.mean()), float(lo), float(hi), p


def score_one(f1, cand, ref):
    """RadGraph-F1 'partial' reward for ONE hyp/ref pair. f1() returns
    (mean_reward, reward_list, hyp_ann, ref_ann); reward_list[0] is the pair score."""
    out = f1(hyps=[cand], refs=[ref])
    reward_list = out[1]
    return float(reward_list[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-level", default="partial",
                    choices=["simple", "partial", "complete"])
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    # Instantiate FIRST: if the model can't be fetched, fail loudly (no fabrication).
    try:
        from radgraph import F1RadGraph
        f1 = F1RadGraph(reward_level=args.reward_level, model_type="radgraph")
    except Exception as e:
        print(f"[FATAL] RadGraph model unavailable -> cannot score: {e}", file=sys.stderr)
        sys.exit(2)

    test_ids = json.load(open(SPLITS))["test"]
    gt = load_gt()
    sys_texts = {name: load_system(p) for name, p in SYSTEMS.items()}

    common = [c for c in test_ids
              if c in gt and all(c in sys_texts[s] for s in SYSTEMS)]
    common.sort()
    refs = [gt[c] for c in common]
    print(f"[info] common across GT + all systems: {len(common)}")

    failures = {name: [c for c in common if not sys_texts[name].get(c, "").strip()]
                for name in SYSTEMS}
    for name, fc in failures.items():
        if fc:
            print(f"[warn] {name}: {len(fc)} empty/failed case(s) -> scored 0.0: {fc}")

    per_system_scores = {}
    for name in SYSTEMS:
        scores = []
        for cid, ref in zip(common, refs):
            cand = sys_texts[name].get(cid, "").strip()
            if not cand or not ref.strip():
                scores.append(0.0)
                continue
            scores.append(score_one(f1, cand, ref))
        assert len(scores) == len(common), f"{name}: {len(scores)} != {len(common)}"
        per_system_scores[name] = scores
        arr = np.asarray(scores)
        print(f"[scored] {name}: mean={arr.mean():.4f} std={arr.std():.4f} n={len(arr)}")

    comparisons = {}
    for other in ("heuristic", "nf_single", "m3d", "llava"):
        if "nf_cot" in per_system_scores and other in per_system_scores:
            d, lo, hi, p = paired_bootstrap_ci(
                per_system_scores["nf_cot"], per_system_scores[other], n_boot=args.n_boot)
            comparisons[f"nf_cot_minus_{other}"] = {
                "delta": d, "ci95_lo": lo, "ci95_hi": hi, "p_two_sided": p,
                "significant": bool(lo > 0 or hi < 0),
            }

    out = {
        "metric": "RadGraph-F1", "reward_level": args.reward_level,
        "model_type": "radgraph", "text_field": "findings + impression",
        "scoring": "per-case (alignment-safe); empty candidate -> 0.0",
        "n_common_cases": len(common), "common_case_ids": common,
        "failures_scored_zero": failures,
        "systems": {name: {
            "mean": float(np.mean(per_system_scores[name])),
            "std": float(np.std(per_system_scores[name])),
            "n_cases": len(per_system_scores[name]),
        } for name in SYSTEMS},
        "per_case": {name: {c: per_system_scores[name][i] for i, c in enumerate(common)}
                     for name in SYSTEMS},
        "comparisons_vs_nf_cot": comparisons,
        "n_boot": args.n_boot,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n=== RadGraph-F1 ({args.reward_level}, n={len(common)}) ===")
    print(f"{'system':<12} {'RadGraphF1':>11} {'std':>7} {'n':>4}")
    for name in SYSTEMS:
        s = out["systems"][name]
        print(f"{name:<12} {s['mean']:>11.4f} {s['std']:>7.4f} {s['n_cases']:>4}")
    print("\n--- nf_cot paired deltas (CI excludes 0 => significant) ---")
    for k, v in comparisons.items():
        flag = "SIG" if v["significant"] else "tie"
        print(f"{k:<28} delta={v['delta']:+.4f}  CI[{v['ci95_lo']:+.4f},{v['ci95_hi']:+.4f}]  "
              f"p={v['p_two_sided']:.4f}  [{flag}]")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
