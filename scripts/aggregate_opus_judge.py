#!/usr/bin/env python3
"""Aggregate the blinded Opus-4.8 agent-judge shard scores into per-system
clinical numbers, and compare against the Llama-3.1-8B judge (robustness:
do two independent judges produce the same NF-vs-baselines ranking?).

Reads out/opus_judge_inputs/scores/*.json (per-shard {uid,...5 dims...}) +
out/opus_judge_inputs/uid_map_{gli,men}.json (private uid->system,case_id),
joins, groups by (system,cohort), computes per-dim mean + composite (5-dim
mean) + paired-bootstrap 95% CI, and tabulates Opus vs Llama composites.
"""
from __future__ import annotations
import json, glob, os
import numpy as np

DIMS = ("lesion_id", "characterization", "mass_effect", "ddx", "narrative")
SYS_ORDER = ["NeuroFusion", "NeuroFusion-Mistral", "M3D-LaMed", "LLaVA-Med", "AutoRG-Brain"]
LLAMA = {  # composite_mean from out/crossdata_clinical_*.json (clean)
    ("NeuroFusion", "gli"): 2.937, ("M3D-LaMed", "gli"): 1.500,
    ("LLaVA-Med", "gli"): 2.867, ("AutoRG-Brain", "gli"): 1.767,
    ("NeuroFusion", "men"): 2.513, ("M3D-LaMed", "men"): 2.043,
    ("LLaVA-Med", "men"): 1.993, ("AutoRG-Brain", "men"): 2.057,
}


def main():
    base = "out/opus_judge_inputs"
    uidmap = {}
    for coh in ("gli", "men", "brats2020"):
        p = f"{base}/uid_map_{coh}.json"
        if os.path.exists(p):
            uidmap.update(json.load(open(p)))
    # collect scores
    by = {}  # (system,cohort) -> list of {dim: val}
    missing = 0
    for f in sorted(glob.glob(f"{base}/scores/*.json")):
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        for r in rows:
            uid = r.get("uid")
            meta = uidmap.get(uid)
            if not meta or not all(d in r for d in DIMS):
                missing += 1
                continue
            key = (meta["system"], meta["cohort"])
            by.setdefault(key, []).append({d: float(r[d]) for d in DIMS})

    rng = np.random.default_rng(0)
    out = {"per_system": {}, "comparison": []}
    print(f"{'system':14s} {'coh':3s} {'n':>3s}  " + "  ".join(f"{d[:5]:>5s}" for d in DIMS) + "   comp  [95% CI]      Llama")
    for coh in ("gli", "men", "brats2020"):
        for s in SYS_ORDER:
            rows = by.get((s, coh), [])
            if not rows:
                continue
            arr = {d: np.array([r[d] for r in rows]) for d in DIMS}
            comp_per_case = np.mean(np.stack([arr[d] for d in DIMS]), axis=0)
            boot = np.array([rng.choice(comp_per_case, len(comp_per_case), replace=True).mean()
                             for _ in range(5000)])
            ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
            rec = {"n": len(rows), "composite": round(float(comp_per_case.mean()), 3), "ci95": ci,
                   "per_dim": {d: round(float(arr[d].mean()), 3) for d in DIMS},
                   "llama_composite": LLAMA.get((s, coh))}
            out["per_system"][f"{s}|{coh}"] = rec
            dims_s = "  ".join(f"{rec['per_dim'][d]:5.2f}" for d in DIMS)
            print(f"{s:14s} {coh:3s} {rec['n']:3d}  {dims_s}   {rec['composite']:.3f} {ci}   {LLAMA.get((s,coh))}")
    # ranking agreement
    print("\n=== ranking agreement (composite, higher=better) ===")
    for coh in ("gli", "men", "brats2020"):
        present = [(s, out['per_system'].get(f'{s}|{coh}')) for s in SYS_ORDER]
        present = [(s, r) for s, r in present if r]
        opus_rank = sorted(present, key=lambda x: -x[1]['composite'])
        llama_rank = sorted([(s, LLAMA[(s, coh)]) for s, _ in present if (s, coh) in LLAMA], key=lambda x: -x[1])
        o = [s for s, _ in opus_rank]; l = [s for s, _ in llama_rank]
        out["comparison"].append({"cohort": coh, "opus_rank": o, "llama_rank": l, "nf_top_both": o[0] == "NeuroFusion" == l[0]})
        print(f"  {coh}: Opus={o}")
        print(f"       Llama={l}  | NF#1 in both judges: {o[0]=='NeuroFusion'==l[0]}")
    out["scored_shards"] = len(glob.glob(f"{base}/scores/*.json"))
    out["unmapped_rows"] = missing
    json.dump(out, open("out/opus_judge_summary.json", "w"), indent=2)
    print(f"\nwrote out/opus_judge_summary.json  (shards scored: {out['scored_shards']}, unmapped rows: {missing})")


if __name__ == "__main__":
    main()
