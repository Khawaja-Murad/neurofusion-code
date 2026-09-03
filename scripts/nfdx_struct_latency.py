#!/usr/bin/env python3
"""What do the structural dx features COST, in wall-clock seconds, on the service path?

The question Khawaja asked: the structural-feature head reaches 0.876/MET 0.842 -- what is
the latency with those numbers in? This measures the MARGINAL cost only, because everything
upstream (trunk forward, tau/cc filtering, the native label map) is already paid for by the
segmentation the service produces anyway.

So the marginal cost is exactly two calls:
    env, eedt = envelope(t1ce)          # once per case, label-independent
    x         = case_features(pred,...)  # once per case, on the PREDICTED label map

NOT timed (deliberately, with the reason stated):
  - disk I/O of the volumes. The service already holds these arrays in memory; charging
    nii.gz decompression to the feature head would inflate the answer by the cost of
    something the service does not do at this point in the pipeline.
  - `case_features(gt, ...)`. The offline job computes GT and PRED features; the service
    has no GT. Timing both would DOUBLE the reported cost.
  - the head's own forward pass (logreg / hgb over 363 dims) -- measured separately below,
    and it is microseconds, but it is measured rather than asserted.

☠️ SINGLE-THREADED BY CONSTRUCTION. The offline job used 16 workers across CASES, which
says nothing about one case's latency. A service handles one case at a time, so the
per-case serial cost is the only figure that answers the question. OMP/BLAS threads are
pinned to 1 by the wrapper so a threaded BLAS cannot silently flatter the number.

Reports median AND p95 with n, because Khawaja's 100 s ruling is a distribution question
and a median alone hid 4/7 over-budget cases once already.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SCRATCH = Path(os.path.expanduser("~/scratch"))
VAL267_CACHE = SCRATCH / "nf_frozen/probs_val267"


def _percentile(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stratified", action="store_true",
                    help="round-robin across cohorts (MET is the worst case for the "
                         "per-lesion loop and rcs[:n] draws GLI only)")
    ap.add_argument("--out", default=str(REPO / "out/NFDX_STRUCT_LATENCY.json"))
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v) != "1":
            print(f"FATAL: {v}={os.environ.get(v)!r}, expected '1' -- a threaded BLAS "
                  f"would understate the serial per-case cost", file=sys.stderr)
            return 2

    from nfdx_clinical_struct import (case_features, envelope, val267_roster,
                                      _service_multiclass, TRUNK_SHA)
    from clean_trunk_met60_baseline import load_case

    rcs, lab = val267_roster()
    if a.stratified:
        # ☠️ The first run took rcs[:12] and drew 12/12 GLI. `case_features` loops over
        # LESIONS (marching cubes per component), so MET -- the multi-lesion cohort -- is
        # the worst case and was the one cohort NOT measured. Sample round-robin across
        # cohorts so the reported p95 describes the cohort that actually stresses the loop.
        by = {}
        for rc, y in zip(rcs, lab):
            by.setdefault(int(y), []).append(rc)
        order, k = [], 0
        while len(order) < a.n and any(v[k:] for v in by.values()):
            for c in sorted(by):
                if k < len(by[c]) and len(order) < a.n:
                    order.append(by[c][k])
            k += 1
        rcs = order
        print(f"[latency] stratified draw over cohorts {sorted(by)}: "
              f"{ {c: sum(1 for r in rcs if r in set(by[c])) for c in sorted(by)} }", flush=True)
    else:
        rcs = rcs[: a.n]
    print(f"[latency] n={len(rcs)} cases, single-threaded, service wiring "
          f"(_service_multiclass)", flush=True)

    t_env, t_feat, t_pred, shapes, rows = [], [], [], [], []
    for i, rc in enumerate(rcs):
        stem = rc.split("/")[-1]
        z = np.load(VAL267_CACHE / f"{stem}.npz", allow_pickle=True)
        if str(z["trunk_sha256"]) != TRUNK_SHA:
            raise RuntimeError(f"{rc}: cache trunk identity mismatch")
        mri, gt, sp = load_case(SCRATCH / rc)
        argmax = z["argmax"].astype(np.int64)
        p_fg = 1.0 - z["p_bg"].astype(np.float32)
        m0 = np.asarray(mri[0] if mri.ndim == 4 else mri, dtype=np.float32)

        # --- everything above this line is I/O the service has already done ---
        t0 = time.perf_counter()
        pred = _service_multiclass(argmax, p_fg, gt.shape, sp)
        t1 = time.perf_counter()
        env, eedt = envelope(m0, sp)
        t2 = time.perf_counter()
        x = case_features(pred, env, eedt, sp)
        t3 = time.perf_counter()

        t_pred.append(t1 - t0)
        t_env.append(t2 - t1)
        t_feat.append(t3 - t2)
        shapes.append(list(map(int, pred.shape)))
        nz = int((pred > 0).sum())
        from lesion_detection_metrics import connected_components
        _, ncc = connected_components(pred > 0, 6)
        rows.append({"case": stem, "n_components": int(ncc), "fg_voxels": nz,
                     "label_map_s": t1 - t0, "envelope_s": t2 - t1,
                     "features_s": t3 - t2,
                     "finite": int(np.isfinite(x).sum()), "n_features": int(x.size)})
        print(f"  {i+1}/{len(rcs)} {stem}  cc={ncc} fg_vox={nz}  "
              f"label_map {t1-t0:.2f}s  envelope {t2-t1:.2f}s  features {t3-t2:.2f}s  "
              f"finite={int(np.isfinite(x).sum())}/{x.size}", flush=True)

    # the head forward itself -- measured, not asserted
    rng = np.random.default_rng(0)
    from sklearn.linear_model import LogisticRegression
    Xtr = rng.normal(size=(500, 363)).astype(np.float64)
    ytr = rng.integers(0, 3, size=500)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    xq = rng.normal(size=(1, 363))
    t0 = time.perf_counter()
    for _ in range(1000):
        clf.predict_proba(xq)
    t_head = (time.perf_counter() - t0) / 1000.0

    tot = [p + e + f for p, e, f in zip(t_pred, t_env, t_feat)]
    res = {
        "n": len(rcs),
        "single_threaded": True,
        "wiring": "service_multiclass",
        "note": "MARGINAL cost only; volume I/O and GT features excluded by design",
        "label_map_s": {"median": _percentile(t_pred, 50), "p95": _percentile(t_pred, 95)},
        "envelope_s": {"median": _percentile(t_env, 50), "p95": _percentile(t_env, 95)},
        "features_s": {"median": _percentile(t_feat, 50), "p95": _percentile(t_feat, 95)},
        "head_forward_s": t_head,
        "TOTAL_marginal_s": {"median": _percentile(tot, 50), "p95": _percentile(tot, 95),
                             "min": float(min(tot)), "max": float(max(tot))},
        "shapes": shapes,
        "per_case": rows,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n=== MARGINAL COST OF THE STRUCTURAL DX FEATURES (n={len(rcs)}) ===")
    print(f"  label map (service rule) median {res['label_map_s']['median']:.2f}s  "
          f"p95 {res['label_map_s']['p95']:.2f}s")
    print(f"  envelope + EDT          median {res['envelope_s']['median']:.2f}s  "
          f"p95 {res['envelope_s']['p95']:.2f}s")
    print(f"  43 features             median {res['features_s']['median']:.2f}s  "
          f"p95 {res['features_s']['p95']:.2f}s")
    print(f"  head forward            {t_head*1000:.3f} ms")
    print(f"  ★ TOTAL                 median {res['TOTAL_marginal_s']['median']:.2f}s  "
          f"p95 {res['TOTAL_marginal_s']['p95']:.2f}s  "
          f"(min {res['TOTAL_marginal_s']['min']:.2f} max {res['TOTAL_marginal_s']['max']:.2f})")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
