#!/usr/bin/env python3
"""Acquisition-confound probe — the committed, reproducible replacement for F6a.

WHY THIS EXISTS
---------------
`out/m6_research/F6a_residualization_glimen.json` reported connector_IN GLI-vs-MEN 0.950 -> 0.623 and
was used to demote diagnosis project-wide. It had **no script**, its nuisance regression was fit
**in-sample**, and it had **no null**. Independent verification (10 CV seeds, 1000 permutations)
decomposed the reported 0.318 "collapse" as:

    raw connector_IN                                   0.9390
      - mechanical cost of removing ANY 24 dof   ->    0.8356   (permuted null)
      - acquisition-specific cost                ->    0.7653   (honest, p = 0.005)
      - in-sample overfitting artifact           ->    0.6212   (F6a as published)

i.e. only ~22% of the published drop is acquisition-attributable. The confound is REAL and
significant, but SMALL, and 0.765 is +0.239 over the 0.526 majority class.

THE PRIMARY INSTRUMENT IS THE SYMMETRIC TEST, NOT RESIDUALIZATION
-----------------------------------------------------------------
Residualization is a poor instrument on this data:
  * `Xin` has participation-ratio effective rank ~2.78 (top-2 PCs = 80% of variance). Residualizing a
    rank-~3 matrix on ANY 24-dim nuisance is close to rank-annihilating -- a random 24-dim projection
    already achieves R^2 ~ 0.99.
  * Residualization is NOT subtractive: regressing a binary indicator on 24 continuous predictors can
    RAISE accuracy (observed: 0.477 -> 0.519).

So the headline is the SYMMETRIC pair:
    features residualized on acquisition   (how much survives)
    acquisition residualized on features   (how much acquisition adds that features lack)
If the second is at chance while the first is well above it, the nuisance is a SUBSET of the signal and
the diagnosis claim survives. That conclusion needs no null model and replicates across feature dumps.

USAGE
    python scripts/confound_probe.py --pair GLI MEN
    python scripts/confound_probe.py --three-way --n-perm 1000
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
COHORTS = ("GLI", "MEN", "MET")
# The 24-dim nuisance F6a used: 4 modalities x 6 stats. A SUBSET of the 59-feature fair probe.
MODS = ("t1n", "t1c", "t2w", "t2f")
STATS = ("mean", "std", "skew", "q05", "q50", "q95")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_ids() -> dict[str, set[str]]:
    p = REPO / "out" / "splits_eval_canonical.json"
    s = json.loads(p.read_text())
    return {c: set(s[f"test_{c.lower()}"]) for c in COHORTS}


def load_data(npz_path: Path, nuisance_path: Path, cohorts: tuple[str, ...]):
    """Returns X (features), N (nuisance), y, ids -- intersected on the CANONICAL basis.

    The intersection is EXPLICIT. F6a's n=114 was legal only by accident: intersecting the npz with
    the nuisance cache happened to trim GLI from 60 to the canonical 54. Never rely on that.
    """
    d = np.load(npz_path, allow_pickle=True)
    key = "Xin" if "Xin" in d else "feat_in_dom"
    X_all = np.asarray(d[key], dtype=np.float64)
    ids_all = [str(i) for i in np.asarray(d["ids"]).tolist()]

    nz = json.loads(nuisance_path.read_text())
    recs = nz["records"] if isinstance(nz, dict) and "records" in nz else nz
    nmap, nkeys = {}, None
    for r in recs:
        feats = r["feats"]
        if nkeys is None:
            nkeys = [k for k in feats if any(k.startswith(m + "_") for m in MODS)
                     and k.split("_", 1)[1] in STATS]
            nkeys.sort()
        nmap[r["id"]] = [feats[k] for k in nkeys]

    canon = canonical_ids()
    allowed = set().union(*(canon[c] for c in cohorts))
    keep = [i for i, cid in enumerate(ids_all) if cid in allowed and cid in nmap]

    ids = [ids_all[i] for i in keep]
    X = X_all[keep]
    N = np.asarray([nmap[c] for c in ids], dtype=np.float64)
    y = np.asarray([cohorts.index(next(c for c in cohorts if cid in canon[c])) for cid in ids])
    return X, N, y, ids, nkeys


def _clf(kind: str):
    est = (RidgeClassifier() if kind == "ridge"
           else LogisticRegression(class_weight="balanced", max_iter=5000))
    return Pipeline([("sc", StandardScaler()), ("clf", est)])


def cv_acc(X, y, seed: int, kind: str) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    ok = 0
    for tr, te in cv.split(X, y):
        m = _clf(kind).fit(X[tr], y[tr])
        ok += int((m.predict(X[te]) == y[te]).sum())
    return ok / len(y)


def cv_acc_residualized(X, N, y, seed: int, kind: str) -> float:
    """OOF residualization: the nuisance regression is fit on TRAIN ROWS ONLY, inside each fold.

    F6a fit it on all rows and then cross-validated the residuals, which inflated the apparent
    collapse by ~0.144 accuracy points (~45% of the entire published drop).
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    ok = 0
    for tr, te in cv.split(X, y):
        reg = LinearRegression().fit(N[tr], X[tr])
        Rtr = X[tr] - reg.predict(N[tr])
        Rte = X[te] - reg.predict(N[te])
        m = _clf(kind).fit(Rtr, y[tr])
        ok += int((m.predict(Rte) == y[te]).sum())
    return ok / len(y)


def multi_seed(fn, *a, seeds=range(10), **kw) -> tuple[float, float, list[float]]:
    v = [fn(*a, seed=s, **kw) for s in seeds]
    return float(np.mean(v)), float(np.std(v)), v


def main() -> int:
    # ☠️☠️ GATED 2026-08-11 — see scripts/held_out_gate.py. Body preserved and unreachable.
    from held_out_gate import refuse
    return refuse(
        "confound_probe.py",
        "canonical_ids() (l.63-66) reads test_gli/test_men/test_met and load_data (l.91-98)\n"
        "restricts the feature matrix to exactly those ids. cv_acc / cv_acc_residualized\n"
        "(l.108-131) then run 5-fold CV FIT AND SCORED INSIDE the held-out cases, x10 seeds\n"
        "(l.134-136), producing four accuracies (l.166-169) plus a 1000-draw permuted null.\n"
        "Fitting a classifier on the test cases and reporting its accuracy is a look.",
        "★ ITS npz EXISTS TODAY and it needs no GPU — sklearn over a cached feature dump.\n"
        "▶ This measures a property of the DATA (is cohort predictable from acquisition\n"
        "  signature?), not of the MODEL. That is a weaker form of test-set adaptation than\n"
        "  scoring a checkpoint, and I am not pretending otherwise — but it still produces a\n"
        "  fresh number on the held-out cases that has already moved a project-wide decision\n"
        "  (it demoted diagnosis), so it is counted rather than waved through.")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=str(REPO / ".claude/worktrees/agent-ab7d12f61b0d0ce9b"
                                              / "out/m6_research/gate_c_mistral_features.npz"),
                    help="feature dump. NOTE two different files share this basename across the two "
                         "trees (main: GLI 60 LEAKY / MET 36; $WT: GLI 54 clean / MET 60). Pinned, "
                         "and its md5 is recorded in the output.")
    ap.add_argument("--nuisance", default=str(REPO / "out/m6_research/cohort_source_leak_features.json"))
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--three-way", action="store_true")
    ap.add_argument("--clf", choices=["logreg", "ridge"], default="logreg")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "out/m6_research/confound_probe_result.json"))
    a = ap.parse_args()

    cohorts = tuple(COHORTS) if a.three_way else tuple(a.pair or ("GLI", "MEN"))
    npz_p, nui_p = Path(a.npz), Path(a.nuisance)
    X, N, y, ids, nkeys = load_data(npz_p, nui_p, cohorts)
    seeds = range(a.seeds)
    maj = float(np.bincount(y).max() / len(y))

    print(f"# confound_probe | cohorts={'/'.join(cohorts)} n={len(y)} "
          f"X={X.shape} N={N.shape} majority={maj:.4f} clf={a.clf}")
    print(f"# npz={npz_p}  md5={md5(npz_p)[:12]}")

    raw_m, raw_s, _ = multi_seed(cv_acc, X, y, seeds=seeds, kind=a.clf)
    acq_m, acq_s, _ = multi_seed(cv_acc, N, y, seeds=seeds, kind=a.clf)
    res_m, res_s, _ = multi_seed(cv_acc_residualized, X, N, y, seeds=seeds, kind=a.clf)
    rev_m, rev_s, _ = multi_seed(cv_acc_residualized, N, X, y, seeds=seeds, kind=a.clf)

    print(f"\n  features alone                        {raw_m:.4f} +/- {raw_s:.4f}")
    print(f"  acquisition alone                     {acq_m:.4f} +/- {acq_s:.4f}")
    print(f"  features residualized on acquisition  {res_m:.4f} +/- {res_s:.4f}   <- how much survives")
    print(f"  acquisition residualized on features  {rev_m:.4f} +/- {rev_s:.4f}   <- PRIMARY: at chance?")
    print(f"  chance / majority                     {maj:.4f}")

    # Permuted-nuisance null: how much accuracy is lost by removing ANY 24 dimensions?
    rng = np.random.default_rng(0)
    null = []
    for i in range(a.n_perm):
        Np = N[rng.permutation(len(N))]
        null.append(cv_acc_residualized(X, Np, y, seed=i % a.seeds, kind=a.clf))
    null = np.asarray(null)
    nm, ns = float(null.mean()), float(null.std())
    z = (res_m - nm) / ns if ns else float("nan")
    p_one = float((null <= res_m).mean())

    print(f"\n  permuted-nuisance null                {nm:.4f} +/- {ns:.4f}  ({a.n_perm} reps)")
    print(f"  acquisition-attributable gap          {nm - res_m:+.4f}   z={z:.2f}  p(one-sided)={p_one:.4f}")
    share = (nm - res_m) / (raw_m - res_m) if raw_m > res_m else float("nan")
    print(f"  share of the raw->resid drop that is acquisition: {share:.1%}")

    verdict = ("acquisition is a SUBSET of the feature signal (reverse test at chance) -- diagnosis is "
               "NOT reducible to scanner signature"
               if rev_m <= maj + 2 * max(rev_s, 1e-9) else
               "acquisition carries information the features lack -- investigate")
    print(f"\n  VERDICT: {verdict}")

    out = {
        "cohorts": list(cohorts), "n": int(len(y)), "majority": round(maj, 4),
        "classifier": a.clf, "n_seeds": a.seeds, "n_perm": a.n_perm,
        "npz": str(npz_p), "npz_md5": md5(npz_p), "nuisance": str(nui_p),
        "nuisance_dims": len(nkeys), "nuisance_features": nkeys,
        "features_alone": [round(raw_m, 4), round(raw_s, 4)],
        "acquisition_alone": [round(acq_m, 4), round(acq_s, 4)],
        "features_residualized_on_acquisition": [round(res_m, 4), round(res_s, 4)],
        "PRIMARY_acquisition_residualized_on_features": [round(rev_m, 4), round(rev_s, 4)],
        "permuted_null": [round(nm, 4), round(ns, 4)],
        "acquisition_gap": round(nm - res_m, 4), "z": round(z, 3), "p_one_sided": round(p_one, 4),
        "acquisition_share_of_drop": None if np.isnan(share) else round(share, 4),
        "verdict": verdict,
        "note": ("Residualization is a WEAK instrument here: Xin effective rank ~2.78 vs a 24-dim "
                 "nuisance is near rank-annihilating, and residualization is not subtractive. The "
                 "symmetric (reverse) test is the primary result."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
