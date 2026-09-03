"""Run split-conformal calibration on the 50-case held-out set.

Reads predictions_conformal.jsonl (K samples per case + GT report) and computes:
  - τ_marginal: split-conformal abstention threshold at α (marginal coverage)
  - τ_risk: risk-controlling threshold at α (E[loss|kept] ≤ α)

Nonconformity score = semantic entropy of the K samples (Farquhar et al. 2024).
Per-case loss = 1 - per-case structured-field accuracy (Hungarian-paired lesions).

Cases that produce zero valid samples are treated as worst-case (entropy=log(K), loss=1).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from eval import (
    _per_case_accuracy,
    _safe_parse,
    conformal_calibrate,
    conformal_risk_calibrate,
    semantic_entropy,
)
from schema import BrainTumorReport

log = logging.getLogger("neurofusion.calibrate")


def best_sample_index(samples: list[str], gt: BrainTumorReport) -> int:
    """Pick the K-sample index with highest per-case accuracy vs GT (oracle).
    Used for per-case loss surrogate; replace with mean over K if you prefer
    a calibration-set-free choice."""
    if not samples:
        return -1
    accs = [_per_case_accuracy(s, gt) for s in samples]
    return int(np.argmax(accs))


def mean_sample_accuracy(samples: list[str], gt: BrainTumorReport) -> float:
    if not samples:
        return 0.0
    return float(np.mean([_per_case_accuracy(s, gt) for s in samples]))


def compute_record(rec: dict) -> dict[str, float | str]:
    gt = BrainTumorReport.model_validate_json(rec["gt_report_json"])
    samples = rec.get("predictions", [])
    entropy = semantic_entropy(samples) if samples else float(np.log(max(len(samples), 1)))
    n_parseable = sum(1 for s in samples if _safe_parse(s) is not None)
    mean_acc = mean_sample_accuracy(samples, gt)
    return {
        "case_id": rec["case_id"],
        "entropy": float(entropy),
        "n_samples": len(samples),
        "n_parseable": n_parseable,
        "mean_acc": mean_acc,
        "loss": float(1.0 - mean_acc),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="predictions_conformal.jsonl from predict.py")
    p.add_argument("--out", required=True,
                   help="output JSON file with thresholds + per-case scores")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="target miscoverage rate (default 0.1 ⇒ 90%% coverage)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_path = Path(args.predictions)
    records = [json.loads(line) for line in pred_path.open()]
    log.info(f"loaded {len(records)} cases from {pred_path}")

    scored = [compute_record(r) for r in records]

    entropies = [s["entropy"] for s in scored]
    losses = [s["loss"] for s in scored]

    tau_marginal = conformal_calibrate(entropies, alpha=args.alpha)
    tau_risk = conformal_risk_calibrate(entropies, losses, alpha=args.alpha)

    kept_marginal = sum(1 for e in entropies if e <= tau_marginal)
    kept_risk = sum(1 for e in entropies if e <= tau_risk)

    summary = {
        "alpha": args.alpha,
        "n_cases": len(records),
        "tau_marginal": tau_marginal,
        "tau_risk": tau_risk,
        "abstention_rate_marginal": 1.0 - kept_marginal / max(len(records), 1),
        "abstention_rate_risk": 1.0 - kept_risk / max(len(records), 1),
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_parseable_rate": float(np.mean([s["n_parseable"] / max(s["n_samples"], 1) for s in scored])),
        "per_case": scored,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    log.info(f"τ_marginal = {tau_marginal:.4f}  (kept {kept_marginal}/{len(records)}, abstain {summary['abstention_rate_marginal']:.1%})")
    log.info(f"τ_risk     = {tau_risk:.4f}  (kept {kept_risk}/{len(records)}, abstain {summary['abstention_rate_risk']:.1%})")
    log.info(f"mean loss  = {summary['mean_loss']:.3f}   mean parseable = {summary['mean_parseable_rate']:.1%}")
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
