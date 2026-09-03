"""Generate K=8 sampled JSON predictions per case on the test + conformal sets.

Loads fold 4's best.pt (highest val macro_F1 = 0.6772) as the production model,
runs the model on each held-out case, and writes per-case JSONL with K samples for
downstream conformal calibration and judge-based eval.

Usage:
    python predict.py \
        --ckpt ~/scratch/runs/fold4/best.pt \
        --mednext-checkpoint ~/scratch/checkpoints/phase1_v5/best.pt \
        --jsonl out/neurofusion_369.jsonl \
        --brats-root $BRATS_ROOT \
        --splits out/splits.json \
        --out-dir out/predictions \
        --k 8
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import NeuroFusionDataset, case_ids_for, collate_fn, load_splits
from model import NeuroFusion, NeuroFusionConfig
from train import load_checkpoint, setup_logging, setup_seed

log = logging.getLogger("neurofusion.predict")


def predict_split(
    model: NeuroFusion,
    loader: DataLoader,
    device: torch.device,
    k: int,
    out_path: Path,
    use_json_constraint: bool = True,
    use_heuristics: bool = True,
    late_fusion: bool = False,
    lambda_per_field: dict[str, float] | None = None,
) -> None:
    model.eval()
    n_total = len(loader.dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if a partial output exists, skip case_ids already written
    # and append, so a walltime-killed long run can be requeued without redoing
    # completed cases. Does not change the per-case inference path.
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as fprev:
            for line in fprev:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["case_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # tolerate a truncated final line from the killed run
        log.info(f"resume: {len(done_ids)} cases already in {out_path}, skipping them")
    open_mode = "a" if done_ids else "w"

    with out_path.open(open_mode) as fout, torch.no_grad():
        for i, batch in enumerate(loader):
            case_ids = batch["case_ids"]
            if all(cid in done_ids for cid in case_ids):
                continue  # already written by a previous (killed) run

            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            reports = batch["reports"]
            report_jsons = batch["report_jsons"]

            t0 = time.time()
            out = model(mri, seg, training=False)

            # Late-fusion (ACL plan §Methodology). Strictly opt-in via --late-fusion.
            # When enabled, convert field-head logits to per-batch per-lesion softmax
            # dicts and pass them down so the constrained decoder biases typed-field
            # value tokens by `lambda_f * log p_head(c)`.
            field_softmax_arg = None
            lambda_per_field_arg = None
            if late_fusion:
                from model import field_logits_to_per_lesion_softmax
                field_softmax_arg = field_logits_to_per_lesion_softmax(
                    out["field_logits"],
                    out["routing"]["batch_idx"],
                    out["routing"]["lesion_idx"],
                )
                lambda_per_field_arg = lambda_per_field

            samples = model.lm.generate(
                visual_tokens=out["visual_tokens"],
                batch_idx=out["routing"]["batch_idx"],
                centroids=out["routing"]["centroids"],
                heuristic_strings=out["heuristic_strings"],
                k_samples=k,
                use_json_constraint=use_json_constraint,
                use_heuristics=use_heuristics,
                case_ids=case_ids,
                field_head_softmax_per_batch=field_softmax_arg,
                lambda_per_field=lambda_per_field_arg,
            )

            hstrings = out["heuristic_strings"]
            for b, cid in enumerate(case_ids):
                rec = {
                    "case_id": cid,
                    "gt_report_json": report_jsons[b],
                    "heuristic_strings": [hstrings[b]] if b < len(hstrings) else [],
                    "predictions": samples[b] if b < len(samples) else [],
                }
                fout.write(json.dumps(rec) + "\n")
            fout.flush()  # observable progress on disk for debugging long-running jobs

            log.info(f"[{(i + 1) * loader.batch_size}/{n_total}] {case_ids} in {time.time() - t0:.1f}s")

    log.info(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best.pt to load (LoRA + Q-Former + field heads)")
    p.add_argument("--mednext-checkpoint", required=True, help="Phase 1 MedNeXt weights")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--brats-root", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--which", choices=["test", "conformal", "both", "val"], default="both")
    p.add_argument("--fold", type=int, default=None,
                   help="required when --which val")
    # Ablation flags (paper §15)
    p.add_argument("--no-json-constraint", action="store_true",
                   help="A1 ablation: disable XGrammar (free-generation baseline)")
    p.add_argument("--no-heuristics", action="store_true",
                   help="A2 ablation: drop the [CLASSIFIER HEURISTICS] line from prompts")
    p.add_argument("--use-stage2-mrope", action="store_true",
                   help="Activate the 4-axis Stage-2 M-RoPE path at inference. REQUIRED for "
                        "checkpoints trained with --use-stage2-mrope (geometry must match training).")
    p.add_argument("--use-global-pool", action="store_true",
                   help="Global-pool ablation: bypass per-lesion connected-component routing; "
                        "the model sees ONE foreground-masked adaptive_avg_pool3d feature. REQUIRED "
                        "for checkpoints trained with --use-global-pool (geometry must match training).")
    # Late-fusion (ACL plan headline mechanism). Strictly opt-in so existing
    # checkpoints reproduce their locked numbers without it.
    p.add_argument("--late-fusion", action="store_true",
                   help="Enable late-fusion of FieldClassificationHead logits into the "
                        "XGrammar-constrained LM decoder. ACL paper headline mechanism.")
    p.add_argument("--lambda-per-field", type=str, default=None,
                   help="JSON dict of per-field lambda values for late-fusion, "
                        "e.g. '{\"composition\":1.5,\"enhancement_pattern\":1.0}'. "
                        "Default: all fields lambda=1.0. lambda=0 disables that field.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    setup_seed(args.seed)
    device = torch.device(args.device)
    cfg = NeuroFusionConfig(use_stage2_mrope=args.use_stage2_mrope,
                            use_global_pool=args.use_global_pool)
    log.info(f"use_stage2_mrope={cfg.use_stage2_mrope} (4-axis M-RoPE {'ACTIVE' if cfg.use_stage2_mrope else 'identity-mode'})")
    log.info(f"use_global_pool={cfg.use_global_pool} ({'GLOBAL-POOL ablation' if cfg.use_global_pool else 'per-lesion routing'})")

    log.info(f"loading model with mednext_checkpoint={args.mednext_checkpoint}")
    model = NeuroFusion(cfg, mednext_checkpoint=args.mednext_checkpoint).to(device)
    log.info(f"loading fine-tuned weights from {args.ckpt}")
    load_checkpoint(Path(args.ckpt), model)
    model.eval()

    splits = load_splits(Path(args.splits))

    splits_to_run = []
    if args.which in ("test", "both"):
        splits_to_run.append(("test", case_ids_for(splits, "test")))
    if args.which in ("conformal", "both"):
        splits_to_run.append(("conformal", case_ids_for(splits, "conformal_calibration")))
    if args.which == "val":
        assert args.fold is not None, "--fold is required when --which val"
        splits_to_run.append((f"val_fold{args.fold}", case_ids_for(splits, "val", fold=args.fold)))

    for name, ids in splits_to_run:
        log.info(f"=== split '{name}': {len(ids)} cases, K={args.k} ===")
        ds = NeuroFusionDataset(
            Path(args.jsonl), Path(args.brats_root),
            case_ids=ids, cache_volumes=False, target_shape=(128, 128, 128),
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        lambda_per_field = None
        if args.lambda_per_field:
            lambda_per_field = json.loads(args.lambda_per_field)
        predict_split(
            model, loader, device, args.k,
            out_dir / f"predictions_{name}.jsonl",
            use_json_constraint=not args.no_json_constraint,
            use_heuristics=not args.no_heuristics,
            late_fusion=args.late_fusion,
            lambda_per_field=lambda_per_field,
        )

    log.info("done")


if __name__ == "__main__":
    main()
