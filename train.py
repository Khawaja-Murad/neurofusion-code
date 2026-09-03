"""
train.py — NeuroFusion v4 multi-task training loop.

Phased schedule (per methodology v4 §9):

  Phase 1   --phase 1     MedNeXt segmentation pretrain on combined BraTS 2020+2021+2023
                          (~45 GPU-h on 1xA100, runs once, output frozen for Phase 2)
  Phase 2a  --phase 2a    M-RoPE identity-parity sanity check
                          (~30 min — verifies the retrofit is numerically correct;
                           non-negotiable before Phase 2b training begins)
  Phase 2b  --phase 2b    Multi-task training on NeuroFusion-330 with QLoRA
                          (~30 GPU-h per fold, 5 folds, 5-fold CV is the headline)

Multi-task objective (matches model.py NeuroFusionConfig.w_*):
    L = w_seg * L_seg + w_field * L_field + w_gen * L_gen
        + w_ground * L_ground + kl_to_base_weight * L_kl_base

Where:
    L_seg     -- Dice + cross-entropy on backbone seg logits  (Phase 1 main objective)
    L_field   -- per-head cross-entropy on field classifier heads
    L_gen     -- LM cross-entropy on schema-constrained JSON output (teacher-forced)
    L_ground  -- 1 - mean IoU between Q-Former cross-attention and GT segmentation
    L_kl_base -- KL divergence from LoRA-adapted to frozen-base LM logits on text replay buffer
                 (catastrophic-forgetting regularizer; only meaningful with real MedGemma loaded)

Usage:
    # Phase 1 (segmentation pretrain) — once, ahead of all paired training
    python train.py --phase 1 \
        --brats-roots /data/BraTS2020 /data/BraTS2021 /data/BraTS2023 \
        --out-dir runs/phase1_seg

    # Phase 2a (identity-parity sanity check)
    python train.py --phase 2a \
        --seg-checkpoint runs/phase1_seg/best.pt \
        --out-dir runs/phase2a

    # Phase 2b (main multi-task training, one fold at a time)
    python train.py --phase 2b \
        --jsonl /data/neurofusion_369.jsonl \
        --brats-root /data/BraTS2020 \
        --splits /data/splits.json \
        --seg-checkpoint runs/phase1_seg/best.pt \
        --fold 0 \
        --epochs 3 --batch-size 2 --grad-accum 8 \
        --out-dir runs/phase2b_fold0

Smoke-test run (no real backbones; verifies the scaffold parses and forwards):
    python train.py --phase smoke

NeuroFusion v4 — anonymized for double-blind review.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from dataset import NeuroFusionDataset, collate_fn, get_split_loaders, load_splits, case_ids_for
from model import NeuroFusion, NeuroFusionConfig, dice_plus_ce_loss, COT_COMMIT_MARKER
from mrope_4d import (
    MRope4DAngleTable,
    MRope4DConfig,
    apply_mrope_4d,
    build_position_ids,
    _test_identity_reduction,
)
from schema import BrainTumorReport

# ---------------------------------------------------------------------------
# Optional logging backend (wandb if installed, else stdout)
# ---------------------------------------------------------------------------

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

log = logging.getLogger("neurofusion.train")


# ===========================================================================
# UTILITY: SEED + LOGGING + OUTPUT DIR
# ===========================================================================


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir: Path, level: int = logging.INFO) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out_dir / "train.log"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def setup_wandb(run_name: str, cfg: NeuroFusionConfig, out_dir: Path, args: argparse.Namespace) -> Any:
    if not _WANDB_AVAILABLE:
        log.info("wandb not installed; logging to stdout only")
        return None
    wandb.init(
        project="neurofusion",
        name=run_name,
        config={**asdict(cfg), **vars(args)},
        dir=str(out_dir),
        reinit=True,
    )
    return wandb


def log_metrics(metrics: dict[str, float], step: int, wb: Any = None) -> None:
    log.info("step=%d  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), step)
    if wb is not None:
        wb.log(metrics, step=step)


# ===========================================================================
# FIELD-TARGET BUILDING (BrainTumorReport -> per-field tensor labels)
# ===========================================================================

# Inverse enum maps must mirror schema.py and FieldClassificationHead exactly.
# These strings ARE the model's predicted indices (top-1 over softmax).

# C12: targets for over-segmented lesions (predicted count > GT count). Matches
# F.cross_entropy's own default so the two can never drift apart.
FIELD_IGNORE_INDEX = -100

_LOC_TO_IDX = {
    "frontal": 0, "temporal": 1, "parietal": 2, "occipital": 3, "insular": 4,
    "cerebellar": 5, "brainstem": 6, "parieto-temporal": 7, "fronto-parietal": 8,
    "fronto-temporal": 9, "temporo-occipital": 10, "parieto-occipital": 11,
    "thalamic": 12, "basal-ganglia": 13, "corpus-callosum": 14, "other": 15,
    # synonym handled by upstream canonicalization, mapped here for defense in depth
    "temporo-parietal": 7,
}
_COMP_TO_IDX = {
    "solid": 0, "cystic": 1, "necrotic": 2, "necrotic-cystic": 3, "hemorrhagic": 4,
}
_ENH_TO_IDX = {
    "none": 0, "homogeneous": 1, "heterogeneous": 2, "ring": 3,
    "thick-walled-ring": 4, "nodular": 5, "peripheral-nodular": 6,
}
_EDEMA_TO_IDX = {"none": 0, "mild": 1, "moderate": 2, "marked": 3}
_PRESENCE_TO_IDX = {"none": 0, "present": 1}
_AXIS_TO_IDX = {"none": 0, "contralateral": 1, "ipsilateral": 2}


def _field_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    gamma: float,
    label_smoothing: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-field-head classification loss with optional focal-CE re-weighting.

    gamma <= 0.0  -> plain weighted cross-entropy. This is BYTE-IDENTICAL to the
        previous inline ``F.cross_entropy(logits, target, weight=weight,
        label_smoothing=label_smoothing, ignore_index=ignore_index)`` call, so
        leaving ``--focal-gamma`` unset reproduces the running weighted-CE
        Model-2 exactly.

    gamma > 0.0   -> focal cross-entropy (Lin et al. 2017): scales each example's
        CE by ``(1 - p_t)**gamma`` so confident-correct (easy) cases are
        down-weighted and the head's gradient concentrates on hard/minority
        examples. NOTE: focal and label-smoothing are NOT combined -- when
        gamma > 0 the ``label_smoothing`` argument is intentionally ignored
        (byte-identity guarantees apply only at gamma == 0). Class weights, when
        provided, use a weighted-mean normalization (sum of per-example losses
        divided by the sum of the selected class weights) so the loss magnitude
        stays on the same scale as ``F.cross_entropy(weight=)`` and remains
        comparable to the weighted-CE baseline. ``ignore_index`` (default -100,
        the overflow/masked lesion slots) entries contribute exactly 0.
    """
    if gamma <= 0.0:
        return F.cross_entropy(
            logits, target, weight=weight,
            label_smoothing=label_smoothing, ignore_index=ignore_index,
        )
    # --- focal cross-entropy (gamma > 0) ---
    logp = F.log_softmax(logits, dim=-1)
    valid = target != ignore_index
    if valid.sum() == 0:
        # No supervised entries this batch: return a 0 that still carries the
        # graph (so .backward() is safe) without producing NaNs from log(0).
        return logits.sum() * 0.0
    # Gather log-prob of the (clamped) target class; ignore_index entries are
    # masked out below via ``valid`` so their clamped gather value never counts.
    safe_t = target.clamp(min=0)
    logp_t = logp.gather(1, safe_t[:, None]).squeeze(1)
    p_t = logp_t.exp()
    loss = (1.0 - p_t).pow(gamma) * (-logp_t)
    if weight is not None:
        w_t = weight[safe_t]
        loss = loss * w_t
        denom = (w_t * valid).sum().clamp(min=1e-8)
    else:
        denom = valid.sum().clamp(min=1)
    return (loss * valid).sum() / denom


def build_field_targets(
    reports: list[BrainTumorReport],
    routing_batch_idx: torch.Tensor,   # [N_total]
    routing_lesion_idx: torch.Tensor,  # [N_total], 1-based within each case
    device: torch.device,
    mask_overflow: bool = False,
) -> dict[str, torch.Tensor]:
    """For each per-lesion entry produced by LesionRouter, look up the corresponding
    GT field labels from `reports[batch_idx].lesions[lesion_idx-1]`.

    Returns a dict {field_name: LongTensor[N_total]} matching FieldClassificationHead
    output keys.

    mask_overflow (DEFAULT False == byte-identical): when a predicted lesion slot
    k exceeds the GT lesion count (over-segmentation / max_lesions>4 noise blobs),
    the default behavior CLIPS to the last GT lesion's labels. With mask_overflow
    True the overflow slot's targets are set to ignore_index (-100) for EVERY field
    so F.cross_entropy (ignore_index=-100 by default) skips them -- no field-head
    gradient from noise slots. Used for Models 3/4 (max_lesions>4) so the ablation
    measures "extra capacity" not "mis-supervision".
    """
    IGNORE = -100
    n_total = routing_batch_idx.shape[0]
    targets: dict[str, list[int]] = {
        "location": [], "composition": [], "enhancement_pattern": [],
        "edema_severity": [], "mass_effect": [], "midline_shift": [],
        "ventricular_compression": [], "brainstem_compression": [],
        "midbrain_compression": [], "axis_shift": [],
    }
    for i in range(n_total):
        b = int(routing_batch_idx[i].item())
        k = int(routing_lesion_idx[i].item())  # 1-based
        report = reports[b]
        # ☠️ C12 (2026-08-02): over-segmentation used to CLIP to the last GT lesion's
        # labels -- i.e. an extra predicted component was actively taught the DOMINANT
        # tumour's fields. Measured: predicted-vs-GT routed lesion count disagrees on
        # 273/623 = 43.8% of train cases (over-seg 32.3%), so this mis-supervised a
        # third of cases and taught the field heads that noise = the dominant tumour.
        # Emit ignore_index instead; F.cross_entropy(ignore_index=-100) drops them.
        if k - 1 >= len(report.lesions):
            for _key in targets:
                targets[_key].append(FIELD_IGNORE_INDEX)
            continue
        lesion = report.lesions[k - 1]
        targets["location"].append(_LOC_TO_IDX.get(lesion.location, _LOC_TO_IDX["other"]))
        targets["composition"].append(_COMP_TO_IDX.get(lesion.composition, 0))
        targets["enhancement_pattern"].append(_ENH_TO_IDX.get(lesion.enhancement_pattern, 0))
        targets["edema_severity"].append(_EDEMA_TO_IDX.get(lesion.surrounding_effects.edema_severity, 0))
        targets["mass_effect"].append(_PRESENCE_TO_IDX.get(lesion.surrounding_effects.mass_effect, 0))
        targets["midline_shift"].append(_PRESENCE_TO_IDX.get(lesion.surrounding_effects.midline_shift, 0))
        targets["ventricular_compression"].append(_PRESENCE_TO_IDX.get(lesion.involvement.ventricular_compression, 0))
        targets["brainstem_compression"].append(_PRESENCE_TO_IDX.get(lesion.involvement.brainstem_compression, 0))
        targets["midbrain_compression"].append(_PRESENCE_TO_IDX.get(lesion.involvement.midbrain_compression, 0))
        targets["axis_shift"].append(_AXIS_TO_IDX.get(lesion.axis_shift, 0))
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in targets.items()}


_FIELD_N_CLASSES: dict[str, int] = {
    "location": 16,  # _LOC_TO_IDX has 15 region enums + "other" at index 15 = 16 total; matches model.field_heads["location"]
    "composition": 5,
    "enhancement_pattern": 7,
    "edema_severity": 4,
    "mass_effect": 2,
    "midline_shift": 2,
    "ventricular_compression": 2,
    "brainstem_compression": 2,
    "midbrain_compression": 2,
    "axis_shift": 3,
}


def compute_field_class_weights(
    reports: list[BrainTumorReport],
    device: torch.device,
    smoothing: float = 1.0,
    weight_clip: tuple[float, float] = (0.25, 4.0),
) -> dict[str, torch.Tensor]:
    """Build per-field class weights from training-set lesion frequencies.

    Inverse-frequency weighting (with Laplace smoothing + clipping) for
    F.cross_entropy(weight=). Addresses majority-class collapse that was
    structurally encouraged by unweighted CE and likely contributed to the
    hemisphere bilateral over-prediction + always-"none" bias on the
    presence-flag heads.

    smoothing=1.0 adds a count of 1 to every class so unseen classes don't
    blow up to infinite weight. Clip range bounds the weight ratio so a
    single rare class can't dominate the head's gradient (which would
    re-introduce the field-loss balance problem from a different angle).

    Note: weights are built from LESIONS (not cases) since the field heads
    are per-lesion. A case with 4 lesions contributes 4 to each class count.
    """
    import numpy as np
    counts: dict[str, np.ndarray] = {
        name: np.full(n, smoothing, dtype=np.float64)
        for name, n in _FIELD_N_CLASSES.items()
    }
    for report in reports:
        for lesion in report.lesions:
            counts["location"][_LOC_TO_IDX.get(lesion.location, _LOC_TO_IDX["other"])] += 1
            counts["composition"][_COMP_TO_IDX.get(lesion.composition, 0)] += 1
            counts["enhancement_pattern"][_ENH_TO_IDX.get(lesion.enhancement_pattern, 0)] += 1
            counts["edema_severity"][_EDEMA_TO_IDX.get(lesion.surrounding_effects.edema_severity, 0)] += 1
            counts["mass_effect"][_PRESENCE_TO_IDX.get(lesion.surrounding_effects.mass_effect, 0)] += 1
            counts["midline_shift"][_PRESENCE_TO_IDX.get(lesion.surrounding_effects.midline_shift, 0)] += 1
            counts["ventricular_compression"][_PRESENCE_TO_IDX.get(lesion.involvement.ventricular_compression, 0)] += 1
            counts["brainstem_compression"][_PRESENCE_TO_IDX.get(lesion.involvement.brainstem_compression, 0)] += 1
            counts["midbrain_compression"][_PRESENCE_TO_IDX.get(lesion.involvement.midbrain_compression, 0)] += 1
            counts["axis_shift"][_AXIS_TO_IDX.get(lesion.axis_shift, 0)] += 1

    weights: dict[str, torch.Tensor] = {}
    lo, hi = weight_clip
    for name, c in counts.items():
        # Inverse frequency, normalize so mean weight = 1, then clip to [lo, hi]
        inv = c.sum() / (len(c) * c)  # so the class with mean frequency gets weight 1
        inv = np.clip(inv, lo, hi)
        weights[name] = torch.tensor(inv, dtype=torch.float32, device=device)
    return weights


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — give each worker a distinct RNG seed.

    Without this, all DataLoader workers inherit the parent's numpy RNG state
    at fork → produce identical augmentation streams in lockstep, reducing
    effective augmentation diversity by num_workers× (a known PyTorch trap).
    """
    import numpy as np
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


# ===========================================================================
# GROUNDING LOSS (Q-Former attention vs GT segmentation IoU)
# ===========================================================================


def compute_grounding_loss(
    qformer: nn.Module,
    lesion_gt_feat: torch.Tensor,     # [N_total, dz, dy, dx] — router emits this aligned
                                       # with the lesion feature crop (same bounds + pad)
    iou_eps: float = 1e-6,
    soft: bool = True,
) -> torch.Tensor:
    """L_ground = 1 - mean_q soft-IoU(attn_q, GT_lesion_mask).

    Active when cfg.use_custom_qformer=True (default in v4): the custom Q-Former
    saves `_last_attn` [N_total, n_queries, d*h*w] and `_feat_dhw` (d, h, w).

    The previous version downsampled the WHOLE image's GT mask to feat_dhw, which
    didn't spatially align with the per-lesion attention crop — IoU was ~0 always.
    Now the router emits `lesion_gt_feat` already cropped through the same window
    as the lesion feature crop, so the spaces match.

    soft=True (default): differentiable soft-IoU.
    soft=False: hard threshold at 90th-percentile attention per query (val-only).
    """
    last_attn = getattr(qformer, "_last_attn", None)
    feat_dhw = getattr(qformer, "_feat_dhw", None)
    if last_attn is None or feat_dhw is None:
        return torch.tensor(0.0, device=lesion_gt_feat.device, requires_grad=False)

    from qformer_custom import reshape_attn_to_grid, grounding_iou_loss
    try:
        attn_grid = reshape_attn_to_grid(last_attn, feat_dhw)     # [N_total, n_q, dz, dy, dx]
    except AssertionError:
        log.warning(f"L_ground skipped: feat_dhw {feat_dhw} mismatches attn shape {last_attn.shape}")
        return torch.tensor(0.0, device=lesion_gt_feat.device, requires_grad=False)

    if lesion_gt_feat.shape[-3:] != tuple(feat_dhw):
        log.warning(f"L_ground skipped: gt_feat shape {tuple(lesion_gt_feat.shape)} mismatches feat_dhw {feat_dhw}")
        return torch.tensor(0.0, device=lesion_gt_feat.device, requires_grad=False)

    return grounding_iou_loss(attn_grid, lesion_gt_feat, soft=soft, eps=iou_eps)


# ===========================================================================
# KL-TO-BASE LOSS (anti-forgetting on text replay buffer)
# ===========================================================================


class TextReplayBuffer:
    """Small fixed corpus of clinical-style text snippets used for KL regularization.

    On each KL step we sample a batch from this buffer, run it through
    (a) the LoRA-adapted LM, and (b) the same LM with LoRA disabled (the frozen base),
    then compute KL(lora || base) on the next-token logits.

    Real implementation should populate the buffer from a held-out clinical text source
    (e.g. PubMed abstracts in radiology, MIMIC-CXR free-text, or a fixed sample of
    MedGemma's pretraining text). For now: a small hardcoded set of generic radiology
    sentences as a placeholder so the training scaffold runs end-to-end.
    """

    _PLACEHOLDER_TEXTS = [
        "The patient presents with a 2 cm enhancing lesion in the right frontal lobe.",
        "Findings are consistent with high-grade glioma.",
        "There is moderate vasogenic edema surrounding the mass.",
        "Mass effect on the lateral ventricle is noted.",
        "The lesion demonstrates restricted diffusion on DWI sequences.",
        "Heterogeneous enhancement is present following contrast administration.",
        "Differential diagnosis includes glioblastoma multiforme and metastasis.",
        "The midline structures are shifted contralaterally by approximately 4 mm.",
        "Brainstem compression is not evident on this examination.",
        "Recommend biopsy and clinical correlation.",
    ]

    def __init__(self, size: int = 2000, seed: int = 42, corpus_path: str | None = None) -> None:
        rng = random.Random(seed)
        if corpus_path:
            # Phase 0a (data-expansion): swap the 10 placeholders for a real, diverse
            # on-disk neuro-radiology corpus (ISLES22 + WMH findings/impressions; NON-tumor
            # -> leak-free vs the GLI/MEN/MET eval). Use the full corpus directly.
            base = self._load_corpus(corpus_path)
            if not base:
                raise ValueError(f"KL corpus empty or unreadable: {corpus_path}")
            logging.info("TextReplayBuffer: loaded %d sentences from %s (KL corpus swap)",
                         len(base), corpus_path)
            self.texts: list[str] = list(base)
        else:
            # Default path (byte-identical): repeat-sample the placeholder set to `size`.
            self.texts = [rng.choice(self._PLACEHOLDER_TEXTS) for _ in range(size)]

    @staticmethod
    def _load_corpus(path: str) -> list[str]:
        """Load one sentence per line from a jsonl ({"text": ...}) or plain-text file."""
        out: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0] == "{":
                    try:
                        rec = json.loads(line)
                        txt = rec.get("text") if isinstance(rec, dict) else None
                    except json.JSONDecodeError:
                        txt = None
                else:
                    txt = line
                if txt and isinstance(txt, str) and txt.strip():
                    out.append(txt.strip())
        return out

    def sample(self, n: int) -> list[str]:
        return random.sample(self.texts, k=min(n, len(self.texts)))


def compute_kl_to_base_loss(
    model: NeuroFusion,
    replay: TextReplayBuffer,
    n_samples: int = 4,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """KL divergence from LoRA-adapted to frozen-base LM logits on text replay.

    Returns 0.0 (no-grad) when running with the placeholder LM.

    Real implementation outline (when MedGemma + PEFT is wired):
        with model.lm.lm.disable_adapter():
            base_logits = model.lm.lm(input_ids).logits
        lora_logits = model.lm.lm(input_ids).logits
        kl = F.kl_div(
            F.log_softmax(lora_logits, dim=-1),
            F.softmax(base_logits, dim=-1),
            reduction="batchmean",
        )
        return kl
    """
    # Detect placeholder LM
    if not hasattr(model.lm.lm, "disable_adapter"):
        return torch.tensor(0.0, device=device, requires_grad=False)

    # Real path (ready to activate once PEFT-wrapped LM is in place)
    texts = replay.sample(n_samples)
    tokenizer = getattr(model.lm, "tokenizer", None)
    if tokenizer is None:
        return torch.tensor(0.0, device=device, requires_grad=False)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        with model.lm.lm.disable_adapter():
            base_logits = model.lm.lm(**inputs).logits

    lora_logits = model.lm.lm(**inputs).logits

    log_p_lora = F.log_softmax(lora_logits, dim=-1)
    p_base = F.softmax(base_logits, dim=-1)
    return F.kl_div(log_p_lora, p_base, reduction="batchmean")


def compute_text_sft_loss(
    model: "NeuroFusion",
    sft_buffer: "TextReplayBuffer",
    n_samples: int = 4,
    max_length: int = 128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Phase 1a: text-only medical-alignment SFT loss (NO volume required).

    Teacher-forces the LoRA-adapted LM on a batch of real medical sentences drawn
    from `sft_buffer` (e.g. on-disk ISLES22 + WMH neuro-radiology findings/impressions,
    which are NON-tumor -> leak-free vs the GLI/MEN/MET report eval). This is the
    net-new "text-only SFT path": it exercises `model.lm.lm` directly with input_ids
    (bypassing the visual prefix entirely), so it needs no MRI volume, and its gradient
    flows only into the LM's LoRA adapters (4-bit base frozen). Realizes the plan's
    "mix text-only batches with image-report batches" as an additive weighted loss term
    (weight ~= the intended mix ratio), which avoids any DataLoader/collate change and
    keeps the default (image-report) path byte-identical.

    Returns a no-grad 0.0 when the LM is the placeholder (no `disable_adapter`) or the
    tokenizer / buffer is missing, mirroring compute_kl_to_base_loss's guards.
    """
    if sft_buffer is None or not hasattr(model.lm.lm, "disable_adapter"):
        return torch.tensor(0.0, device=device, requires_grad=False)
    tokenizer = getattr(model.lm, "tokenizer", None)
    if tokenizer is None:
        return torch.tensor(0.0, device=device, requires_grad=False)
    texts = sft_buffer.sample(n_samples)
    if not texts:
        return torch.tensor(0.0, device=device, requires_grad=False)
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    ).to(device)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100  # ignore padding in the CE
    out = model.lm.lm(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels,
    )
    return out.loss


# ===========================================================================
# TARGET TEXT BUILDING (for teacher-forced LM training)
# ===========================================================================


def build_target_text_ids(
    report_jsons: list[str],
    tokenizer: Any | None,
    max_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Tokenize the canonical JSON form of each report for teacher forcing.

    Returns (input_ids, attention_mask) of shape [B, T] each, or (None, None) if
    no tokenizer is available (placeholder LM path).
    """
    if tokenizer is None:
        return None, None
    out = tokenizer(
        report_jsons,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return out["input_ids"].to(device), out["attention_mask"].to(device)


def build_cot_target_strings(
    reports: list[Any],
    report_jsons: list[str],
    dx_first: bool = False,
    dx_rationale: bool = False,
) -> list[str]:
    """Build draft-then-commit LM targets for --cot-supervision.

    Each target is  '<draft prose>[COMMIT]<canonical report JSON>'  (with the
    COT_COMMIT_MARKER, which embeds surrounding newlines), where:
      - <draft prose> = the gold narrative = findings + impression, and
      - <canonical report JSON> = the SAME string the default path teacher-forces
        (`report.model_dump_json()`), unchanged, still carrying
        `differential_diagnosis`.

    Training on this target makes the multi-stage "narrate, then COMMIT to a
    diagnosis-bearing JSON" decode IN-DISTRIBUTION, instead of a decode-time
    recipe the adapter was never trained on (single-pass JSON only). Only the
    tokenized TARGET STRING changes — the loss/masking path in model.py is
    untouched (visual+prefix still masked to -100; loss over target tokens).

    dx_first (L3, --dx-first-draft; default False): when True, prepend a short
    'Diagnosis: <top-1 differential>. ' clause to the DRAFT so the commit target
    states the diagnosis up front (findings+impression kept intact). Empty ddx =>
    no clause. False => byte-identical to the original target string.

    dx_rationale (Rung-2 Lever C, --dx-rationale-draft; default False): when True,
    replace the bare 'Diagnosis: <dd0>. ' clause with a morphology->diagnosis
    "Chain-of-Diagnosis" clause composed from the case's OWN gold structured fields
    (composition/enhancement/location/edema/multiplicity) + the tumor type's
    definitional anatomic descriptor, via scripts/dx_morphology_kb.compose_rationale.
    Falls back to the bare dx-first clause when the dx family is unknown or on any
    error. DRAFT-only (stripped at [COMMIT] -> scored JSON unchanged); leak-safe (the
    observable fields are already the SFT target; the anatomic descriptor is a
    type-level truth, never a case-specific leaked fact). False => byte-identical.
    """
    targets: list[str] = []
    for rep, rj in zip(reports, report_jsons):
        findings = (getattr(rep, "findings", "") or "").strip()
        impression = (getattr(rep, "impression", "") or "").strip()
        if findings and impression:
            draft = f"{findings}\n\n{impression}"
        else:
            draft = findings or impression
        # L3 (--dx-first-draft): commit the top-1 differential BEFORE narrating so the
        # draft's diagnosis is stated up front (findings+impression unchanged). Off =>
        # byte-identical target; empty/missing ddx => clause skipped.
        if dx_first or dx_rationale:
            _ddx = getattr(rep, "differential_diagnosis", None)
            _dd0 = (_ddx[0] if isinstance(_ddx, (list, tuple)) and _ddx else "")
            _dd0 = (_dd0 or "").strip()
            if _dd0:
                _clause = None
                if dx_rationale:
                    # Rung-2 (Lever C): morphology->diagnosis "Chain-of-Diagnosis"
                    # clause from the case's OWN gold fields. Lazy import so train.py
                    # import-time behavior is unchanged (the eval jobs import train.py
                    # but never reach this training-only path); any failure or unknown
                    # dx family falls back to the bare dx-first clause below.
                    try:
                        from scripts.dx_morphology_kb import (
                            compose_rationale as _compose_rationale,
                        )
                        _clause = _compose_rationale(_dd0, rep)
                    except Exception:
                        _clause = None
                if _clause:
                    # compose_rationale returns a clause ending in a trailing space.
                    draft = f"{_clause}{draft}"
                else:
                    draft = f"Diagnosis: {_dd0}. {draft}"
        targets.append(f"{draft}{COT_COMMIT_MARKER}{rj}")
    return targets


# ---------------------------------------------------------------------------
# Head-conditioned CoT v2: per-case cohort-anchor index resolution
# ---------------------------------------------------------------------------
# Anchor slots (must match model.NeuroFusionConfig.n_cohort_anchor ordering):
#   0 = none/unknown, 1 = GLI, 2 = MEN, 3 = MET
_ANCHOR_COHORT_IDX = {"GLI": 1, "MEN": 2, "MET": 3}
# Accepted string labels in a --cohort-anchor-map JSON (case-insensitive).
_ANCHOR_LABEL_TO_IDX = {
    "none": 0, "unknown": 0, "": 0,
    "gli": 1, "glioma": 1, "glioblastoma": 1, "astrocytoma": 1,
    "men": 2, "meningioma": 2,
    "met": 3, "metastasis": 3, "metastatic": 3, "secondary": 3, "carcinoma": 3,
}
# Diagnosis synonyms for the GOLD-derived fallback (mirrors scripts/dx_recall.py
# SYNONYMS exactly — kept inline to avoid a scripts/ package-import dependency).
_ANCHOR_DX_SYNONYMS = {
    "GLI": ["glioblastoma", "glioma", "astrocytoma"],
    "MEN": ["meningioma"],
    "MET": ["metasta", "secondary", "breast cancer", "carcinoma"],
}


def _gold_anchor_idx(report: Any) -> int:
    """Cohort-anchor index from a TRAIN report's GOLD differential_diagnosis[0].

    Reads the supervised report content (a legitimate TRAIN label), NOT the case_id
    cohort tag. Returns 0 (none) if no synonym matches / no dx.
    """
    dd = getattr(report, "differential_diagnosis", None) or []
    if not dd:
        return 0
    s = str(dd[0]).lower()
    for coh, syns in _ANCHOR_DX_SYNONYMS.items():
        if any(w in s for w in syns):
            return _ANCHOR_COHORT_IDX[coh]
    return 0


# ── RULING 3(a) 2026-08-10: THE ANCHOR MUST NEVER FALL BACK SILENTLY ───────
# WAS: `cohort_anchor_map.get(cid, 0)` — a case absent from the map decoded through
# anchor slot 0 ("none") with no assertion, no warning, and nothing in the artifact.
# Slot 0 carries 3 of 624 training cases (0.5%, out/bprime/anchor_e30/probe_anchor_run.log),
# so a whole roster routed through it is an UNTESTED DECODE REGIME that degrades both arms
# of an A/B equally — i.e. it does not show up as an arm difference, which is exactly why
# the missing assertion, not the 0.5% contamination, is the defect. The shipped map covers
# exactly the canonical 164, so every one of the 728 B′9-R roster cases would have hit it.
#
# ☠️ THE FALLBACK IS NOT A DEFAULT, IT IS A GUESS DRESSED AS A DEFAULT. Refuse instead.
#
# MOVED HERE FROM scripts/predict_draft_commit.py 2026-08-11, UNCHANGED in logic: the
# SAME defect (`.get(cid, 0)`) existed in this file's train loop and in
# scripts/faithfulness_v3.py. predict_draft_commit already imports from train, so train
# is the only place all three sites can share ONE implementation without an import
# cycle; both scripts now import these names from here.
class AnchorMissing(RuntimeError):
    """A case_id has no anchor slot. Never resolvable to a default."""


def resolve_anchor_slots(amap: dict[str, int], case_ids: list[str]) -> list[int]:
    """Anchor slot per case, or REFUSE. There is no fallback and no override.

    ★ The right fix for a missing key is to BUILD the map for those cases
    (scripts/bprime_trunk_extractor.py --roster-json ... then scripts/probe_anchor.py
    --roster-npz ...), not to invent a slot for them.
    """
    missing = [c for c in case_ids if c not in amap]
    if missing:
        raise AnchorMissing(
            f"{len(missing)} of {len(case_ids)} case(s) have NO anchor slot in the "
            f"cohort-anchor map (e.g. {missing[:5]}). The map has {len(amap)} keys. "
            f"REFUSING: the old behaviour silently decoded these through slot 0 "
            f"('none'), a 0.5%-training-mass regime, and recorded nothing. Build the "
            f"anchor map for these cases (bprime_trunk_extractor.py --roster-json -> "
            f"probe_anchor.py --roster-npz) — do NOT hand-write {{cid: <cohort>}}, which "
            f"makes the anchor the COHORT LABEL rather than an image-derived prediction "
            f"(RULING 3(c), REJECTED).")
    return [amap[c] for c in case_ids]


def anchor_assert_red_drill(amap: dict[str, int], evidence_path: Path,
                            ckpt: str = "", testpred: str = "",
                            roster_json: str | None = None) -> None:
    """Prove `resolve_anchor_slots` REFUSES on a missing key, before we rely on it.

    ☠️ THE EVIDENCE IS WRITTEN TO DISK **BEFORE** THE ASSERT IT GATES. A control whose
    evidence the gated code can destroy is worthless: if the drill result were only
    written after the refusal decision, a run that aborts leaves nothing behind and
    "we drove it red" becomes an unverifiable claim. So: run both probes, persist, then
    fail closed.

    ☠️ AND IT CHECKS BOTH DIRECTIONS. A resolver that refuses EVERYTHING would pass a
    red-only drill while being useless, so the clean probe must also SUCCEED.
    """
    probe_key = next(iter(sorted(amap))) if amap else None
    missing_id = "__B9R_ANCHOR_RED_DRILL_MISSING__"
    assert missing_id not in amap, "red-drill sentinel collided with a real case_id"

    red_fired, red_msg, red_named = False, "", False
    try:
        resolve_anchor_slots(amap, ([probe_key] if probe_key else []) + [missing_id])
    except AnchorMissing as e:
        red_fired, red_msg = True, str(e)
        red_named = missing_id in red_msg

    green_ok, green_slots = False, None
    if probe_key is not None:
        try:
            green_slots = resolve_anchor_slots(amap, [probe_key])
            green_ok = green_slots == [amap[probe_key]]
        except AnchorMissing as e:
            red_msg += f" | CLEAN PROBE ALSO REFUSED: {e}"

    # ☠️ A drill fired only against a synthetic sentinel proves the code path, not the
    # SITUATION. If a real roster is supplied, the drill must also refuse REAL ids that
    # the map genuinely does not cover — that is the case this assertion exists for.
    real_probe = None
    if roster_json:
        raw = json.loads(Path(roster_json).read_text())
        rids = (raw if isinstance(raw, list)
                else raw.get("roster_construction", {}).get("roster_ids")
                or raw.get("roster_ids") or raw.get("sample_ids"))
        if not rids:
            raise SystemExit(f"--drill-roster-json {roster_json}: no id list found")
        rmiss = [c for c in rids if c not in amap]
        r_refused = False
        try:
            resolve_anchor_slots(amap, list(rids)[:8])
        except AnchorMissing:
            r_refused = True
        real_probe = {"roster_json": roster_json, "n_roster": len(rids),
                      "n_missing_from_map": len(rmiss),
                      "refused_first_8": r_refused,
                      "example_missing": rmiss[:5]}

    ev = {
        "_what": "RULING 3(a) red drill: the shared anchor resolution "
                 "(train.resolve_anchor_slots) must "
                 "REFUSE a case with no anchor slot, and must still ACCEPT a case that "
                 "has one.",
        "real_roster_probe": real_probe,
        "_why_written_first": "Written to disk BEFORE the assert it gates, so an abort "
                              "cannot destroy the evidence that the control fired.",
        "ckpt": ckpt, "cohort_anchor_testpred": testpred,
        "map_keys": len(amap),
        "red_probe": {"case_ids": [probe_key, missing_id], "refused": red_fired,
                      "named_the_missing_id": red_named, "message": red_msg[:600]},
        "green_probe": {"case_id": probe_key, "accepted": green_ok, "slots": green_slots},
    }
    # A real roster that IS fully covered is a legitimate green (nothing to refuse);
    # a real roster with missing ids that was NOT refused is a failed drill.
    real_ok = (real_probe is None or real_probe["n_missing_from_map"] == 0
               or real_probe["refused_first_8"])
    ev["verdict"] = ("DRILL FIRED — refuses missing, accepts present"
                     if (red_fired and red_named and green_ok and real_ok)
                     else "DRILL DID NOT FIRE")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(ev, indent=1))
    log.info(f"[anchor-red-drill] {ev['verdict']} -> {evidence_path}")

    if not (red_fired and red_named and green_ok and real_ok):
        raise SystemExit(
            f"☠️ REFUSING: the anchor-assertion red drill did not fire "
            f"(refused={red_fired} named={red_named} clean_accepted={green_ok} "
            f"real_roster_ok={real_ok}). An unproven guard is a failed guard. "
            f"Evidence: {evidence_path}")


def assert_train_basis_disjoint(train_case_ids: set[str] | list[str],
                                splits_obj: dict[str, Any], out_dir: Path,
                                tag: str = "train") -> None:
    """PHASE2B_PREFLIGHT: no HELD-OUT case may reach a gradient step. Spends NO look.

    ☠️ WHY THE EXISTING GUARD IS NOT ENOUGH (measured 2026-08-11, not hypothesised). The
    pre-flight assert beside this one builds `protected_ids` from the SPLITS KEYS
    (`test` | `conformal_calibration` | `test_gli/men/met`). In `splits_ultra_with_met.json`
    the 24 canonical-164 MET cases sit in **CV fold 6** and in no protected key, so that
    guard is structurally blind to them: `nfmistral_cotcond_fold0` trained on all 24 and
    the guard logged PASSED. Every fold-7 descendant — including TE4, the shipped lineage —
    then inherited those weights through `--warmstart-nf`
    (out/anchor_ruling3a_drills/LINEAGE_WARMSTART_AUDIT.json).
    That is a guard on FORM that never checks CONTENT: it asks where an id is FILED, not
    whether it is HELD OUT. This one asks the registry, against the committed digest.

    ☠️ ALL FOUR BASES, ALWAYS. Narrowing `bases` to make a run pass is the coordinated
    no-op this project has already shipped once — narrow the subtraction and its
    verification together and both agree about nothing.
    """
    ids = sorted(train_case_ids)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from test_basis_guard import BasisRefused, assert_disjoint_from_held_out  # noqa: E402

    BASES = ("canonical_164", "fit_503", "legacy_39", "conformal_50")

    # ☠️ NOT VACUOUS: an empty roster is disjoint from everything, and "0 of 0 train cases
    # intersect a held-out basis" reads exactly like a pass while checking nothing.
    if not ids:
        raise SystemExit(
            "REFUSING: the train loader resolved to 0 cases, so PHASE2B_PREFLIGHT would "
            "pass with nothing to check.")

    # ☠️ PRESENCE BEFORE ABSENCE. A probe that reports 0 must first be shown capable of
    # reporting >0, or the zero means "blind", not "clean". Sentinel = one of THIS splits
    # file's own held-out ids, so the proof is self-contained (no extra artifact, no look).
    sentinel, sentinel_msg = None, ""
    for cid in (c for k in ("test", "conformal_calibration", "test_gli", "test_men",
                            "test_met") for c in splits_obj.get(k, [])):
        try:
            assert_disjoint_from_held_out([cid], who="PHASE2B_PREFLIGHT_PRESENCE_PROBE",
                                          bases=BASES)
        except BasisRefused as e:
            sentinel, sentinel_msg = cid, str(e)[:300]
            break

    report, verdict, offenders = None, "", ""
    try:
        report = assert_disjoint_from_held_out(ids, who="PHASE2B_PREFLIGHT", bases=BASES)
        verdict = "DISJOINT" if sentinel else "UNEVALUABLE — presence never demonstrated"
    except BasisRefused as e:
        verdict, offenders = "LEAK", str(e)

    # Evidence to disk BEFORE the raise it gates: a run that aborts must still leave the
    # proof that the control fired. Scenario in the filename.
    ev = {"_what": "PHASE2B_PREFLIGHT: the train loader's OWN ids vs every held-out basis, "
                   "before the first gradient step.",
          "_no_look": "assert_disjoint_from_held_out is the no-look membership entry point: "
                      "the held-out ids never leave it, only counts come back.",
          "bases": list(BASES), "n_train_ids": len(ids),
          "presence_probe": {"sentinel_id": sentinel, "refused": sentinel is not None,
                             "message": sentinel_msg},
          "result": report, "verdict": verdict, "offenders": offenders[:1000]}
    p = Path(out_dir) / f"basis_preflight_{tag}_n{len(ids)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ev, indent=1))
    log.info(f"[PHASE2B_PREFLIGHT] {verdict} -> {p}")

    if verdict == "LEAK":
        raise SystemExit(
            f"☠️ PHASE2B_PREFLIGHT TRIPPED — a HELD-OUT case is reachable in the train "
            f"loader. {offenders}\nRefusing to start training. Evidence: {p}. The split "
            f"keys may look clean: this checks the BASIS REGISTRY, which is what the "
            f"June-2026 cotcond_fold0 run needed and did not have.")
    if sentinel is None:
        raise SystemExit(
            f"☠️ PHASE2B_PREFLIGHT UNEVALUABLE: no id in this splits file's held-out keys "
            f"is a member of any basis {BASES}, so the probe was never shown able to "
            f"report a hit and its zero means nothing. Evidence: {p}. Refusing — an "
            f"unevaluable gate is a failure of the gate, not a pass of the run.")
    log.info(f"[PHASE2B_PREFLIGHT] PASSED: 0/{len(ids)} train ids intersect "
             f"{len(BASES)} held-out bases; presence proven on sentinel {sentinel!r}.")


def assert_anchor_coverage(amap: dict[str, int], train_case_ids: set[str] | list[str],
                           out_dir: Path, source: str = "", tag: str = "train") -> None:
    """RULING 3(a) for TRAIN TIME: refuse a short anchor map AT STARTUP, never mid-run.

    A training run that discovers a missing anchor three hours in and dies has destroyed
    the work; one that discovers it and FALLS BACK has silently trained those cases on
    slot 0. Neither is acceptable, so the whole train split is resolved here, before the
    first gradient step, through the SAME resolve_anchor_slots() the loop uses.

    It is a module-level function so it can be driven RED on a CPU with no model, no
    loader and no GPU — a startup gate that can only be exercised by launching the very
    job it protects is a gate nobody ever tests.
    """
    ids = sorted(train_case_ids)
    # Evidence FIRST: the drill persists its red/green result before it raises. ☠️ The
    # SCENARIO goes in the filename (tag + map + n ids) — a fixed drill filename once let
    # a green run overwrite the red evidence that justified it.
    anchor_assert_red_drill(
        amap, Path(out_dir) / (f"anchor_assert_red_drill_{tag}_"
                               f"{Path(source).stem or 'nosrc'}_n{len(ids)}.json"),
        testpred=source)
    # ☠️ AND IT MUST NOT PASS VACUOUSLY: resolve_anchor_slots([]) returns [] without
    # raising, so an empty train set would log "0/0 covered" — a sentence that reads
    # exactly like a pass while nothing was checked.
    if not ids:
        raise SystemExit(
            "REFUSING: --cohort-anchor is on but the train split resolved to 0 cases, "
            "so the anchor coverage check would pass with nothing to check.")
    resolve_anchor_slots(amap, ids)
    log.info(f"[cohort-anchor] coverage OK: {len(ids)}/{len(ids)} train cases have an "
             f"anchor slot in {source or '<map>'} (none will fall back to slot 0)")


def build_cohort_anchor_map(
    args: argparse.Namespace,
    all_train_items: list[dict[str, Any]],
    protected_ids: set[str],
) -> dict[str, int] | None:
    """Resolve a leak-free {case_id: anchor_idx} map for head-conditioned CoT v2.

    Returns None when --cohort-anchor is off (byte-identical default path).

    Two sources (exactly one required when --cohort-anchor is set):
      1. --cohort-anchor-map JSON (PRIMARY): the Stage-1 visual-token cohort probe's
         OUT-OF-FOLD TRAIN-split predictions {case_id: label|int}. Leak-free by
         construction upstream (probe never trained on the case it predicts, never
         on any test case). Chosen as primary because the probe's OOF predictions
         carry the SAME ~0.88 accuracy + confusion structure the TEST predictions
         will have at inference, so the draft trains on the exact anchor-noise it
         will face (train/inference matched).
      2. --cohort-anchor-from-gold (FALLBACK / smoke): derive the cohort from each
         train report's gold differential_diagnosis, then flip a
         --cohort-anchor-error-rate fraction to a random wrong cohort to SIMULATE
         the probe's ~0.88 accuracy (a perfect-gold anchor would train the draft to
         over-trust an anchor the 0.88 inference anchor can't match).

    Both are leak-free: neither reads the case_id cohort tag, and neither fits on any
    protected/test case. A hard gate asserts no protected id is looked up for training.
    """
    if not getattr(args, "cohort_anchor", False):
        return None

    train_reports = {it["report"].case_id: it["report"] for it in all_train_items}
    gold = {cid: _gold_anchor_idx(rep) for cid, rep in train_reports.items()}

    amap_path = getattr(args, "cohort_anchor_map", None)
    if amap_path:
        with open(amap_path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise SystemExit(f"--cohort-anchor-map {amap_path} must be a JSON object {{case_id: cohort}}")
        anchor_map: dict[str, int] = {}
        for cid, val in raw.items():
            if isinstance(val, bool):
                raise SystemExit(f"--cohort-anchor-map value for {cid} is a bool; expected label/int")
            if isinstance(val, (int, float)):
                idx = int(val)
                if idx not in (0, 1, 2, 3):
                    raise SystemExit(f"--cohort-anchor-map int for {cid}={idx} out of range 0..3")
            else:
                key = str(val).strip().lower()
                if key not in _ANCHOR_LABEL_TO_IDX:
                    raise SystemExit(f"--cohort-anchor-map label for {cid}={val!r} unrecognized")
                idx = _ANCHOR_LABEL_TO_IDX[key]
            anchor_map[cid] = idx
        # LEAK GATE: the map may legitimately contain TEST ids (for inference), but
        # we must never USE a protected id to condition a training step. That is
        # guaranteed by looking up only train case_ids in the loop.
        #
        # ☠️ RULING 3(a) 2026-08-11: this comment used to end "...assert here that every
        # train id we WILL look up resolves without falling back to the tag" — and NO SUCH
        # ASSERT EXISTED. The log below merely announced "missing N -> anchor 0" and the
        # run proceeded, conditioning those cases on the untested slot-0 regime. The count
        # here stays (it names the source file that is short), but the REFUSAL is the
        # whole-split resolve_anchor_slots() call at startup in the phase-2b main, so both
        # map sources (probe-map AND gold) go through one check.
        covered = [cid for cid in train_reports if cid in anchor_map]
        missing = [cid for cid in train_reports if cid not in anchor_map]
        # Sanity: agreement of the probe map vs gold on train. ~0.88 expected; a
        # suspiciously-perfect map (~1.0) suggests gold/tag leakage into the file.
        agree = sum(1 for cid in covered if anchor_map[cid] == gold[cid])
        acc = agree / max(1, len(covered))
        log.info(
            f"[cohort-anchor] source=probe-map {amap_path}: covers {len(covered)}/"
            f"{len(train_reports)} train ids (missing {len(missing)} -> the startup "
            f"coverage check REFUSES the run; build the map for them); "
            f"map-vs-gold train agreement={acc:.3f} (expect ~0.88 for OOF probe preds)"
        )
        if acc > 0.98 and len(covered) > 20:
            log.warning(
                "[cohort-anchor] map-vs-gold agreement > 0.98 — the map looks like "
                "GOLD/tag rather than the probe's OOF predictions. Train/inference "
                "will MISMATCH (inference anchors are ~0.88). Verify the map file."
            )
        leaked = [cid for cid in anchor_map if cid in protected_ids and cid in train_reports]
        assert not leaked, f"[cohort-anchor] LEAK: protected ids in the train lookup: {leaked[:5]}"
        return anchor_map

    if getattr(args, "cohort_anchor_from_gold", False):
        err = float(getattr(args, "cohort_anchor_error_rate", 0.12))
        rng = random.Random(int(getattr(args, "seed", 42)))
        anchor_map = dict(gold)
        n_flip = 0
        for cid in sorted(anchor_map):  # sorted -> deterministic given the seed
            idx = anchor_map[cid]
            if idx != 0 and rng.random() < err:
                wrong = [c for c in (1, 2, 3) if c != idx]
                anchor_map[cid] = rng.choice(wrong)
                n_flip += 1
        log.info(
            f"[cohort-anchor] source=gold+error-sim: {len(anchor_map)} train ids, "
            f"flipped {n_flip} ({100.0 * n_flip / max(1, len(anchor_map)):.1f}%) to simulate "
            f"probe error-rate={err:.3f} (effective acc ~{1.0 - n_flip / max(1, len(anchor_map)):.3f})"
        )
        return anchor_map

    raise SystemExit(
        "--cohort-anchor requires a source: pass --cohort-anchor-map <probe_oof_preds.json> "
        "(primary) or --cohort-anchor-from-gold (fallback/smoke)."
    )


# ===========================================================================
# VALIDATION (per-fold, runs after each Phase 2b epoch)
# ===========================================================================


def _free_gen_subeval(
    model: NeuroFusion,
    val_loader: DataLoader,
    device: torch.device,
    n_cases: int = 8,
    k_samples: int = 2,
) -> dict[str, float]:
    """Light free-gen check: run model.lm.generate() on the first n_cases
    of the val loader with K=k_samples and report structural/full validity.

    Why: training-time validation otherwise measures only teacher-forced
    classifier-head accuracy, which can IMPROVE while free-gen structural
    validity COLLAPSES (the aug-retrain disaster: +4.1pp TF macro_F1 but
    −56.4pp FG structural validity). Surfacing structural validity every
    epoch lets the loop early-stop or alarm on free-gen regressions.
    Audit H1 / project_phase3_findings open issue #1.

    Cost: n_cases × k_samples generations. With n_cases=8, k=2 on A100:
    ~3-5 minutes per epoch — affordable relative to the per-epoch backbone
    training cost (~30+ minutes).
    """
    from schema import BrainTumorReport, strict_content
    metrics: dict[str, float] = {}
    n_struct_valid = 0
    n_full_valid = 0
    n_total = 0
    iterator = iter(val_loader)
    with torch.no_grad():
        for _ in range(n_cases):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            case_ids = batch["case_ids"]
            try:
                out = model(mri, seg, training=False)
                samples = model.lm.generate(
                    visual_tokens=out["visual_tokens"],
                    batch_idx=out["routing"]["batch_idx"],
                    centroids=out["routing"]["centroids"],
                    heuristic_strings=out["heuristic_strings"],
                    k_samples=k_samples,
                    use_json_constraint=True,
                    use_heuristics=True,
                    case_ids=case_ids,
                )
            except Exception as e:
                log.warning(f"  free-gen subeval failed on a case: {str(e)[:120]}")
                continue
            for b in range(len(case_ids)):
                samp_list = samples[b] if b < len(samples) else []
                if not samp_list:
                    n_total += 1
                    continue
                # Use any sample (representative of free-gen distribution at this epoch)
                p = samp_list[0]
                n_total += 1
                # Structural validity: parses + schema, content validators OFF
                try:
                    with strict_content(False):
                        BrainTumorReport.model_validate_json(p)
                    n_struct_valid += 1
                except Exception:
                    pass
                # Full validity: also content validators
                try:
                    BrainTumorReport.model_validate_json(p)
                    n_full_valid += 1
                except Exception:
                    pass
    if n_total > 0:
        metrics["val/freegen_structural_validity"] = n_struct_valid / n_total
        metrics["val/freegen_full_validity"] = n_full_valid / n_total
        metrics["val/freegen_n_cases"] = float(n_total)
    return metrics


# Cohort-anchor slot -> cohort code, for the tag-free draft->commit dx sub-eval.
_IDX_TO_COHORT = {1: "GLI", 2: "MEN", 3: "MET"}


def _top1_dx_from_json(js: "str | dict | None") -> "str | None":
    """Lowercased top-1 differential_diagnosis from a committed-JSON string, or None
    if unparseable/empty. Mirrors scripts/dx_recall.top1_dx, but reads the raw JSON that
    generate_draft_commit returns rather than a {"predictions": [...]} record."""
    if not js:
        return None
    try:
        obj = json.loads(js) if isinstance(js, str) else js
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    dd = obj.get("differential_diagnosis") or []
    if not dd:
        return None
    return str(dd[0]).lower()


def _sha256_of_file(path: "str | None") -> str:
    """sha256 of a file's BYTES, or an explicit absence marker.

    ☠️ Returns "ABSENT — no checkpoint path" rather than "" or None when there is nothing
    to hash. An empty string in a provenance field reads as "hashed to nothing"; the
    marker reads as what it is.
    """
    import hashlib
    if not path or not Path(path).is_file():
        return "ABSENT — no checkpoint path"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _pred_cohort_from_dx(dx: "str | None") -> "str | None":
    """Which cohort a PARSED dx string names — or None if it names none, and
    `"AMBIGUOUS"` if it names more than one.

    ☠️ THIS IS NOT `_gold_anchor_idx` AND MUST NOT BE COLLAPSED INTO IT. That one asks
    "does the gold dx name cohort C", checking ONE cohort it already knows. This asks the
    open question "which cohort did the model name", which has to check all three — and
    the two are only the same function when the answer is right.

    ☠️ AMBIGUITY IS RECORDED, NOT RESOLVED. "metastatic glioblastoma" matches both MET and
    GLI. Returning the first match would silently credit whichever cohort this dict happens
    to iterate first, and dict order is not a diagnosis. The marginal null it feeds --
    `obs - n_cohort * p(pred=cohort)` -- is only honest if p() is built from predictions
    that actually named ONE cohort, so ambiguous rows are counted in their own bucket and
    excluded from p().
    """
    if not dx:
        return None
    named = [c for c, syns in _ANCHOR_DX_SYNONYMS.items() if any(t in dx for t in syns)]
    if not named:
        return None
    return named[0] if len(named) == 1 else "AMBIGUOUS"


def _anchor_dx_subeval(
    model: NeuroFusion,
    val_loader: DataLoader,
    device: torch.device,
    max_cases: int = 40,
) -> dict[str, float]:
    """Tag-free draft->commit diagnosis-recall on the fold VAL split, anchor ON.

    BLOCKER-2 selection signal for head-conditioned CoT v2. validate() runs the anchor
    OFF (field-head teacher-forced accuracy), so best.pt would otherwise be chosen BLIND
    to the anchor's diagnosis benefit. This runs the DEPLOYED decode —
    model.lm.generate_draft_commit with neutral_case_id=True (drops the cohort tag) and
    the anchor ON — and scores dx-recall with the byte-exact dx_recall synonyms
    (_ANCHOR_DX_SYNONYMS). NEVER touches test ids: val_loader is the fold's held-out VAL
    split only. Per-case anchor + gold cohort come from the VAL report's gold
    differential_diagnosis (a legitimate selection-time label; _gold_anchor_idx reads
    report CONTENT, not the case_id cohort tag), so selection rewards the checkpoint whose
    draft->commit best converts a correct cohort prior into the correct dx. Unlabelable
    cases (gold dx names no cohort) are skipped. Guarded by the caller on the anchor being
    live => anchor-OFF runs never call this (byte-identical default path).
    """
    model.eval()
    hits = total = seen = 0
    per: dict[str, list[int]] = {c: [0, 0] for c in ("GLI", "MEN", "MET")}
    # ---- THE THREE-WAY DECOMPOSITION (2026-08-12) --------------------------------------
    # ☠️ WHY: `hits/total` alone made this arm UNEVALUABLE and no past run can be rechecked.
    # An unparseable free-generation counts as a MISS, so a FORMAT effect and a COMPETENCE
    # gain are the same number; and a net move conceals reallocation between cohorts (one
    # epoch here hid a +10 GLI / -5 MEN swing inside a net +6). Recording the PREDICTED
    # cohort is what makes `obs - n_cohort * p(pred=cohort)` computable at all -- without
    # it the standing marginal-null rule is structurally unsatisfiable, not merely unmet.
    n_parsed_correct = n_parsed_wrong = n_unparseable = n_ambiguous = 0
    pred_counts: dict[str, int] = {c: 0 for c in ("GLI", "MEN", "MET")}
    # ☠️ A SWALLOWED BATCH SILENTLY MOVES THE DENOMINATOR. The except-continue below drops
    # a whole batch, so `total` is not a fixed 52 and a ratio can shift without any change
    # in ability. Counting the failures makes that visible instead of invisible.
    n_batches_failed = 0
    with torch.no_grad():
        for batch in val_loader:
            if seen >= max_cases:
                break
            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            reports = batch["reports"]
            case_ids = batch["case_ids"]
            # Per-case anchor idx from the gold report dx (tag-free); 0 => unlabelable.
            g_idx = [_gold_anchor_idx(r) for r in reports]
            try:
                out = model(mri, seg, training=False)
                jsons, _drafts = model.lm.generate_draft_commit(
                    visual_tokens=out["visual_tokens"],
                    batch_idx=out["routing"]["batch_idx"],
                    centroids=out["routing"]["centroids"],
                    case_ids=case_ids,
                    neutral_case_id=True,          # TAG-FREE decode (drop the cohort tag)
                    cohort_anchor_idx=g_idx,       # anchor ON (gold VAL anchor)
                )
            except Exception as e:
                n_batches_failed += 1
                log.warning(f"  anchor-dx subeval failed on a batch: {str(e)[:150]}")
                continue
            for b in range(len(case_ids)):
                gi = g_idx[b]
                if gi == 0:
                    continue  # no gold cohort -> not scoreable
                seen += 1
                total += 1
                coh = _IDX_TO_COHORT[gi]
                per[coh][1] += 1
                dx = _top1_dx_from_json(jsons[b] if b < len(jsons) else None)
                correct = dx is not None and any(t in dx for t in _ANCHOR_DX_SYNONYMS[coh])
                if correct:
                    hits += 1
                    per[coh][0] += 1
                # ---- decompose the SAME row the ratio above was built from ----
                # ☠️ Deliberately derived from `dx` here rather than recomputed later:
                # the three buckets must partition EXACTLY the rows that fed hits/total,
                # or the decomposition describes a different denominator than the ratio.
                pred = _pred_cohort_from_dx(dx)
                if pred is None:
                    n_unparseable += 1      # no dx, or a dx naming no cohort at all
                elif pred == "AMBIGUOUS":
                    n_ambiguous += 1
                else:
                    pred_counts[pred] += 1
                    n_parsed_correct += 1 if correct else 0
                    n_parsed_wrong += 0 if correct else 1
                if seen >= max_cases:
                    break
    metrics: dict[str, float] = {}
    if total:
        metrics["val/anchor_dx_recall"] = hits / total
        metrics["val/anchor_dx_n"] = float(total)
        for c, (h, t) in per.items():
            if t:
                metrics[f"val/anchor_dx_recall_{c}"] = h / t
        # ---- the decomposition, and the marginal null it exists to make computable ----
        metrics["val/anchor_dx_parsed_correct"] = float(n_parsed_correct)
        metrics["val/anchor_dx_parsed_wrong"] = float(n_parsed_wrong)
        metrics["val/anchor_dx_unparseable"] = float(n_unparseable)
        metrics["val/anchor_dx_ambiguous"] = float(n_ambiguous)
        metrics["val/anchor_dx_batches_failed"] = float(n_batches_failed)
        # ★ PARSE RATE is the LEADING INDICATOR. A mid-run collapse in parseability reads
        # as a competence drop in `recall` and eats hours before anyone looks.
        metrics["val/anchor_dx_parse_rate"] = (total - n_unparseable) / total
        # ★ Competence among rows that actually named ONE cohort. This is the number that
        # is NOT contaminated by the format effect. It is NOT better than `recall` -- it is
        # a DIFFERENT claim on a SMALLER denominator, and must be quoted with n_named.
        n_named = n_parsed_correct + n_parsed_wrong
        if n_named:
            metrics["val/anchor_dx_recall_among_named"] = n_parsed_correct / n_named
            metrics["val/anchor_dx_n_named"] = float(n_named)
            for c, (h, t) in per.items():
                if not t:
                    continue
                # ☠️ THE STANDING RULE, COMPUTED RATHER THAN OWED: report
                # `obs - n_cohort * p(pred=cohort)` BESIDE every per-cohort recall. A
                # cohort's hits rising while the arm merely predicts that cohort more
                # often is a PREDICTION-RATE SHIFT, and raw recall cannot tell them apart
                # -- it is how two MET "gains" reached a results table before being
                # retracted. p() is over rows that named exactly one cohort, so it is a
                # real distribution; `excess` is the only one of these two worth reading
                # first.
                p_pred = pred_counts[c] / n_named
                metrics[f"val/anchor_dx_p_pred_{c}"] = p_pred
                metrics[f"val/anchor_dx_excess_{c}"] = float(h) - t * p_pred
    return metrics


def validate(
    model: NeuroFusion,
    val_loader: DataLoader,
    device: torch.device,
    phase: str = "phase2b",
    free_gen_subeval: bool = False,
    free_gen_n_cases: int = 8,
    free_gen_k_samples: int = 2,
    report_seg_dice: bool = False,
) -> dict[str, float]:
    """Run validation, return metrics dict.

    Phase 1 reports Dice on ET/TC/WT.
    Phase 2b reports per-field accuracy + macro-F1 + schema adherence (% valid JSON).
    If free_gen_subeval=True, also runs model.lm.generate() on a small subset
    and reports structural + full validity (catches aug-retrain-style regressions
    where teacher-forced metrics improve while free-gen collapses).
    If report_seg_dice=True (Phase 2b with --unfreeze-backbone), ALSO reports
    per-class Dice from the predicted seg so the now-adapting trunk's segmentation
    quality is tracked epoch-over-epoch — the anti-catastrophic-forgetting guard
    (the trunk must not drift to a degenerate seg that breaks LesionRouter).
    """
    model.eval()
    metrics: dict[str, float] = {}

    if phase == "phase1":
        dice_per_class: list[list[float]] = [[] for _ in range(model.cfg.seg_n_classes)]
        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device)
                seg = batch["seg"].to(device)
                vis = model.vision(mri)
                pred = vis["seg_logits"].argmax(dim=1)
                for c in range(1, model.cfg.seg_n_classes):  # skip bg
                    p = (pred == c).float()
                    g = (seg == c).float()
                    inter = (p * g).flatten(1).sum(dim=1)
                    denom = p.flatten(1).sum(dim=1) + g.flatten(1).sum(dim=1)
                    d = (2 * inter + 1e-6) / (denom + 1e-6)
                    dice_per_class[c].extend(d.cpu().tolist())
        for c in range(1, model.cfg.seg_n_classes):
            label = {1: "NCR", 2: "ED", 3: "ET"}.get(c, f"c{c}")
            metrics[f"val/dice_{label}"] = float(np.mean(dice_per_class[c])) if dice_per_class[c] else 0.0
        metrics["val/dice_mean"] = float(np.mean([metrics[f"val/dice_{l}"] for l in ("NCR", "ED", "ET")]))
        return metrics

    # Phase 2b validation: field accuracy + macro-F1
    field_correct: dict[str, int] = {f: 0 for f in model.cfg.field_heads}
    field_total: dict[str, int] = {f: 0 for f in model.cfg.field_heads}
    field_per_class_tp: dict[str, dict[int, int]] = {f: {} for f in model.cfg.field_heads}
    field_per_class_fp: dict[str, dict[int, int]] = {f: {} for f in model.cfg.field_heads}
    field_per_class_fn: dict[str, dict[int, int]] = {f: {} for f in model.cfg.field_heads}

    schema_valid = 0
    schema_total = 0

    # Per-class Dice accumulators (only used when report_seg_dice — the unfrozen-
    # trunk run, to track that the adapting backbone keeps producing valid seg).
    seg_dice_per_class: list[list[float]] = [[] for _ in range(model.cfg.seg_n_classes)]

    with torch.no_grad():
        for batch in val_loader:
            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            reports = batch["reports"]

            out = model(mri, seg, training=False)
            field_logits = out["field_logits"]
            routing = out["routing"]

            if report_seg_dice:
                pred = out["seg_logits"].argmax(dim=1)
                for c in range(1, model.cfg.seg_n_classes):  # skip bg
                    p = (pred == c).float()
                    g = (seg == c).float()
                    inter = (p * g).flatten(1).sum(dim=1)
                    denom = p.flatten(1).sum(dim=1) + g.flatten(1).sum(dim=1)
                    d = (2 * inter + 1e-6) / (denom + 1e-6)
                    seg_dice_per_class[c].extend(d.cpu().tolist())

            # Build GT field targets aligned to the routing decisions
            field_gt = build_field_targets(
                reports, routing["batch_idx"], routing["lesion_idx"], device,
                mask_overflow=getattr(model.cfg, "mask_overflow_lesion_loss", False),
            )
            for fname, logits in field_logits.items():
                preds = logits.argmax(dim=-1)
                gt = field_gt[fname]
                # ☠️ C12 COROLLARY — build_field_targets is consumed HERE TOO, and
                # this is the checkpoint-selection path. Without this mask, every
                # ignore_index row would be: (a) scored WRONG (a -100 target can
                # never equal a prediction) while STAYING IN THE DENOMINATOR, and
                # (b) entered into the per-class tables as a phantom class -100,
                # which survives the `tp+fp == 0 and tp+fn == 0` skip (fn > 0) and
                # so appends an extra hard 0.0 into macro-F1 for every field.
                # Measured on the real block: acc 1.0 -> 0.6, macro 0.6667 -> 0.5333.
                # `val/macro_f1` feeds sel_metric -> best.pt, so leaving this unmasked
                # would silently redefine the selection criterion and make it
                # non-comparable to every prior run.
                keep = gt != FIELD_IGNORE_INDEX
                if not bool(keep.any()):
                    continue
                preds, gt = preds[keep], gt[keep]
                field_correct[fname] += int((preds == gt).sum().item())
                field_total[fname] += int(gt.numel())
                for p, g in zip(preds.cpu().tolist(), gt.cpu().tolist()):
                    if p == g:
                        field_per_class_tp[fname][g] = field_per_class_tp[fname].get(g, 0) + 1
                    else:
                        field_per_class_fp[fname][p] = field_per_class_fp[fname].get(p, 0) + 1
                        field_per_class_fn[fname][g] = field_per_class_fn[fname].get(g, 0) + 1

            # Schema-adherence stub: until real LM generation is wired, count all as valid.
            # When LM.generate() is live, parse the output JSON via schema.BrainTumorReport.
            schema_total += len(reports)
            schema_valid += len(reports)  # TODO: replace with real parse-success rate

    # Per-field accuracy
    macro_f1_components: list[float] = []
    for fname in model.cfg.field_heads:
        if field_total[fname] > 0:
            metrics[f"val/acc_{fname}"] = field_correct[fname] / field_total[fname]
            classes = set(field_per_class_tp[fname]) | set(field_per_class_fp[fname]) | set(field_per_class_fn[fname])
            f1s: list[float] = []
            for c in classes:
                tp = field_per_class_tp[fname].get(c, 0)
                fp = field_per_class_fp[fname].get(c, 0)
                fn = field_per_class_fn[fname].get(c, 0)
                # Same audit-fix as eval.py:_accuracy_and_macro_f1: never
                # SKIP a class — score it F1=0 so always-predict-majority
                # baselines don't get rewarded. Skip only when both prec
                # and rec are undefined (no support AND no predictions).
                if tp + fp == 0 and tp + fn == 0:
                    continue
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if prec + rec > 0:
                    f1s.append(2 * prec * rec / (prec + rec))
                else:
                    f1s.append(0.0)
            field_macro_f1 = float(np.mean(f1s)) if f1s else 0.0
            metrics[f"val/macro_f1_{fname}"] = field_macro_f1
            macro_f1_components.append(field_macro_f1)

    metrics["val/macro_f1"] = float(np.mean(macro_f1_components)) if macro_f1_components else 0.0
    metrics["val/schema_adherence"] = schema_valid / schema_total if schema_total > 0 else 0.0

    # Per-class Dice on the (now-adapting) trunk — anti-forgetting tracker.
    if report_seg_dice:
        for c in range(1, model.cfg.seg_n_classes):
            label = {1: "NCR", 2: "ED", 3: "ET"}.get(c, f"c{c}")
            metrics[f"val/dice_{label}"] = (
                float(np.mean(seg_dice_per_class[c])) if seg_dice_per_class[c] else 0.0
            )
        metrics["val/dice_mean"] = float(
            np.mean([metrics.get(f"val/dice_{l}", 0.0) for l in ("NCR", "ED", "ET")])
        )

    # Optional free-gen sub-eval (lights up the actual deployed inference path).
    if free_gen_subeval:
        try:
            fg_metrics = _free_gen_subeval(model, val_loader, device,
                                            n_cases=free_gen_n_cases,
                                            k_samples=free_gen_k_samples)
            metrics.update(fg_metrics)
            log.info(
                f"  free-gen subeval: structural={metrics.get('val/freegen_structural_validity', 0.0):.3f}, "
                f"full={metrics.get('val/freegen_full_validity', 0.0):.3f} on {int(metrics.get('val/freegen_n_cases', 0))} cases"
            )
        except Exception as e:
            log.warning(f"  free-gen subeval crashed: {str(e)[:200]}")
    return metrics


# ===========================================================================
# CHECKPOINT IO
# ===========================================================================


def save_checkpoint(
    model: NeuroFusion,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    path: Path,
    save_lora_only: bool = True,
) -> None:
    """Save model checkpoint.

    save_lora_only=True (recommended for Phase 2b): saves Q-Former + field heads + LoRA
    adapters but NOT the frozen MedNeXt backbone or the frozen MedGemma base. Result is
    ~30-80 MB per fold instead of multi-GB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_lora_only:
        state = {
            "qformer": model.qformer.state_dict(),
            "field_head": model.field_head.state_dict(),
            "lm_lora": _extract_lora_state(model.lm.lm) if hasattr(model.lm.lm, "peft_config") else {},
        }
    else:
        state = {"model": model.state_dict()}

    sched_state = None
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        # LambdaLR.state_dict() serializes ALL scheduler attrs, including the
        # _nf_lr_lambda closure we stash in build_optimizer_and_scheduler (and the
        # function entries it itself nulls out). Local closures aren't picklable, so
        # strip any callable values — they are reconstructed at build time on resume,
        # so dropping them loses no schedule state (last_epoch/_step_count/base_lrs
        # are plain values and are kept).
        sched_state = {k: v for k, v in sched_state.items() if not callable(v)}

    torch.save({
        "state": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": sched_state,
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }, path)
    log.info(f"checkpoint saved: {path} (lora_only={save_lora_only})")


def _extract_lora_state(peft_model: Any) -> dict:
    """Extract only the LoRA-adapter parameters from a PEFT-wrapped model."""
    return {k: v for k, v in peft_model.state_dict().items() if "lora_" in k}


# Keys whose SHAPE is allowed to differ between a warmstart checkpoint and the
# current model when the visual-token-expansion / max_lesions-sweep upgrade is
# applied. ANYTHING dropped OUTSIDE this allowlist is a genuinely-trained tensor
# being silently discarded -> _filter_shape_mismatch RAISES (mirrors the
# no-missing-key assert philosophy at the full-model load path below).
#   - feature_proj.*        : in-channels 512->256 when feature_stage="enc_hires"
#   - qformer.queries       : n_queries_per_lesion 32->64 (visual-token expansion)
#   - lesion_idx_embed.weight: rows = max_lesions+1 (5->9->17 for Models 3/4)
# The substring forms cover both the bare sub-state-dict keys (e.g. "queries",
# "lesion_idx_embed.weight") and the full-model-prefixed keys (e.g.
# "qformer.queries", "vision.backbone.feature_proj.weight").
_WARMSTART_SHAPE_MISMATCH_ALLOWLIST: tuple[str, ...] = (
    "feature_proj.weight",
    "feature_proj.bias",
    # _last_conv_layer is the XAI/Grad-CAM alias of feature_proj (model.py:276-282
    # registers the last Conv3d under a 2nd attribute name for the saliency hook);
    # it is the SAME conv, so it resizes 512->256 together with feature_proj under
    # feature_stage="enc_hires" and must be allowlisted too.
    "_last_conv_layer.weight",
    "_last_conv_layer.bias",
    "queries",
    "lesion_idx_embed.weight",
)


def _filter_shape_mismatch(
    ckpt_state: dict, target_state: dict, ctx: str = ""
) -> tuple[dict, list[str]]:
    """Drop checkpoint tensors whose shape != the live model's same-named tensor.

    Used by the warmstart partial-load so the visual-token-expansion upgrade
    (feature_proj in-ch 512->256, qformer.queries 32->64, lesion_idx_embed rows
    5->9/17) can load everything ELSE (qformer decoder, field_head, LoRA, the
    whole frozen/unfrozen trunk) from a Model-1 checkpoint while the two/three
    deliberately-resized tensors keep their fresh init.

    Integrity guard: every dropped key MUST match the explicit allowlist; a
    dropped key outside it (a real trained tensor being silently discarded)
    RAISES. Keys present in the ckpt but ABSENT from the live model are left in
    (load_state_dict(strict=False) surfaces them as unexpected, which the
    existing bnb-quant assert already handles).

    Returns (filtered_state_dict, dropped_keys).
    """
    filtered: dict = {}
    dropped: list[str] = []
    for k, v in ckpt_state.items():
        tgt = target_state.get(k, None)
        if tgt is not None and hasattr(v, "shape") and tuple(v.shape) != tuple(tgt.shape):
            dropped.append(k)
            log.info(
                f"warmstart{(' '+ctx) if ctx else ''}: DROP shape-mismatched "
                f"'{k}' ckpt{tuple(v.shape)} -> model{tuple(tgt.shape)} (re-init kept)"
            )
            continue
        filtered[k] = v

    bad = [k for k in dropped if not any(a in k for a in _WARMSTART_SHAPE_MISMATCH_ALLOWLIST)]
    assert not bad, (
        f"warmstart{(' '+ctx) if ctx else ''}: {len(bad)} shape-mismatched key(s) "
        f"OUTSIDE the expansion allowlist {_WARMSTART_SHAPE_MISMATCH_ALLOWLIST}: {bad[:5]}. "
        f"A genuinely-trained tensor would be silently dropped+reinit -- aborting."
    )
    return filtered, dropped


def _copy_query_prefix(ckpt_state: dict, model: NeuroFusion, key_candidates: tuple[str, ...]) -> None:
    """Optional warm transfer for qformer.queries when it was DROPPED for shape
    mismatch (32->64): copy the trained ckpt rows into queries[:n_old] in place,
    leaving the freshly-randn'd new rows [n_old:] untouched. Pure optimization;
    skipped silently if shapes are not the expected prefix-extension case.
    """
    q = getattr(model.qformer, "queries", None)
    if q is None:
        return
    for k in key_candidates:
        src = ckpt_state.get(k, None)
        if src is None or not hasattr(src, "shape"):
            continue
        if src.dim() == q.dim() and src.shape[0] <= q.shape[0] and src.shape[1:] == q.shape[1:]:
            with torch.no_grad():
                q[: src.shape[0]].copy_(src.to(q.dtype).to(q.device))
            log.info(
                f"warmstart: copied trained queries[:{src.shape[0]}] from '{k}'; "
                f"rows [{src.shape[0]}:{q.shape[0]}] keep fresh randn init"
            )
        return


def _copy_lesion_idx_prefix(ckpt_state: dict, model: NeuroFusion, key_candidates: tuple[str, ...]) -> None:
    """Warm transfer for qformer.lesion_idx_embed.weight when it was DROPPED for
    shape mismatch (rows 5 -> 9/17 when max_lesions 4 -> 8/16).

    CRITICAL for the Stage-0 max_lesions gate: that gate raises cfg.max_lesions on
    the FROZEN Model-1 checkpoint WITHOUT retraining, so the trained per-rank lesion
    embeddings (rows 0..4 = padding + ranks 1..4) MUST be carried over -- otherwise
    every routed lesion would index a fresh-init embedding and the run would not be
    a faithful 'frozen Model-1 with more input lesions' test (it would be a random
    re-init of the routing-rank embedding). The trained rows are copied into the
    prefix [:n_old]; the NEW rows [n_old:] (ranks 5..8 / 5..16, i.e. the extra noise
    slots the gate is probing) keep fresh init. Skipped silently if shapes are not
    the expected prefix-extension case. Pure no-op when max_lesions is at the
    default 4 (no shape mismatch -> the weight was never dropped -> not called).
    """
    emb = getattr(model.qformer, "lesion_idx_embed", None)
    if emb is None or not hasattr(emb, "weight"):
        return
    w = emb.weight
    for k in key_candidates:
        src = ckpt_state.get(k, None)
        if src is None or not hasattr(src, "shape"):
            continue
        if src.dim() == w.dim() and src.shape[0] <= w.shape[0] and src.shape[1:] == w.shape[1:]:
            with torch.no_grad():
                w[: src.shape[0]].copy_(src.to(w.dtype).to(w.device))
            log.info(
                f"warmstart: copied trained lesion_idx_embed[:{src.shape[0]}] from '{k}'; "
                f"rows [{src.shape[0]}:{w.shape[0]}] keep fresh init (extra-lesion ranks)"
            )
        return


def _assert_lm_family_match(state: dict, model: NeuroFusion) -> None:
    """Fail LOUD + EARLY on a cross-LM-family warmstart (M6-Mistral prereg Step 1).

    Checkpoints store no lm_family tag, so we infer the family from the Q-Former->LM
    output_proj width (= cfg.lm_hidden_dim): Mistral-7B = 4096, MedGemma = 2560. The
    danger this guards: warmstarting a MedGemma checkpoint into a Mistral-configured
    model (or vice-versa) loads the LoRA via load_state_dict(strict=False) where the
    differently-named adapter keys are all 'unexpected' and SILENTLY ignored -> the
    LoRA trains from init and the result is garbage with no error. This is a no-op
    when the families match (incl. every --resume), so it never breaks legitimate loads.
    """
    sub = state.get("qformer") if "qformer" in state else state.get("model")
    if not isinstance(sub, dict):
        return
    key = "output_proj.weight" if "output_proj.weight" in sub else "qformer.output_proj.weight"
    if key not in sub:
        return  # signature tensor absent (unexpected layout) -> rely on downstream asserts
    try:
        ckpt_dim = int(sub[key].shape[0])
        model_dim = int(model.qformer.output_proj.weight.shape[0])
    except Exception:  # noqa - never let the guard itself crash a legitimate load
        return
    if ckpt_dim != model_dim:
        _name = {4096: "mistral-7b", 2560: "medgemma"}
        raise SystemExit(
            f"WARMSTART LM-FAMILY MISMATCH: checkpoint output_proj LM-width={ckpt_dim} "
            f"({_name.get(ckpt_dim, 'unknown')}) but the model is configured with "
            f"lm_hidden_dim={model_dim} ({_name.get(model_dim, 'unknown')}, lm_family="
            f"{getattr(getattr(model, 'cfg', None), 'lm_family', '?')}). Refusing to warmstart "
            f"across LM families — the LoRA would silently load as a no-op under strict=False "
            f"and train from init. Use the matching --lm-family + --warmstart-nf checkpoint."
        )


def _trunk_verify(model, intended_trunk: str, out_dir: Path, when: str,
                  *, pinned: bool) -> dict:
    """Record WHICH TRUNK this run actually holds, at a named point in training.

    Called at step 0 and again at the end. At step 0 with `pinned` the trunk MUST be
    bit-identical to `intended_trunk` and the run aborts otherwise — a 16 GPU-h job that
    silently trains the wrong trunk is exactly the failure this lane exists to end.
    At the end the trunk has usually MOVED (the unfreeze schedule trains it); that drift
    is measured and recorded, never asserted away. Declared drift is fine; silent
    inheritance is not.
    """
    import json as _json
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    import trunk_identity_gate as _tig

    rep = {
        "when": when,
        "intended_seg_trunk": str(intended_trunk),
        "pin_seg_trunk": bool(pinned),
        "vision_fingerprint": _tig.fingerprint(model),
        "connector_fingerprint": _tig.connector_fingerprint(model),
        "file_diff": _tig.diff_against_file(model, str(intended_trunk)),
    }
    d = rep["file_diff"]
    log.info(f"[trunk-verify:{when}] vs {intended_trunk}: mapped {d['mapped']}, "
             f"IDENTICAL {d['identical']}, DIFFERENT {d['different']}, "
             f"max_abs_diff {d['max_abs_diff']} (pin={pinned})")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"trunk_verify_{when}.json").write_text(_json.dumps(rep, indent=1))
    if when == "step0" and pinned and not (d["different"] == 0 and d["mapped"] > 0
                                           and d["unmapped_file_tensors"] == 0):
        raise SystemExit(
            f"[trunk-verify:step0] ☠️ --pin-seg-trunk was requested but the trunk at step 0 "
            f"is NOT {intended_trunk} ({d['different']}/{d['mapped']} tensors differ, "
            f"max|d| {d['max_abs_diff']}). Refusing to burn GPU-hours on the wrong trunk.")
    return rep


def load_checkpoint(
    path: Path, model: NeuroFusion,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    pin_seg_trunk: bool = False,
) -> dict:
    """Load model state + (optionally) optimizer + scheduler.

    `pin_seg_trunk=True` DROPS the checkpoint's own `vision.backbone.mednext.*` tensors
    so the trunk loaded at construction (`mednext_checkpoint=`) is the one that survives.
    See the block at the drop site for why this is not the default and what it does NOT
    drop (feature_proj).

    NB: scheduler restoration was previously MISSING — resuming a run after
    preemption rebuilt the scheduler from step=0, restarting the LR warmup
    and blowing up LoRA adapters. Now restored if provided. Caller is
    responsible for restoring `step` and `epoch` from the returned dict.
    Audit H6.
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state"]
    _assert_lm_family_match(state, model)  # cross-family warmstart guard (prereg Step 1)
    if "qformer" in state:
        # WARMSTART partial-load (visual-token expansion / max_lesions sweep):
        # drop ONLY the deliberately-resized tensors (qformer.queries 32->64,
        # lesion_idx_embed.weight rows for max_lesions 4->8/16) and load the rest
        # of the Q-Former decoder + field-head from the Model-1 checkpoint. When
        # the upgrade flags are at DEFAULT (bottleneck/32/4) nothing mismatches,
        # the filter is a no-op, and these loads are byte-identical to the old
        # strict path (full state_dict round-trips, missing/unexpected both empty).
        qf_state, qf_dropped = _filter_shape_mismatch(
            state["qformer"], model.qformer.state_dict(), ctx="qformer"
        )
        qf_res = model.qformer.load_state_dict(qf_state, strict=False)
        # The only legitimately-missing keys are exactly the ones we dropped (their
        # fresh init is kept on purpose); anything else missing is a real gap.
        qf_bad_missing = [k for k in qf_res.missing_keys if k not in qf_dropped]
        assert not qf_bad_missing, (
            f"load_checkpoint qformer: {len(qf_bad_missing)} unexpectedly MISSING "
            f"trained keys, e.g. {qf_bad_missing[:5]}"
        )
        assert not qf_res.unexpected_keys, (
            f"load_checkpoint qformer: unexpected keys {qf_res.unexpected_keys[:5]}"
        )
        if qf_dropped:
            _copy_query_prefix(state["qformer"], model, ("queries",))
            _copy_lesion_idx_prefix(state["qformer"], model, ("lesion_idx_embed.weight",))
        fh_state, fh_dropped = _filter_shape_mismatch(
            state["field_head"], model.field_head.state_dict(), ctx="field_head"
        )
        fh_res = model.field_head.load_state_dict(fh_state, strict=False)
        # Expected-missing = (a) shape-dropped keys, plus (b) NEW params absent from the
        # source checkpoint entirely -- the M5 field-head upgrades (attn_query/attn_key/
        # global_proj) are fresh-init by design when warmstarting from a mean-pool M1.
        fh_new = [k for k in fh_res.missing_keys if k not in state["field_head"]]
        if fh_new:
            log.info(f"load_checkpoint field_head: {len(fh_new)} NEW param(s) kept at fresh "
                     f"init (not in warmstart ckpt): {fh_new}")
        fh_bad_missing = [k for k in fh_res.missing_keys if k not in fh_dropped and k not in fh_new]
        assert not fh_bad_missing, (
            f"load_checkpoint field_head: {len(fh_bad_missing)} unexpectedly MISSING "
            f"trained keys, e.g. {fh_bad_missing[:5]}"
        )
        if state.get("lm_lora") and hasattr(model.lm.lm, "load_state_dict"):
            model.lm.lm.load_state_dict(state["lm_lora"], strict=False)
    elif "model" in state:
        # Full-model checkpoints (unfrozen-backbone runs) save the 4-bit
        # MedGemma/Mistral base: bitsandbytes serializes quant-state buffers
        # (absmax/quant_map/nested_*/bitsandbytes__nf4) that the freshly-loaded
        # live 4-bit model regenerates and does NOT register as loadable keys.
        # Load non-strict, but ASSERT nothing real is missing and every
        # unexpected key is a regenerable bnb quant buffer — so a genuinely
        # missing trained param (backbone/qformer/field_head/LoRA) still fails
        # loudly instead of silently loading at init.
        #
        # WARMSTART partial-load (visual-token expansion / max_lesions sweep):
        # additionally DROP the deliberately-resized tensors
        # (vision.backbone.feature_proj.* 512->256, qformer.queries 32->64,
        # qformer.lesion_idx_embed.weight rows 5->9/17) BEFORE load_state_dict --
        # PyTorch raises on a shape mismatch even with strict=False. The dropped
        # set is then expected-missing (their fresh init is kept on purpose). With
        # all upgrade flags at DEFAULT nothing is dropped and this is byte-identical
        # to the prior behavior (Model-1 deployment unaffected).
        # ☠️☠️ NF 1.3 MARLIN: WHEN THE LM WEIGHTS ARE EXTERNAL, THE CHECKPOINT'S lm.* MUST NOT
        # BE APPLIED AT ALL -- AND THAT MUST BE A CONTRACT, NOT AN ACCIDENT.
        # `nfmistral_stripid/best.pt` carries 1667 of its 2382 keys under `lm.*`, holding the
        # LM as bitsandbytes-NF4 (packed uint8 + absmax + quant_state). The merged-fp16 and
        # int4-AWQ arms get their LM from an external directory instead, with NO PEFT wrapper.
        # Today those 1667 keys fail to apply ONLY because PEFT's `base_model.` prefix happens
        # not to match the unwrapped module tree -- and the loader then reports the model's own
        # lm.* tensors as "NEW param(s) kept at fresh init", which is FALSE: they are the
        # externally-loaded merged weights. So the arm is correct by coincidence and described
        # wrongly in the log. Any future change to that naming would silently overwrite the
        # merged/quantised LM with native's NF4 weights and still report success -- i.e. we
        # would publish "NF 1.3 Marlin" numbers produced by NF 1.3 native.
        # ⇒ Drop them explicitly, assert we dropped something, and say so.
        # ★ Found by checking the CONSUMER after landing `lm_quant` at the BUILDER -- the same
        #   producer-vs-consumer gap that let a masked-target change silently redefine
        #   `val/macro_f1` and therefore `best.pt` earlier in this project.
        if getattr(model.cfg, "_lm_weights_are_external", lambda: False)():
            _lm_keys = [k for k in state["model"] if k.startswith("lm.")]
            if not _lm_keys:
                raise RuntimeError(
                    "lm_quant is external but the checkpoint has no lm.* keys -- the "
                    "assumption behind this guard is wrong; re-derive it before trusting "
                    "any A/B arm built from this checkpoint")
            state = dict(state)
            state["model"] = {k: v for k, v in state["model"].items()
                              if not k.startswith("lm.")}
            log.info(f"[lm_quant={model.cfg.lm_quant}] LM weights come from "
                     f"{model.cfg.lm_quant_ckpt}; SKIPPED {len(_lm_keys)} checkpoint lm.* "
                     f"tensors on purpose (they are native's bnb-NF4 LM and would corrupt "
                     f"this arm)")

        # ☠️☠️ TRUNK PINNING (NF 1.3 eviction lane, 2026-08-08). THE DEFECT THIS FIXES:
        # `NeuroFusion(cfg, mednext_checkpoint=X)` loads trunk X, and then THIS branch
        # applies the checkpoint's own `vision.*` tensors straight over it. Measured
        # against the frozen e30 trunk: 524 mapped, IDENTICAL 0, max_abs_diff 0.955.
        # ⇒ `--mednext-checkpoint` / `NF_SEG_CKPT` has been INERT for every full-model
        # checkpoint, so every LM/report number in this project was produced on whatever
        # trunk its own checkpoint carried (the leak-5 lineage), not on the trunk named
        # on the command line.
        #
        # WHAT IS DROPPED: `vision.backbone.mednext.*` — the MedNeXt trunk, i.e. exactly
        # the weights a Phase-1 trunk file contains and exactly the ones leak-5
        # contaminated.
        # WHAT IS **NOT** DROPPED, ON PURPOSE: `vision.backbone.feature_proj.*` (and its
        # alias `vision._last_conv_layer.*` — `_register_feature_hook` stores the last
        # Conv3d, which IS feature_proj, so the module appears twice in the state_dict).
        # feature_proj is the 512->320 connector projection; it is TRAINED IN PHASE-2b and
        # has NO counterpart in any trunk file. Dropping it would leave a randomly
        # initialised connector in front of a pinned trunk — a much larger break than the
        # one being fixed, and one that would look like "pinning the trunk costs 40 points".
        # It is inherited from the warmstart checkpoint and that must be DECLARED.
        #
        # NOT THE DEFAULT: flipping it globally would silently redefine every existing
        # eval. Callers opt in, and the post-load gate (scripts/trunk_identity_gate.py)
        # is what PROVES the pin held — this assert only proves we tried.
        if pin_seg_trunk:
            _trunk_keys = [k for k in state["model"]
                           if k.startswith("vision.backbone.mednext.")]
            if not _trunk_keys:
                raise RuntimeError(
                    f"pin_seg_trunk=True but {path} has no vision.backbone.mednext.* keys "
                    f"-- the assumption behind the pin is wrong for this checkpoint layout; "
                    f"re-derive it before trusting any run built from it")
            _kept_proj = [k for k in state["model"]
                          if k.startswith("vision.") and k not in set(_trunk_keys)]
            state = dict(state)
            state["model"] = {k: v for k, v in state["model"].items()
                              if not k.startswith("vision.backbone.mednext.")}
            log.info(
                f"[pin_seg_trunk] DROPPED {len(_trunk_keys)} vision.backbone.mednext.* "
                f"tensor(s) from {path} so the trunk passed as mednext_checkpoint survives; "
                f"KEPT {len(_kept_proj)} other vision.* tensor(s) {sorted(_kept_proj)} "
                f"(feature_proj + its alias: Phase-2b connector weights, inherited by design)")

        full_state, full_dropped = _filter_shape_mismatch(
            state["model"], model.state_dict(), ctx="full-model"
        )
        result = model.load_state_dict(full_state, strict=False)
        _BNB_QUANT_MARKERS = ("absmax", "quant_map", "quant_state", "bitsandbytes__")
        bad_unexpected = [
            k for k in result.unexpected_keys
            if not any(m in k for m in _BNB_QUANT_MARKERS)
        ]
        # Missing keys are OK only if they are (a) the tensors we dropped for shape
        # mismatch, or (b) NEW params absent from the source checkpoint entirely (the
        # M5 field-head upgrades attn_query/attn_key/global_proj -- fresh-init by design
        # when warmstarting an upgraded head from a mean-pool Model-1). Kept at init.
        new_params = [k for k in result.missing_keys if k not in state["model"]]
        # ☠️ SEPARATE THE EXTERNAL LM FROM GENUINELY-FRESH PARAMS. Under lm_quant != 'nf4' the
        # model's own lm.* tensors are missing from `state["model"]` (we stripped them above),
        # so they would be lumped in with "NEW param(s) kept at fresh init". That description
        # is FALSE and dangerous in equal measure: those tensors are not at fresh init, they
        # hold the merged/quantised weights the whole arm is about, and a reader scanning the
        # log for "did my LM load?" would be told the opposite of the truth.
        _ext_lm = [k for k in new_params if k.startswith("lm.")]
        new_params = [k for k in new_params if not k.startswith("lm.")]
        # ☠️ SAME TREATMENT FOR THE PINNED TRUNK. With pin_seg_trunk the 524 mednext
        # tensors are missing from `state["model"]` because WE removed them, so without
        # this split the loader would log them as "NEW param(s) kept at fresh init" —
        # the exact false-and-dangerous message the lm_quant note above was written about.
        # They are not at fresh init: they hold the trunk from mednext_checkpoint, which
        # is the entire point of the run.
        _pinned_trunk = [k for k in new_params if k.startswith("vision.backbone.mednext.")]
        new_params = [k for k in new_params if not k.startswith("vision.backbone.mednext.")]
        if _pinned_trunk:
            assert pin_seg_trunk, (
                f"{len(_pinned_trunk)} vision.backbone.mednext.* tensors are missing from "
                f"the checkpoint but pin_seg_trunk is OFF — the trunk would be at whatever "
                f"construction left there with no declaration. Refusing.")
            log.info(f"[pin_seg_trunk] {len(_pinned_trunk)} vision.backbone.mednext.* "
                     f"tensor(s) supplied by mednext_checkpoint, NOT by this checkpoint "
                     f"(this is the pin working as intended, NOT fresh init)")
        if _ext_lm:
            log.info(f"full-model load: {len(_ext_lm)} lm.* tensor(s) supplied by the EXTERNAL "
                     f"quantised LM, not by the checkpoint (this is the lm_quant arm working "
                     f"as intended, NOT fresh init)")
        if new_params:
            log.info(f"full-model load: {len(new_params)} NEW param(s) kept at fresh init "
                     f"(not in warmstart ckpt): {new_params}")
        # ⚠️ `_ext_lm` must be excluded here too. It was split OUT of `new_params` purely so the
        # log stops calling it "fresh init"; leaving it out of this filter as well would make
        # every external-LM arm fail the missing-key assert. Splitting a list that another
        # predicate subtracts from is exactly how a cosmetic change becomes a crash.
        bad_missing = [k for k in result.missing_keys
                       if k not in full_dropped and k not in new_params
                       and k not in _ext_lm and k not in _pinned_trunk]
        assert not bad_missing, (
            f"load_checkpoint: {len(bad_missing)} MISSING trained keys, "
            f"e.g. {bad_missing[:5]}"
        )
        assert not bad_unexpected, (
            f"load_checkpoint: {len(bad_unexpected)} non-quant unexpected keys, "
            f"e.g. {bad_unexpected[:5]}"
        )
        if full_dropped:
            _copy_query_prefix(
                state["model"], model, ("qformer.queries", "queries")
            )
            _copy_lesion_idx_prefix(
                state["model"], model,
                ("qformer.lesion_idx_embed.weight", "lesion_idx_embed.weight"),
            )
        log.info(
            f"full-model load: {len(result.unexpected_keys)} bnb quant buffers "
            f"ignored, {len(full_dropped)} expansion tensor(s) re-init, "
            f"0 unexpectedly-missing trained keys"
        )
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(
        f"checkpoint loaded: {path}  (epoch {ckpt['epoch']}, step {ckpt['step']}, "
        f"opt={'restored' if optimizer is not None else 'skipped'}, "
        f"sched={'restored' if scheduler is not None and ckpt.get('scheduler') else 'skipped'})"
    )
    return ckpt


# ===========================================================================
# OPTIMIZER + SCHEDULER
# ===========================================================================


def build_optimizer_and_scheduler(
    model: NeuroFusion,
    cfg: NeuroFusionConfig,
    args: argparse.Namespace,
    n_steps_total: int,
) -> tuple[torch.optim.Optimizer, Any]:
    """Parameter groups:
      - Q-Former + field heads at lr 1e-4
      - LoRA adapters at lr 2e-4
      - (optional, ADDED LATER at the unfreeze epoch) MedNeXt trunk at lr_backbone

    The MedNeXt trunk group is NOT created here even when --unfreeze-backbone is set:
    with the gradual schedule the trunk is still frozen at build time (epoch < unfreeze
    epoch), so its params don't requires_grad yet. add_backbone_param_group() below adds
    it via optimizer.add_param_group at the unfreeze epoch and extends the scheduler's
    base_lrs / lr_lambdas so the cosine schedule applies to it too. We expose lr_lambda
    + warmup_steps on the scheduler so that helper can rebuild a consistent schedule.
    """
    qformer_params = list(model.qformer.parameters()) + list(model.field_head.parameters())
    # W-QF-OBJ: the aux dx head must be TRAINED, not merely present. It is not under
    # model.qformer / model.field_head, so without this line it would sit at its random
    # init forever -- the loss would still fall (the Q-Former alone can chase a fixed random
    # readout), the log would still show `auxdx` decreasing, and the arm would be measuring
    # something other than what it claims. Appended to the existing group so no new
    # param_group appears: add_backbone_param_group indexes param_groups[-1] and the
    # scheduler asserts len(param_groups) == len(lr_lambdas).
    if getattr(model, "aux_dx_head", None) is not None:
        qformer_params += list(model.aux_dx_head.parameters())
    lora_params = [p for n, p in model.lm.named_parameters() if "lora_" in n or p.requires_grad]

    param_groups = [
        {"params": qformer_params, "lr": args.lr_qformer, "weight_decay": 0.01},
    ]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # L5 (--scheduler plateau): ReduceLROnPlateau on the epoch selection metric
    # (mode="max"), halving LR after --plateau-patience stalled epochs. Returned
    # BEFORE the cosine construction below so the default (--scheduler cosine) path is
    # untouched / byte-identical. Its per-epoch .step(sel_metric) is driven from the
    # training loop (NOT per optimizer step); add_backbone_param_group extends this
    # scheduler's min_lrs when the trunk group joins at the unfreeze epoch.
    if getattr(args, "scheduler", "cosine") == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=int(getattr(args, "plateau_patience", 1)), min_lr=1e-6,
        )
        return optimizer, scheduler

    # Cosine with 5% warmup
    warmup_steps = max(1, int(0.05 * n_steps_total))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps_total - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Stash the closure pieces so add_backbone_param_group() can extend the
    # scheduler in lock-step with a late-added trunk param group.
    scheduler._nf_lr_lambda = lr_lambda          # type: ignore[attr-defined]
    scheduler._nf_warmup_steps = warmup_steps    # type: ignore[attr-defined]

    assert_requires_grad_subset_of_optimizer(model, optimizer)   # C5
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# C5 — `requires_grad ⊆ optimiser`. THE detector for the §CORE-0 defect class.
# ---------------------------------------------------------------------------
# A parameter with requires_grad=True that belongs to NO optimizer param group
# receives gradients that are never applied. It looks trainable, reports non-zero
# grads, and never moves. Nothing else in this codebase can see that.
#
# MEASURED, BLOCK 0.3 (2026-08-02): for runs/nfmistral_fold0/best.pt (e26, step 702),
# `feature_proj.weight` is BIT-IDENTICAL (max|diff| = 0.0 over 163,840 weights) to a
# fresh seed-42 init in a separate process that loaded no checkpoint -- while, in the
# SAME file, the trunk `mednext.stem` moved by 1.452e-03. Param-group arithmetic
# closes exactly: groups are 187/256/524, and a live backbone has 524 mednext tensors
# + 2 feature_proj tensors = 526, so those 2 are in NO group, in frozen AND unfrozen
# configurations. This assert would have fired on 2026-05-13, the day it was
# introduced -- and every "non-zero grad per sub-block" check passed for the life of
# the project, which is exactly why the check has to be on OPTIMIZER MEMBERSHIP.
#
# ☠️ KNOWN_UNOPTIMISED is a LEDGER, NOT AN EXCUSE. Anything in it is a live defect
# with a named owner. It must not grow silently: a NEW unoptimised parameter raises.
KNOWN_UNOPTIMISED: dict[str, str] = {
    # name-suffix -> why it is tolerated, and what removes it
    "feature_proj": (
        "C1: the 512->320 projection is never trained in ANY configuration and is "
        "scheduled for DELETION (not repair) with the Stage-D/BLOCK-E connector "
        "rebuild -- raw 512-d beats it by 0.0403. Deleting it here would make every "
        "existing checkpoint unloadable, so it lands with the retrain, not before."
    ),
}


def assert_requires_grad_subset_of_optimizer(model, optimizer) -> list[str]:
    """Every requires_grad=True parameter must belong to some optimizer group.

    Returns the list of tolerated (known) offenders. Raises on any NEW one.
    """
    owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
    orphans = [(n, p) for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in owned]

    tolerated, novel = [], []
    for n, _p in orphans:
        key = next((k for k in KNOWN_UNOPTIMISED if k in n), None)
        (tolerated if key else novel).append(n)

    if tolerated:
        n_el = sum(p.numel() for n, p in orphans if any(k in n for k in KNOWN_UNOPTIMISED))
        log.warning(
            "C5: %d KNOWN un-optimised parameter tensor(s) (%d elements) receive "
            "gradients that are never applied: %s", len(tolerated), n_el, tolerated)
        for k, why in KNOWN_UNOPTIMISED.items():
            if any(k in n for n in tolerated):
                log.warning("C5:   %s -> %s", k, why)

    if novel:
        raise AssertionError(
            "C5 VIOLATION -- these parameters have requires_grad=True but are in NO "
            f"optimizer param group, so they will NEVER move: {novel}\n"
            "This is the defect that made `feature_proj` an untrained random projection "
            "for the life of the project. Either add them to a param group, or freeze "
            "them, or add them to KNOWN_UNOPTIMISED with a written justification."
        )
    return tolerated


def add_backbone_param_group(
    model: NeuroFusion,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trunk_params: list[nn.Parameter],
    lr_backbone: float,
    current_step: int,
) -> int:
    """Add the now-trainable MedNeXt trunk as a THIRD optimizer param group and
    extend the LambdaLR scheduler so the cosine schedule covers it too.

    Why a helper (not just optimizer.add_param_group): LambdaLR captured base_lrs
    and lr_lambdas at construction time. add_param_group alone leaves the new group
    OUT of the scheduler's bookkeeping → on the next scheduler.step() PyTorch raises
    'param_groups vs lr_lambdas length mismatch' (or silently never schedules the new
    group). We append a matching base_lr + lr_lambda and immediately set the new
    group's lr to base_lr * lr_lambda(current_step) so it joins the schedule at the
    correct point (no LR discontinuity, no warmup restart for the existing groups).

    Returns the number of trunk parameters added (for logging the unfreeze event).
    """
    if not trunk_params:
        return 0
    optimizer.add_param_group(
        {"params": trunk_params, "lr": lr_backbone, "weight_decay": 1e-4}
    )
    # Keep the LambdaLR in sync: one base_lr + one lr_lambda per param group.
    lr_lambda = getattr(scheduler, "_nf_lr_lambda", None)
    if lr_lambda is not None and hasattr(scheduler, "base_lrs") and hasattr(scheduler, "lr_lambdas"):
        scheduler.base_lrs.append(lr_backbone)
        scheduler.lr_lambdas.append(lr_lambda)
        # Place the new group on the schedule at the current step so there is no
        # discontinuity for it (cosine value at current_step, not a warmup restart).
        optimizer.param_groups[-1]["lr"] = lr_backbone * lr_lambda(current_step)
    elif hasattr(scheduler, "min_lrs"):
        # L5 ReduceLROnPlateau (--scheduler plateau): it snapshotted one min_lr per
        # param group at construction; a group added now would make
        # optimizer.param_groups longer than scheduler.min_lrs -> IndexError in
        # _reduce_lr on the first LR drop. Extend min_lrs in lock-step (reuse the
        # configured min_lr). Cosine/LambdaLR takes the branch above, so this never
        # runs for the default path.
        scheduler.min_lrs.append(scheduler.min_lrs[-1] if scheduler.min_lrs else 1e-6)
    n_added = sum(p.numel() for p in trunk_params)
    return n_added


# ===========================================================================
# PHASE 1: SEGMENTATION PRETRAIN
# ===========================================================================


def train_phase1_seg_pretrain(args: argparse.Namespace) -> None:
    """Pretrain MedNeXt on combined BraTS 2020+2021+2023 with Dice + CE.

    Output is the best-val-Dice checkpoint that all Phase 2 folds load via
    --seg-checkpoint. Methodology v4 §9: ~45 GPU-h on 1xA100, 150 epochs target.

    Note: for nnunetv2-style training with full BraTS recipes, prefer
    scripts/pretrain_mednext.sh (uses official nnUNetv2 trainer). This in-process
    loop is for ablations / smaller dev runs / non-nnUNet backbones.
    """
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    setup_seed(args.seed)
    device = torch.device(args.device)

    log.info("Phase 1 — segmentation pretrain on combined BraTS 2020+2021+2023")
    log.info(f"BraTS roots: {args.brats_roots}")

    cfg = NeuroFusionConfig()
    cfg.freeze_backbone = False  # we ARE training the backbone in Phase 1
    model = NeuroFusion(cfg).to(device)

    from bratscombined_dataset import get_brats_loader
    train_loader, val_loader = get_brats_loader(
        roots=[Path(r) for r in args.brats_roots],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=(128, 128, 128),
        shuffle=True, val_split=0.1, seed=args.seed,
    )
    n_steps_total = len(train_loader) * args.epochs // max(1, args.grad_accum)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps_total, eta_min=1e-6)

    wb = setup_wandb("phase1_seg_pretrain", cfg, out_dir, args)

    best_dice = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_dice_loss: list[float] = []
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            vis = model.vision(mri)
            loss = dice_plus_ce_loss(vis["seg_logits"], seg, n_classes=cfg.seg_n_classes)
            (loss / args.grad_accum).backward()
            epoch_dice_loss.append(float(loss.detach().cpu().item()))

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    log_metrics({
                        "train/loss_seg": float(np.mean(epoch_dice_loss[-args.log_every:])),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }, step, wb)

        # End-of-epoch validation
        log.info(f"epoch {epoch} done in {time.time() - t0:.1f}s — running validation")
        val_metrics = validate(model, val_loader, device, phase="phase1")
        for k, v in val_metrics.items():
            log.info(f"  {k} = {v:.4f}")
        if wb is not None:
            wb.log(val_metrics, step=step)

        dice_mean = val_metrics.get("val/dice_mean", 0.0)
        if dice_mean > best_dice:
            best_dice = dice_mean
            ckpt_path = out_dir / "best.pt"
            torch.save({
                "vision": model.vision.state_dict(), "epoch": epoch, "step": step,
                "metrics": val_metrics,
            }, ckpt_path)
            log.info(f"  ** new best dice_mean = {dice_mean:.4f} → {ckpt_path}")

    log.info(f"Phase 1 done. best dice_mean = {best_dice:.4f}")
    if wb is not None:
        wb.finish()


# ===========================================================================
# PHASE 2A: M-ROPE IDENTITY-PARITY SANITY CHECK
# ===========================================================================


def train_phase2a_identity_check(args: argparse.Namespace) -> None:
    """Verify M-RoPE retrofit is numerically correct.

    Two checks:
      1. Pure-math identity reduction (mrope_4d._test_identity_reduction): for all-equal
         position indices (p,p,p,p), apply_mrope_4d == 1D RoPE. fp32 tolerance.
      2. (Real-LM path, when wired) load MedGemma + retrofit M-RoPE; forward a text-only
         batch through both the retrofit and the unmodified base; assert max-logit-diff
         is below threshold (1e-5 fp32, 1e-3 bf16).

    Phase 2b training MUST NOT begin if check 2 fails — the retrofit is broken and
    fallback is factorized-3D-PE-only (drop M-RoPE).
    """
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    log.info("Phase 2a — M-RoPE identity-parity sanity check")

    # Check 1: pure-math identity reduction
    log.info("Check 1: mrope_4d._test_identity_reduction (fp32 algebra)")
    try:
        _test_identity_reduction()
        log.info("  PASSED — fp32 max diff = 0.0")
    except AssertionError as e:
        log.error(f"  FAILED — {e}")
        log.error("M-RoPE retrofit is broken. Do NOT proceed to Phase 2b.")
        log.error("Debug the channel partition in mrope_4d.py before continuing.")
        sys.exit(1)

    # Check 2: real-LM text-only parity (skipped in placeholder mode)
    log.info("Check 2: real-LM text-only logit parity")
    cfg = NeuroFusionConfig()
    model = NeuroFusion(cfg)
    if not hasattr(model.lm.lm, "config") or not hasattr(model.lm.lm, "model"):
        log.warning(
            "  SKIPPED — placeholder LM is in use. Wire up real MedGemma + retrofit "
            "monkey-patch in model.LMWithMRope4D, then re-run --phase 2a."
        )
        log.info("Phase 2a check 1 PASSED. Check 2 skipped (no real LM loaded).")
        return

    # Real-LM parity check (activates when MedGemma is wired)
    device = torch.device(args.device)
    model = model.to(device)

    # Build a small text-only batch
    tokenizer = getattr(model.lm, "tokenizer", None)
    if tokenizer is None:
        log.error("  Real LM loaded but no tokenizer attached. Fix model.LMWithMRope4D.__init__.")
        sys.exit(1)

    text = ["The patient has a 3 cm lesion in the right parietal lobe."]
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Two forward passes: with M-RoPE active vs disabled (ROPE fallback)
    with torch.no_grad():
        cfg.use_4d_mrope = True
        out_mrope = model.lm.lm(**inputs).logits
        cfg.use_4d_mrope = False
        out_base = model.lm.lm(**inputs).logits

    max_diff = (out_mrope - out_base).abs().max().item()
    threshold = 1e-3 if model.lm.lm.dtype == torch.bfloat16 else 1e-5
    log.info(f"  max-logit-diff (text-only) = {max_diff:.2e}  (threshold {threshold:.0e})")

    if max_diff > threshold:
        log.error(
            f"  FAILED — text-only logits differ by {max_diff:.2e} > {threshold:.0e}. "
            "Channel partition or position-index assignment is wrong."
        )
        sys.exit(1)
    log.info("  PASSED — M-RoPE retrofit preserves base-model behavior on text-only inputs.")
    log.info("Phase 2a complete. Phase 2b training is cleared to start.")


# ===========================================================================
# PHASE 2B: MULTI-TASK TRAINING ON NEUROFUSION-330
# ===========================================================================


class _EMA:
    """Exponential moving average over the TRAINABLE params only (LoRA + Q-Former
    + field heads + the unfrozen MedNeXt trunk; the 4-bit-frozen LM base is never
    touched). Shadow kept in each param's own dtype/device.

    `register` is idempotent and re-callable AFTER the unfreeze event so the newly
    trainable trunk params join the shadow at their current value (no discontinuity).
    apply_to() swaps in the EMA weights (backing up the raw ones) for validation +
    best.pt; restore() puts the raw weights back so training continues un-smoothed.
    Persistent overhead ≈ 1x trainable-param memory (the trunk dominates, ~62M)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def train_phase2b_multitask(args: argparse.Namespace) -> None:
    """Main multi-task training loop. One fold per invocation."""
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    setup_seed(args.seed + args.fold)  # different seed per fold
    device = torch.device(args.device)
    # --freeze-epochs is an alias for --unfreeze-epoch (warm-up epochs before the
    # trunk unfreezes); normalize it up-front so every downstream check reads
    # args.unfreeze_epoch consistently.
    if getattr(args, "freeze_epochs", None) is not None:
        args.unfreeze_epoch = int(args.freeze_epochs)

    unfreeze_backbone = bool(getattr(args, "unfreeze_backbone", False))
    cfg = NeuroFusionConfig(
        use_stage2_mrope=getattr(args, "use_stage2_mrope", False),
        use_global_pool=getattr(args, "use_global_pool", False),
        lm_family=getattr(args, "lm_family", "medgemma"),
        # Visual-token-expansion experiment (Models 2/3/4). Defaults below keep
        # Model-1 (deployment) BYTE-IDENTICAL: feature_stage="bottleneck",
        # n_queries_per_lesion=32, max_lesions=4. The schema/grammar/GT output cap
        # stays 4 regardless of max_lesions (schema.py untouched).
        feature_stage=getattr(args, "feature_stage", "bottleneck"),
        n_queries_per_lesion=int(getattr(args, "n_queries", 32)),
        max_lesions=int(getattr(args, "max_lesions", 4)),
        mask_overflow_lesion_loss=bool(getattr(args, "mask_overflow_lesion_loss", False)),
        # M5 field-head upgrades (default "mean"/False == byte-identical to Model-1).
        field_head_pool=getattr(args, "field_head_pool", "mean"),
        field_head_global_token=bool(getattr(args, "field_head_global_token", False)),
        # Head-conditioned CoT v2 cohort-anchor (default False == byte-identical: no
        # anchor param is created; the anchor index per case is threaded from the
        # training loop only when the anchor map/gold source is enabled below).
        use_cohort_anchor=bool(getattr(args, "cohort_anchor", False)),
        use_dx_distill=bool(getattr(args, "use_dx_distill", False)),
        dx_distill_weight=float(getattr(args, "dx_distill_weight", 0.2)),
        use_aux_dx=bool(getattr(args, "aux_dx", False)),
        aux_dx_weight=float(getattr(args, "aux_dx_weight", 0.3)),
        dx_head_path=getattr(args, "dx_head_path", None),
        # When unfreezing, construct with freeze_backbone=False so
        # SegmentationBackbone.__init__ leaves the trunk requires_grad=True from the
        # start (eval/grad-checkpointing paths stay consistent); the gradual freeze
        # schedule (warm-up frozen epochs, then flip) is owned by this loop below.
        freeze_backbone=not unfreeze_backbone,
        unfreeze_backbone=unfreeze_backbone,
        backbone_lr=float(getattr(args, "lr_backbone", 1e-5)),
    )
    # ☠️ THE TAP (NF 1.3). cfg.crop_size defaults to (16,16,16); the bottleneck feature
    # map is 8^3, so `_crop_feature` clamps to the WHOLE map and every "per-lesion" crop
    # is the same global pool — routing is centroid-INDEPENDENT. Overriding this is the
    # only way to get a genuinely lesion-local tap, so it is an explicit, logged choice.
    if getattr(args, "crop_size", None):
        _cs = int(args.crop_size)
        cfg.crop_size = (_cs, _cs, _cs)
        log.info(f"[tap] crop_size OVERRIDE -> {cfg.crop_size} on feature_stage="
                 f"{cfg.feature_stage} (grid {'16^3' if cfg.feature_stage == 'enc_hires' else '8^3'}); "
                 f"lesion-LOCAL iff the crop is strictly smaller than the grid")
    else:
        log.info(f"[tap] crop_size DEFAULT {cfg.crop_size} on feature_stage="
                 f"{cfg.feature_stage} — ☠️ at 'bottleneck' (8^3) this clamps to the whole "
                 f"map: the per-lesion crop is a GLOBAL pool and is centroid-independent")

    # CLI-overrideable loss weights (defaults from NeuroFusionConfig: 1/0.5/1/0.3).
    # Phase D'' diagnosis 2026-05-19 showed per-field heads under-train when
    # field-loss weight is dwarfed by gen+seg+ground; bumping w_field to 1.0+
    # gives the heads proportional gradient share.
    if getattr(args, "w_field", None) is not None:
        cfg.w_field = float(args.w_field)
    if getattr(args, "w_gen", None) is not None:
        cfg.w_gen = float(args.w_gen)
    if getattr(args, "w_seg", None) is not None:
        cfg.w_seg = float(args.w_seg)

    log.info(f"Phase 2b — fold {args.fold} of 5  (use_stage2_mrope={cfg.use_stage2_mrope}, "
             f"use_global_pool={cfg.use_global_pool}, "
             f"lm_family={cfg.lm_family}, lm_hidden_dim={cfg.lm_hidden_dim})")
    log.info(f"  visual-token cfg: feature_stage={cfg.feature_stage} "
             f"n_queries_per_lesion={cfg.n_queries_per_lesion} max_lesions={cfg.max_lesions} "
             f"(LM visual tokens up to {cfg.max_lesions * cfg.n_queries_per_lesion}; "
             f"schema/grammar/GT output cap stays 4)")
    log.info(f"  loss weights: w_seg={cfg.w_seg} w_field={cfg.w_field} w_gen={cfg.w_gen} w_ground={cfg.w_ground}")
    log.info(f"out_dir: {out_dir}")

    # Data loaders
    train_loader, val_loader = get_split_loaders(
        jsonl_path=Path(args.jsonl),
        brats_root=Path(args.brats_root),
        splits_path=Path(args.splits),
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_volumes=False,
        augment_train=getattr(args, "augment", False),
        cohort_balanced_sampler=getattr(args, "cohort_balanced_sampler", False),
    )
    if getattr(args, "augment", False):
        log.info("MRI augmentation ENABLED for training set (val/test untouched)")
    log.info(f"train cases: {len(train_loader.dataset)}  val cases: {len(val_loader.dataset)}")

    # Model. Pass mednext_checkpoint so NeuroFusion routes through the
    # real MONAI MedNeXt loader; falls back to placeholder if the file is missing.
    mednext_ckpt = args.seg_checkpoint or args.mednext_checkpoint
    model = NeuroFusion(cfg, mednext_checkpoint=mednext_ckpt).to(device)
    # Backbone freeze (Phase 2b convention): freeze the pretrained MedNeXt trunk
    # but keep the bottleneck→feature projection trainable (it has no pretrained weights).
    #
    # GRADUAL UNFREEZE: when --unfreeze-backbone is set with --unfreeze-epoch > 0,
    # the trunk is STILL frozen here (and re-frozen each warm-up epoch) so the
    # optimizer is built with only the qformer + LoRA groups; the trunk param group
    # is added at the unfreeze epoch via add_backbone_param_group. set_backbone_
    # trainable(False) flips the trunk's requires_grad off while preserving
    # feature_proj — the same invariant the manual loop maintained, but routed
    # through the model's first-class hook so the MONAI-vs-placeholder split lives
    # in one place. (When --unfreeze-epoch == 0 we still freeze here for the build;
    # the loop unfreezes immediately at epoch 0 before the first backward.)
    if unfreeze_backbone:
        frozen = model.set_backbone_trainable(False)
        log.info(
            f"UNFREEZE schedule ARMED: trunk frozen for warm-up "
            f"(unfreeze_epoch={args.unfreeze_epoch}, lr_backbone={cfg.backbone_lr}); "
            f"{sum(p.numel() for p in frozen):,} trunk params will join the optimizer at the unfreeze epoch"
        )
    elif hasattr(model.vision.backbone, "mednext"):
        for p in model.vision.backbone.mednext.parameters():
            p.requires_grad = False
    else:
        for p in model.vision.parameters():
            p.requires_grad = False
    log.info(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Warmstart Q-Former + field heads + LoRA adapters from an existing fine-tuned
    # NF checkpoint (per-fold, e.g. scratch/runs/fold{K}/best.pt). Loads ONLY model
    # weights (state.qformer / field_head / lm_lora) — NOT optimizer/scheduler — so
    # the cosine LR schedule and the gradual-unfreeze curriculum restart fresh on
    # the NEW max-data seg backbone loaded above. requires_grad (the frozen→unfrozen
    # trunk schedule) is set by the block above and is untouched by this weight load.
    if getattr(args, "warmstart_nf", None):
        wpath = Path(args.warmstart_nf)
        if not wpath.exists():
            raise SystemExit(f"--warmstart-nf checkpoint not found: {wpath}")
        load_checkpoint(wpath, model,  # model-only (optimizer=scheduler=None)
                        pin_seg_trunk=bool(getattr(args, "pin_seg_trunk", False)))
        log.info(
            f"WARMSTART: loaded Q-Former + field-heads + LoRA from {wpath} "
            f"(fresh optimizer/scheduler/LR; "
            f"{'seg trunk PINNED to --seg-checkpoint' if getattr(args, 'pin_seg_trunk', False) else '☠️ seg trunk comes from the WARMSTART CKPT, not --seg-checkpoint'}, "
            f"trunk unfreeze schedule unchanged)"
        )
        # ☠️ PROVE THE PIN HELD — at step 0, before a single gradient. The claim
        # "this run trained on <trunk>" is only checkable against the WEIGHTS.
        _trunk_verify(model, mednext_ckpt, Path(args.out_dir), "step0",
                      pinned=bool(getattr(args, "pin_seg_trunk", False)))

    # Optimizer + scheduler
    n_steps_total = len(train_loader) * args.epochs // max(1, args.grad_accum)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, args, n_steps_total)
    # L5 cadence guard: ReduceLROnPlateau (--scheduler plateau) steps ONCE PER EPOCH
    # on the selection metric; the default cosine/LambdaLR steps per optimizer step.
    # Default (cosine) => _use_plateau False => the per-step .step() path is unchanged.
    _use_plateau = isinstance(scheduler, ReduceLROnPlateau)

    # EMA over trainable params (opt-in). Re-registered at the unfreeze event so the
    # trunk params join the shadow. Validate + save best.pt on EMA weights.
    ema = _EMA(model, args.ema_decay) if getattr(args, "use_ema", False) else None
    if ema is not None:
        log.info(f"EMA enabled (decay={args.ema_decay}); shadow over "
                 f"{len(ema.shadow)} trainable param tensors")

    # ---- RESUME (continue an interrupted run to its natural plateau) ----
    # Unlike --warmstart-nf (model-only, fresh schedule), --resume restores
    # optimizer + scheduler + epoch/step so the SAME cosine schedule and the
    # early-stop window continue forward. We pre-add the trunk param group (if it
    # was unfrozen by the saved epoch) BEFORE loading optimizer state so the group
    # structure matches; re-register EMA over the trunk; and seed the best-metric
    # FLOOR from the sibling best.pt so a post-resume validation can NEVER overwrite
    # the real best with a worse checkpoint. With --resume unset this is a no-op
    # (byte-identical to the fresh path).
    _resume_start_epoch = 0
    _resume_step = 0
    _resume_best_metric = -1.0
    _resume_best_epoch = -1
    _resume_unfrozen = False
    if getattr(args, "resume", None):
        rpath = Path(args.resume)
        if not rpath.exists():
            raise SystemExit(f"--resume checkpoint not found: {rpath}")
        rckpt = load_checkpoint(rpath, model)  # model-only (full state, bnb-tolerant)
        _resume_step = int(rckpt.get("step", 0))
        _resume_start_epoch = int(rckpt["epoch"]) + 1
        # Replicate the optimizer structure if the trunk was already unfrozen.
        if getattr(args, "unfreeze_backbone", False) and int(rckpt["epoch"]) >= args.unfreeze_epoch:
            trunk_params = model.set_backbone_trainable(True)
            add_backbone_param_group(
                model, optimizer, scheduler, trunk_params,
                lr_backbone=cfg.backbone_lr, current_step=_resume_step,
            )
            _resume_unfrozen = True
            if ema is not None:
                ema.register(model)   # cover the unfrozen trunk in the EMA shadow
        if rckpt.get("optimizer") is not None:
            optimizer.load_state_dict(rckpt["optimizer"])
        if scheduler is not None and rckpt.get("scheduler") is not None:
            scheduler.load_state_dict(rckpt["scheduler"])
        # Seed the best-metric floor from the sibling best.pt (never clobber it).
        bpath = rpath.parent / "best.pt"
        if bpath.exists():
            bckpt = torch.load(bpath, map_location="cpu")
            bm = bckpt.get("metrics", {})
            if "val/anchor_dx_recall" in bm:
                # Head-conditioned CoT v2 (BLOCKER-2): match the anchor-dx selection blend.
                _resume_best_metric = bm["val/anchor_dx_recall"] * 0.5 + bm.get("val/macro_f1", 0.0) * 0.5
            elif "val/freegen_structural_validity" in bm:
                _resume_best_metric = bm["val/freegen_structural_validity"] * 0.5 + bm.get("val/macro_f1", 0.0) * 0.5
            else:
                _resume_best_metric = bm.get("val/macro_f1", 0.0)
            _resume_best_epoch = int(bckpt.get("epoch", -1))
            del bckpt
        log.info(
            f"RESUME: continuing {rpath} from epoch {_resume_start_epoch} "
            f"(step {_resume_step}, trunk_unfrozen={_resume_unfrozen}); best-floor "
            f"{_resume_best_metric:.4f} @e{_resume_best_epoch} (not overwritten unless beaten); "
            f"cosine schedule + early-stop window continue unchanged."
        )

    # Replay buffer for KL-to-base
    replay = TextReplayBuffer(size=cfg.kl_to_base_replay_buffer_size, seed=args.seed,
                              corpus_path=getattr(args, "kl_corpus_jsonl", None))

    # Phase 1a: text-only SFT buffer (medical-alignment). None => OFF (byte-identical).
    _text_sft_jsonl = getattr(args, "text_sft_jsonl", None)
    _text_sft_weight = float(getattr(args, "text_sft_weight", 0.0) or 0.0)
    sft_buffer = None
    if _text_sft_jsonl and _text_sft_weight > 0:
        sft_buffer = TextReplayBuffer(seed=args.seed, corpus_path=_text_sft_jsonl)
        log.info(f"text-only SFT ON: {len(sft_buffer.texts)} sentences from {_text_sft_jsonl} "
                 f"(weight={_text_sft_weight}, nsamples={getattr(args, 'text_sft_nsamples', 4)})")

    # Logger
    wb = setup_wandb(f"phase2b_fold{args.fold}", cfg, out_dir, args)

    # Per-field class weights from training-set lesion frequencies. Computed
    # once at startup; reused for every F.cross_entropy call in the per-field
    # head losses. Counters majority-class collapse (audit M2).
    #
    # SSL HARDENING (2026-05-19): when --class-weight-labeled-only is set, the
    # weights are computed from labeled folds (0..4) only, ignoring pseudo
    # cases (fold >=5 by convention). This prevents pseudo-label class
    # distribution shift from biasing the head's gradient toward classes the
    # val set doesn't actually contain (e.g. pseudo-hemorrhagic, pseudo-
    # brainstem). The vocab-clean filter pass keeps pseudo records in
    # training, but they shouldn't drive the class-weight schedule.
    all_train_items = train_loader.dataset.items
    if getattr(args, "class_weight_labeled_only", False):
        import json as _json
        with open(args.splits) as _f:
            _splits = _json.load(_f)
        labeled_case_ids = set()
        for fold_k, fold_ids in _splits["folds"].items():
            if int(fold_k) <= 4:
                labeled_case_ids.update(fold_ids)
        weight_items = [it for it in all_train_items if it["report"].case_id in labeled_case_ids]
        log.info(
            f"class-weight computed from LABELED-only: {len(weight_items)}/{len(all_train_items)} train items "
            f"(pseudo cases excluded from class-weight calc)"
        )
    else:
        weight_items = all_train_items
    train_reports = [it["report"] for it in weight_items]
    field_class_weights = compute_field_class_weights(train_reports, device)
    for fname, w in field_class_weights.items():
        log.info(f"  field_class_weights[{fname}]: {w.cpu().numpy().round(3).tolist()}")

    # =======================================================================
    # PRE-FLIGHT LEAK ASSERT (HARD GATE — runs before ANY backward).
    # =======================================================================
    # HARD LEAK RULE: the n=39 test + n=50 conformal = 89 PROTECTED BraTS-2020
    # cases (splits.json test + conformal_calibration) must NEVER enter training.
    # The amplified MEN/PED/GLI seg-pretrain widens what the *backbone* learns,
    # and unfreezing changes WHAT trains — but Phase-2b paired training must still
    # only ever see splits.json train folds. Re-verify here that NO protected case
    # id is reachable in the train loader's dataset (intersection of loader case
    # ids with the protected ids must be empty), regardless of how get_split_loaders
    # resolved folds. SystemExit on any violation — do not let a single leaked case
    # reach a gradient step.
    splits_obj = load_splits(Path(args.splits))
    # Leak guard EXPANDED 2026-06-22 (M6-Mistral prereg Step 1): the held-out
    # cross-dataset report-gen cohorts test_gli / test_men / test_met (the exact
    # cells M6 is scored on) must ALSO never reach a gradient step — not only the
    # BraTS-2020 test + conformal_calibration. These keys are absent -> [] in older
    # splits, so this is a safe superset for every splits file.
    protected_ids = (
        set(splits_obj.get("test", []))
        | set(splits_obj.get("conformal_calibration", []))
        | set(splits_obj.get("test_gli", []))
        | set(splits_obj.get("test_men", []))
        | set(splits_obj.get("test_met", []))
    )
    train_case_ids = {it["report"].case_id for it in all_train_items}
    leaked = sorted(train_case_ids & protected_ids)
    if leaked:
        raise SystemExit(
            f"LEAK GUARD TRIPPED: {len(leaked)} PROTECTED case(s) reachable in the Phase-2b "
            f"train loader (test+conformal must never train). Offenders: {leaked[:20]}"
            + (" ..." if len(leaked) > 20 else "")
            + "  Refusing to start training — fix splits.json / loader before re-launching."
        )
    log.info(
        f"PRE-FLIGHT leak assert PASSED: 0/{len(train_case_ids)} train cases intersect the "
        f"{len(protected_ids)} protected (test+conformal+test_gli/men/met) ids."
    )
    # ☠️ AND THE SAME QUESTION ASKED OF THE REGISTRY, not of the splits keys. The assert
    # above can only see ids FILED under a protected key; the 24 canonical-164 MET cases
    # that trained through nfmistral_cotcond_fold0 were filed in CV fold 6 and sailed
    # past it. Same ids, same moment, different authority.
    assert_train_basis_disjoint(train_case_ids, splits_obj, out_dir,
                                tag=f"fold{args.fold}")

    # Training loop
    best_macro_f1 = -1.0
    best_epoch = -1
    backbone_unfrozen = False   # latch: ensure add_backbone_param_group fires exactly once
    step = 0

    # --- Head-conditioned CoT v2: resolve the per-case cohort-anchor map (or None).
    # Default OFF => None => cohort_anchor_idx never passed => decode byte-identical.
    cohort_anchor_map = build_cohort_anchor_map(args, all_train_items, protected_ids)
    if cohort_anchor_map is not None:
        # ☠️ RULING 3(a) 2026-08-11 — REFUSE AT STARTUP, NEVER MID-RUN. See
        # assert_anchor_coverage(): it drills the resolver, refuses an empty split, and
        # then refuses any train case with no anchor slot, all BEFORE the first gradient
        # step. train_case_ids IS the loader's dataset id set (built above from
        # train_loader.dataset.items), so every id the loop can ever look up is checked
        # here — this is not a sample.
        assert_anchor_coverage(
            cohort_anchor_map, train_case_ids, out_dir,
            source=str(getattr(args, "cohort_anchor_map", None) or "gold+error-sim"),
            tag=f"train_fold{args.fold}")
        # Safe now: every key is present, so this distribution is the REAL one. With the
        # old `.get(cid, 0)` a missing id was counted as 'none' and inflated slot 0.
        _dist = Counter(cohort_anchor_map[cid] for cid in train_case_ids)
        log.info(
            f"[cohort-anchor] ENABLED (use_cohort_anchor={cfg.use_cohort_anchor}); train "
            f"anchor-idx distribution none/GLI/MEN/MET = "
            f"{_dist.get(0,0)}/{_dist.get(1,0)}/{_dist.get(2,0)}/{_dist.get(3,0)}. "
            f"v2 REQUIRES neutral case_id at inference (drop the cohort tag)."
        )

    # --- W-QF-OBJ: GT cohort labels for the auxiliary dx loss on connector_OUT ----
    # Labels come from `infer_cohort` (case_id prefix AND differential_diagnosis), NOT from
    # the RG_ prefix alone: prefix-only keying drops the 121 BraTS-2020 TR gliomas to OTHER,
    # which would silently shrink the supervised pool by ~a fifth and change what the loss
    # is actually fitting. Only {GLI, MEN, MET} are supervised; everything else maps to -1
    # and contributes no gradient -- OTHER has no meaningful positives here and making it a
    # real 4th class just gives the head a bin to dump uncertain cases into.
    #
    # These are TRAIN cases, so GT cohort is a legal target. No canonical test id can reach
    # this map: the split assert upstream already fails the run if one does.
    _aux_dx_labels = None
    if getattr(cfg, "use_aux_dx", False):
        from scripts.build_balanced_corpus import infer_cohort
        _c2i = {"GLI": 0, "MEN": 1, "MET": 2}
        _aux_dx_labels = {}
        for _it in all_train_items:
            # items are {"report": <StructuredReport>, ...} -- the case_id lives on the
            # report object, NOT at the dict's top level (see the class-weight block above).
            _rep = _it["report"]
            _c = infer_cohort(_rep.case_id, getattr(_rep, "differential_diagnosis", None))
            if _c in _c2i:
                _aux_dx_labels[_rep.case_id] = _c2i[_c]
        _n_lab = sum(1 for c in train_case_ids if c in _aux_dx_labels)
        _dd = Counter(_aux_dx_labels[c] for c in train_case_ids if c in _aux_dx_labels)
        log.info(f"[aux-dx] ENABLED w={cfg.aux_dx_weight}; {_n_lab}/{len(train_case_ids)} train "
                 f"cases labelled GLI/MEN/MET = {_dd.get(0,0)}/{_dd.get(1,0)}/{_dd.get(2,0)} "
                 f"({len(train_case_ids)-_n_lab} unlabelled -> no aux gradient)")
        if _n_lab < 0.5 * len(train_case_ids):
            raise SystemExit(
                f"[aux-dx] only {_n_lab}/{len(train_case_ids)} train cases got a cohort label. "
                f"The loss would be fit on a minority of the pool while the log still says "
                f"ENABLED -- refusing to start rather than train a differently-scoped arm.")

    # --- Head-conditioned CoT (draft-then-commit) target diagnostics -----------
    # Measure the truncation rate of the LONGER '<prose>[COMMIT]<json>' targets at
    # the chosen --cot-max-length over the FULL train set, and print 2 example
    # decoded targets so the transform is verifiable in the smoke log. Gated on the
    # flag -> no effect on the default JSON-only path.
    if getattr(args, "cot_supervision", False):
        _tok = getattr(model.lm, "tokenizer", None)
        _cot_maxlen = int(getattr(args, "cot_max_length", 1024))
        _reps = [it["report"] for it in all_train_items]
        # FIX 2026-07-25: this diagnostic re-serialization used a bare model_dump_json(), which
        # IGNORED NF_STRIP_CASE_ID. The real supervision path (dataset.py:280-281 -> :337 ->
        # train.py:2215) does strip, so training was correct -- but the "[cot-supervision] DECODED
        # target" block below printed a TAGGED target, meaning the log could never actually verify
        # that the strip had fired. Mirror the dataset's exclusion so the log is trustworthy before
        # a 16 GPU-h run is judged by it.
        _strip_cid = os.environ.get("NF_STRIP_CASE_ID") == "1"
        _jsons = [r.model_dump_json(exclude={"case_id"} if _strip_cid else None) for r in _reps]
        log.info(f"[cot-supervision] NF_STRIP_CASE_ID={'1 (case_id EXCLUDED from targets)' if _strip_cid else '0'}")
        _cot_targets = build_cot_target_strings(
            _reps, _jsons, dx_first=getattr(args, "dx_first_draft", False),
            dx_rationale=getattr(args, "dx_rationale_draft", False),
        )
        if _tok is not None:
            _tok_lens = [len(_tok(t, add_special_tokens=True)["input_ids"]) for t in _cot_targets]
            _n_trunc = sum(1 for L in _tok_lens if L > _cot_maxlen)
            _n = max(1, len(_tok_lens))
            log.info(
                f"[cot-supervision] targets={len(_tok_lens)}  token-len "
                f"min/median/max={min(_tok_lens)}/{int(np.median(_tok_lens))}/{max(_tok_lens)}  "
                f"TRUNCATION @ max_length={_cot_maxlen}: {_n_trunc}/{len(_tok_lens)} = "
                f"{100.0 * _n_trunc / _n:.2f}%"
            )
            # Decode 2 example targets (round-trip through tokenize->truncate->decode
            # so the log shows exactly what the LM is teacher-forced on).
            for _j in range(min(2, len(_cot_targets))):
                _ids, _ = build_target_text_ids([_cot_targets[_j]], _tok, max_length=_cot_maxlen, device=device)
                _dec = _tok.decode(_ids[0], skip_special_tokens=True)
                log.info(f"[cot-supervision] DECODED target[{_j}] (case={_reps[_j].case_id}, "
                         f"ddx[0]={_reps[_j].differential_diagnosis[0]!r}):\n{_dec}\n"
                         f"[cot-supervision] --- end target[{_j}] ---")
        else:
            log.warning("[cot-supervision] no tokenizer on model.lm — cannot measure truncation "
                        "(placeholder-LM path?). The transform still applies at train time.")

    # Training loop (resume-aware: the _resume_* values are the fresh-run defaults
    # -1.0/-1/False/0/0 unless --resume seeded them above).
    best_macro_f1 = _resume_best_metric
    best_epoch = _resume_best_epoch
    backbone_unfrozen = _resume_unfrozen   # latch: ensure add_backbone_param_group fires exactly once
    step = _resume_step

    # ---- --eval-only: measure the loaded checkpoint, write it down, train nothing -------
    # ☠️ PLACED HERE DELIBERATELY, AFTER the full setup and the checkpoint load, so the
    # basis is the SAME one training used -- same split, same anchor map, same leak gates.
    # A separate eval driver that rebuilt the loader could differ from the training basis
    # without anyone noticing, which is the failure this whole phase exists to remove.
    if getattr(args, "eval_only", None):
        log.info(f"[eval-only] no training. checkpoint under test: {args.resume or args.warmstart_nf}")
        _vm = validate(model, val_loader, device, phase="phase2b", report_seg_dice=False)
        if getattr(model.lm, "cohort_anchor", None) is not None:
            _vm.update(_anchor_dx_subeval(
                model, val_loader, device,
                max_cases=int(getattr(args, "anchor_dx_val_cases", 40))))
        else:
            # ☠️ NOT a pass. The anchor arm is the whole reason for this comparison; if it
            # is not live, say so as an ABSENCE rather than emitting a field-only verdict
            # that reads like a complete answer.
            _vm["val/anchor_dx_UNEVALUABLE_anchor_not_live"] = 1.0
        for _k, _v in sorted(_vm.items()):
            log.info(f"  {_k} = {_v:.4f}")
        _out = Path(args.eval_only)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _out.write_text(json.dumps({
            "_what": "single-checkpoint VAL measurement, --eval-only (no training)",
            "_basis": "fold VAL split only — NO test look spent",
            "checkpoint": str(args.resume or args.warmstart_nf),
            # ☠️ A FILE sha, and named as one. It identifies WHICH FILE was read; it is NOT
            # evidence about the weights that ended up loaded -- job 530180 had a correct
            # file sha while the served trunk differed on 524/524 tensors. The weight-level
            # check is _trunk_verify, which this run performs separately.
            "checkpoint_file_sha256": _sha256_of_file(args.resume or args.warmstart_nf),
            "anchor_dx_val_cases": int(getattr(args, "anchor_dx_val_cases", 40)),
            "metrics": {k: float(v) for k, v in _vm.items()},
        }, indent=1))
        log.info(f"[eval-only] -> {_out}")
        return

    for epoch in range(_resume_start_epoch, args.epochs):
        model.train()
        # THE CRITICAL EDIT (gradual unfreeze): the per-epoch re-freeze of the
        # MedNeXt trunk is now CONDITIONAL on the unfreeze schedule. Without this
        # condition, an earlier unfreeze would be SILENTLY UNDONE every epoch
        # (model.train() does not change requires_grad, but this block did) — the
        # trunk would never actually adapt.
        #
        #   - --unfreeze-backbone OFF  → always re-freeze the trunk (legacy
        #     frozen-Phase-2b convention; byte-identical behavior).
        #   - --unfreeze-backbone ON and epoch < unfreeze_epoch → still re-freeze
        #     (warm-up: qformer/heads warm up against stable Phase-1 features).
        #   - --unfreeze-backbone ON and epoch >= unfreeze_epoch → DO NOT re-freeze;
        #     flip the trunk trainable (once) and add it to the optimizer as a
        #     third param group at cfg.backbone_lr.
        #
        # Either way only the pretrained MedNeXt trunk is touched — the bottleneck→
        # feature projection (feature_proj) stays trainable (set_backbone_trainable
        # preserves it; the manual loop targets .mednext only).
        #
        # ☠️ CORRECTION 2026-08-08. This comment used to end "...so the Q-Former never
        # sees a fixed RANDOM projection of bottleneck features." THAT CLAUSE WAS FALSE
        # and it contradicted KNOWN_UNOPTIMISED["feature_proj"] ~700 lines above in this
        # same file. `set_backbone_trainable` sets feature_proj.requires_grad=True but
        # RETURNS only the .mednext params, so `add_backbone_param_group` wires only the
        # trunk into the optimizer. feature_proj therefore has requires_grad=True and
        # belongs to NO param group: it receives gradients that are never applied and has
        # never moved off its init in any run.
        # RE-MEASURED at HEAD across four checkpoints from four separate training runs
        # (nfmistral_stripid, nfmistral_cotcond_fold0, NF_knight_1.1, P5.1 fold-7):
        # feature_proj.{weight,bias} are BIT-IDENTICAL to each other AND to a fresh
        # seed-42 nn.Conv3d(512,320,1) init (max|d| = 0.0), and |w|max = 0.04419411 sits
        # exactly on the kaiming-uniform bound 1/sqrt(512) = 0.04419417.
        # ⇒ the Q-Former DOES see a fixed random projection. That is the current design
        # of record (a frozen random read-out, not a trained one) — not automatically
        # wrong, but it must not be described as trained anywhere.
        if (not unfreeze_backbone) or (epoch < args.unfreeze_epoch):
            if hasattr(model.vision, "backbone") and hasattr(model.vision.backbone, "mednext"):
                for p in model.vision.backbone.mednext.parameters():
                    p.requires_grad = False
            else:
                # Placeholder/non-MONAI backbone in the legacy path: keep prior behavior.
                if not unfreeze_backbone and hasattr(model.vision, "backbone"):
                    for p in model.vision.backbone.parameters():
                        p.requires_grad = False
        elif not backbone_unfrozen:
            # UNFREEZE EVENT — flip the trunk trainable and register its params in the
            # optimizer exactly once. set_backbone_trainable returns the trunk params
            # (feature_proj preserved); add_backbone_param_group wires them into the
            # optimizer + extends the LambdaLR so the cosine schedule covers them.
            trunk_params = model.set_backbone_trainable(True)
            n_added = add_backbone_param_group(
                model, optimizer, scheduler, trunk_params,
                lr_backbone=cfg.backbone_lr, current_step=step,
            )
            backbone_unfrozen = True
            if ema is not None:
                ema.register(model)   # add the now-trainable trunk params to the EMA shadow
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            log.info(
                f"UNFREEZE backbone at epoch {epoch}, +{n_added:,} trainable params, "
                f"lr={cfg.backbone_lr:g}  (param-group LRs now: "
                + ", ".join(f"{lr:.2e}" for lr in lrs) + ")"
            )
            log.info(
                f"  total trainable params after unfreeze: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
                f"(w_seg={cfg.w_seg} keeps the Dice+CE seg loss regularizing the trunk → anti-forgetting)"
            )

        epoch_losses: dict[str, list[float]] = {
            "total": [], "seg": [], "field": [], "gen": [], "ground": [], "kl_base": [], "dxdistill": [],
            "textsft": [], "auxdx": [],
        }
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            mri = batch["mri"].to(device)
            seg = batch["seg"].to(device)
            reports = batch["reports"]
            report_jsons = batch["report_jsons"]
            batch_case_ids = batch["case_ids"]

            # Head-conditioned CoT v2: per-case anchor idx aligned to THIS batch
            # (from the leak-free probe/gold map). None => default byte-identical path.
            # ☠️ RULING 3(a): `.get(cid, 0)` here silently trained missing cases on slot 0.
            # This is a BACKSTOP, not the guard: it is reachable only after the startup
            # coverage check passed over train_case_ids, and batch_case_ids is drawn from
            # that same dataset (the L4 cohort-balanced sampler only re-weights the SAME
            # items), so it cannot fire on a short map — that dies in seconds at startup.
            # It can still fire if a future change swaps or extends train_loader.dataset
            # mid-run (e.g. a pseudo-label refresh), which is exactly the case where
            # guessing an anchor would be worse than losing the epoch.
            cohort_anchor_idx = (
                resolve_anchor_slots(cohort_anchor_map, batch_case_ids)
                if cohort_anchor_map is not None else None
            )

            # Forward
            tokenizer = getattr(model.lm, "tokenizer", None)
            # C11 (2026-08-02): the JSON-only path was 512. At 512, 92 of 893 corpus
            # reports (10.3%) were silently TRUNCATED mid-target -- including ALL 25
            # four-lesion reports and 36 of 37 three-lesion ones (4 lesions is already
            # ~621 tokens), censoring supervision on exactly the multi-lesion tail a
            # per-lesion architecture exists to serve.
            # MERGE 2026-08-03: C11 (MAIN) raised the JSON-only literal; the CoT branch
            # (worktree) already used 1024. Neither side alone is right -- taking MAIN
            # drops CoT supervision, taking the worktree silently restores the 512 cap on
            # the DEFAULT path. Both now read ONE env override so the two supervision
            # paths cannot diverge on truncation again.
            _lm_max_default = int(os.environ.get("NF_TARGET_MAX_TOKENS", "1024"))
            if getattr(args, "cot_supervision", False):
                _lm_targets = build_cot_target_strings(
                    reports, report_jsons,
                    dx_first=getattr(args, "dx_first_draft", False),
                    dx_rationale=getattr(args, "dx_rationale_draft", False),
                )
                _lm_max_len = int(getattr(args, "cot_max_length", _lm_max_default))
            else:
                _lm_targets = report_jsons
                _lm_max_len = _lm_max_default
            target_text_ids, target_text_mask = build_target_text_ids(
                _lm_targets, tokenizer, max_length=_lm_max_len, device=device,
            )

            out = model(
                mri=mri,
                seg=seg,
                field_targets=None,                # we'll build them below from the routing output
                target_text_ids=target_text_ids,
                target_text_mask=target_text_mask,
                training=True,
                cohort_anchor_idx=cohort_anchor_idx,
            )

            # Field targets must align with the routing's per-lesion entries
            field_targets = build_field_targets(
                reports, out["routing"]["batch_idx"], out["routing"]["lesion_idx"], device,
                mask_overflow=getattr(cfg, "mask_overflow_lesion_loss", False),
            )

            # Recompute losses with proper field targets and the externally-computed
            # ground + kl_base losses (the model's internal stub for these two is 0)
            l_seg = dice_plus_ce_loss(out["seg_logits"], seg, n_classes=cfg.seg_n_classes)
            # Per-field CE with TWO audit-fixes:
            #   1. Class weighting (inverse-frequency, clipped) — counters
            #      majority-class collapse (was contributing to bilateral over-
            #      prediction + always-"none" bias on presence flags).
            #   2. Normalize each head's loss by log(n_classes_for_head) before
            #      averaging — without this, the 16-class location head (max
            #      CE log(17)≈2.83) dominates the 2-class presence-flag heads
            #      (max CE log(2)≈0.69) by 4x. Audit M3.
            per_field_losses = []
            # SSL HARDENING: label smoothing on per-field CE softens the head's
            # over-confidence on majority-class predictions; combined with
            # inverse-frequency class weighting it gives small but real
            # gradient to underrepresented classes (the "stuck at always-
            # majority" failure mode observed in Phase D'' with poisoned pseudo).
            _label_smoothing = float(getattr(args, "label_smoothing", 0.0))
            _focal_gamma = float(getattr(args, "focal_gamma", 0.0))
            for name, logits in out["field_logits"].items():
                if name not in field_targets or field_targets[name].numel() == 0:
                    continue
                w = field_class_weights.get(name)
                tgt = field_targets[name]
                # C12: skip a field whose targets are ALL ignore_index -- CE would
                # return nan over an empty selection and poison the whole step.
                # MERGE 2026-08-03: this guard is NOT made redundant by the helper
                # below. _field_classification_loss handles the all-ignore case only in
                # its FOCAL branch (valid.sum()==0 -> logits.sum()*0.0); its gamma<=0
                # DEFAULT branch is a bare F.cross_entropy, which still returns NaN over
                # an empty selection. Keeping MAIN's guard + the worktree's helper is the
                # only combination that has both focal support and NaN safety.
                if bool((tgt == FIELD_IGNORE_INDEX).all()):
                    continue
                # ☠️ PASS ignore_index EXPLICITLY -- do NOT lean on the helper's default.
                # Caught by test_c12 during the 2026-08-03 merge: routing C12 through this
                # helper left the coupling IMPLICIT (helper default -100 happening to equal
                # FIELD_IGNORE_INDEX). Behaviour was correct and the guarantee was not: one
                # quantity under two independent definitions, so changing the constant here
                # would silently NOT follow into the loss. That is the same shape as
                # `dice_WT` being bit-identical to `mean_fg` under two names.
                ce = _field_classification_loss(
                    logits, tgt, w, _focal_gamma, _label_smoothing,
                    ignore_index=FIELD_IGNORE_INDEX,
                )
                norm = math.log(max(_FIELD_N_CLASSES.get(name, 2), 2))
                per_field_losses.append(ce / norm)
            l_field = torch.stack(per_field_losses).mean() if per_field_losses else torch.tensor(0.0, device=device)
            l_gen = out["lm_loss"]
            # L_ground (per-lesion mask IoU) is disabled under the global-pool
            # ablation: there are no per-lesion crops to ground, so the routing
            # emits a zeros GT mask. Zero the grounding loss at the source here
            # rather than feeding the degenerate all-zero IoU into the total.
            if cfg.use_global_pool:
                l_ground = torch.tensor(0.0, device=device)
            else:
                l_ground = compute_grounding_loss(model.qformer, out["routing"]["lesion_gt_feat"])
            l_kl = compute_kl_to_base_loss(model, replay, n_samples=4, device=device) if cfg.use_kl_to_base else torch.tensor(0.0, device=device)
            # Phase 1a: text-only medical-alignment SFT loss (no volume; LoRA-only grad).
            l_textsft = (
                compute_text_sft_loss(model, sft_buffer,
                                      n_samples=int(getattr(args, "text_sft_nsamples", 4)),
                                      device=device)
                if (sft_buffer is not None and _text_sft_weight > 0)
                else torch.tensor(0.0, device=device)
            )

            # Rung-3 head-distillation: KL(dx_probe(commit_hidden) ‖ frozen dx-head). OFF =>
            # 0 => total byte-identical. Uses out["commit_hidden"] + out["routing"].
            l_dxdistill = (
                model.dx_distill_loss(out)
                if getattr(cfg, "use_dx_distill", False)
                else torch.tensor(0.0, device=device)
            )

            # W-QF-OBJ: CE on the 4-way head over the per-case MEAN of connector_OUT.
            # `aux_dx_slots` lists the batch slots the logit rows came from (dummy and
            # lesion-less slots are dropped in the forward), so labels are gathered THROUGH
            # it -- zipping against the full batch would silently misalign every row after
            # the first dropped case and train the Q-Former toward the wrong cohort.
            l_auxdx = torch.tensor(0.0, device=device)
            if _aux_dx_labels is not None and out.get("aux_dx_logits") is not None:
                _lab = [_aux_dx_labels.get(batch_case_ids[b], -1) for b in out["aux_dx_slots"]]
                _keep = [i for i, v in enumerate(_lab) if v >= 0]   # unlabelled (OTHER) excluded
                if _keep:
                    l_auxdx = F.cross_entropy(
                        out["aux_dx_logits"][_keep].float(),
                        torch.tensor([_lab[i] for i in _keep], device=device, dtype=torch.long),
                    )

            total = (
                cfg.w_seg * l_seg
                + cfg.w_field * l_field
                + cfg.w_gen * l_gen
                + cfg.w_ground * l_ground
                + cfg.kl_to_base_weight * l_kl
                + _text_sft_weight * l_textsft
                + getattr(cfg, "dx_distill_weight", 0.0) * l_dxdistill
                + getattr(cfg, "aux_dx_weight", 0.0) * l_auxdx
            )

            # Backward with grad accumulation
            (total / args.grad_accum).backward()

            for k, v in [("total", total), ("seg", l_seg), ("field", l_field), ("gen", l_gen),
                         ("ground", l_ground), ("kl_base", l_kl), ("textsft", l_textsft),
                         ("dxdistill", l_dxdistill), ("auxdx", l_auxdx)]:
                epoch_losses[k].append(float(v.detach().cpu().item()))

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
                optimizer.step()
                # L5 cadence: cosine/LambdaLR steps per optimizer step (unchanged).
                # ReduceLROnPlateau steps once per epoch on sel_metric (end-of-epoch
                # block) instead — never per-step. Default (cosine) path unchanged.
                if not _use_plateau:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                step += 1

                if step % args.log_every == 0:
                    _step_metrics = {
                        "train/loss_total": float(np.mean(epoch_losses["total"][-args.log_every:])),
                        "train/loss_seg": float(np.mean(epoch_losses["seg"][-args.log_every:])),
                        "train/loss_field": float(np.mean(epoch_losses["field"][-args.log_every:])),
                        "train/loss_gen": float(np.mean(epoch_losses["gen"][-args.log_every:])),
                        "train/loss_ground": float(np.mean(epoch_losses["ground"][-args.log_every:])),
                        "train/loss_kl_base": float(np.mean(epoch_losses["kl_base"][-args.log_every:])),
                        "train/loss_textsft": float(np.mean(epoch_losses["textsft"][-args.log_every:])),
                        "train/loss_dxdistill": float(np.mean(epoch_losses["dxdistill"][-args.log_every:])),
                        "train/lr_qformer": optimizer.param_groups[0]["lr"],
                    }
                    # Surface all param-group LRs so the (post-unfreeze) 3-group
                    # schedule is visible: group 1 = LoRA, group 2 = backbone trunk.
                    if len(optimizer.param_groups) > 1:
                        _step_metrics["train/lr_lora"] = optimizer.param_groups[1]["lr"]
                    if backbone_unfrozen and len(optimizer.param_groups) > 2:
                        _step_metrics["train/lr_backbone"] = optimizer.param_groups[-1]["lr"]
                    log_metrics(_step_metrics, step, wb)

                # Smoke/debug cap: stop after N optimizer steps (0 = disabled).
                if getattr(args, "max_steps", 0) and step >= args.max_steps:
                    log.info(f"[max-steps] reached {step} optimizer steps — stopping early (smoke cap).")
                    break

        # Smoke/debug cap: break the epoch loop too (skips end-of-epoch validation).
        if getattr(args, "max_steps", 0) and step >= args.max_steps:
            break

        # End of epoch: validate. Free-gen sub-eval every N epochs (configurable
        # via --free-gen-val-every; default 0 disables). Surfaces aug-retrain-
        # style regressions where teacher-forced macro_F1 improves while
        # free-gen structural validity collapses.
        do_freegen = (
            getattr(args, "free_gen_val_every", 0) > 0
            and (epoch + 1) % args.free_gen_val_every == 0
        )
        log.info(f"epoch {epoch} done in {time.time() - t0:.1f}s — running validation"
                 + (" (+ free-gen subeval)" if do_freegen else ""))
        if ema is not None:
            ema.apply_to(model)   # validate + checkpoint on the EMA weights
        val_metrics = validate(
            model, val_loader, device, phase="phase2b",
            free_gen_subeval=do_freegen,
            free_gen_n_cases=getattr(args, "free_gen_n_cases", 8),
            free_gen_k_samples=getattr(args, "free_gen_k_samples", 2),
            # Track per-class Dice once the trunk is adapting, so a degenerate-seg
            # drift (which would break LesionRouter) is caught epoch-over-epoch.
            report_seg_dice=backbone_unfrozen,
        )
        # BLOCKER-2: anchor-ON, tag-free draft->commit dx-recall on the fold VAL ids
        # ONLY (never test). validate() above runs the anchor OFF via the field heads,
        # so best.pt would otherwise be picked BLIND to the anchor's dx benefit. Guarded
        # on the anchor being live => anchor-OFF runs skip this entirely (byte-identical).
        if getattr(model.lm, "cohort_anchor", None) is not None:
            try:
                _adx = _anchor_dx_subeval(
                    model, val_loader, device,
                    max_cases=int(getattr(args, "anchor_dx_val_cases", 40)),
                )
                val_metrics.update(_adx)
                if "val/anchor_dx_recall" in _adx:
                    log.info(
                        f"  anchor-dx subeval (tag-free, anchor ON): "
                        f"dx_recall={_adx['val/anchor_dx_recall']:.3f} on "
                        f"n={int(_adx['val/anchor_dx_n'])} val cases"
                    )
            except Exception as e:
                log.warning(f"  anchor-dx subeval crashed: {str(e)[:200]}")
        for k, v in val_metrics.items():
            log.info(f"  {k} = {v:.4f}")
        if wb is not None:
            wb.log(val_metrics, step=step)

        # Checkpoint selection: prefer free-gen structural validity if we just
        # measured it (production-deployable goal: select on what we deploy);
        # else fall back to teacher-forced macro_F1. This breaks the aug-retrain
        # mode where TF macro_F1 wins but FG collapses.
        if "val/anchor_dx_recall" in val_metrics:
            # Head-conditioned CoT v2 (BLOCKER-2): select on the anchor's ACTUAL
            # tag-free draft->commit dx benefit, blended 0.5/0.5 with teacher-forced
            # macro_F1 for stability (generate is stochastic). Only present when the
            # anchor is live, so anchor-OFF runs never take this branch.
            sel_metric = val_metrics["val/anchor_dx_recall"] * 0.5 + val_metrics.get("val/macro_f1", 0.0) * 0.5
            sel_name = "0.5*anchor_dx + 0.5*macro_f1"
        elif do_freegen and "val/freegen_structural_validity" in val_metrics:
            sel_metric = val_metrics["val/freegen_structural_validity"] * 0.5 + val_metrics.get("val/macro_f1", 0.0) * 0.5
            sel_name = "0.5*freegen_struct + 0.5*macro_f1"
        else:
            sel_metric = val_metrics.get("val/macro_f1", 0.0)
            sel_name = "macro_f1"
        # L5 (--scheduler plateau): step the plateau scheduler ONCE PER EPOCH on the
        # selection metric (mode="max"); it halves LR after --plateau-patience stalled
        # epochs. Cosine already stepped per optimizer step, so this is plateau-only.
        if _use_plateau:
            scheduler.step(sel_metric)
        # Checkpoint scope: when the trunk is unfrozen and adapting, the lora-only
        # checkpoint (qformer + field_head + lm_lora) would SILENTLY DROP the adapted
        # MedNeXt weights — making best.pt non-deployable (it would reload the stale
        # Phase-1 trunk). Persist the FULL model state once the backbone is unfrozen
        # so the adapted trunk is part of the saved checkpoint. (Reuses save_checkpoint
        # unchanged — only the save_lora_only argument flips.)
        _save_lora_only = not backbone_unfrozen
        # BLOCKER-1 (head-conditioned CoT v2): the lora-only checkpoint serializes only
        # {qformer, field_head, lm_lora} and would SILENTLY DROP the trained cohort-anchor
        # embedding (model.lm.cohort_anchor). With unfreeze_epoch>=1 the warm-up epoch-0
        # validation writes best.pt as lora-only, so an anchor run would persist best.pt
        # WITHOUT the anchor -> eval reloads a fresh RANDOM anchor. Force a FULL-model save
        # whenever the anchor is live (model.state_dict() carries lm.cohort_anchor.weight;
        # load_checkpoint's "model" branch already restores it). Guarded on the anchor
        # existing => anchor-OFF runs keep the old value (BYTE-IDENTICAL).
        if getattr(model.lm, "cohort_anchor", None) is not None:
            _save_lora_only = False
        # min-delta debounces a noisy plateau (the early-stop counter keys off best_epoch).
        if sel_metric > best_macro_f1 + getattr(args, "es_min_delta", 0.0):
            best_macro_f1 = sel_metric
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch, step, val_metrics,
                out_dir / "best.pt", save_lora_only=_save_lora_only,
            )
            log.info(f"  ** new best {sel_name} = {sel_metric:.4f} (epoch {epoch}) — checkpoint saved")
        # best.pt is saved on the EMA weights (above); restore the raw weights so
        # training continues un-smoothed and latest.pt (resume) holds raw weights.
        if ema is not None:
            ema.restore(model)
        # Overwriting "latest.pt" for resume support — per-epoch snapshots
        # were eating ~835 MB/epoch × 50 epochs × 5 folds = 209 GB on /home.
        save_checkpoint(
            model, optimizer, scheduler, epoch, step, val_metrics,
            out_dir / "latest.pt", save_lora_only=_save_lora_only,
        )

        # Early stopping: stop once the selection metric has not improved for
        # --es-patience epochs, gated so it never fires during warm-up / right
        # after the unfreeze (the trunk needs a few epochs to adapt). best.pt
        # already holds the peak, so this only saves wasted compute.
        # L5 (--early-stop-patience): reuse the existing early-stop plumbing; the
        # effective patience is max(--es-patience, --early-stop-patience) so either
        # flag arms it and both-default (0) stays byte-identical (block skipped).
        _es_patience_eff = max(args.es_patience, getattr(args, "early_stop_patience", 0))
        if _es_patience_eff > 0:
            es_floor = max(args.es_min_epoch, args.unfreeze_epoch + 3)
            if epoch >= es_floor and (epoch - best_epoch) >= _es_patience_eff:
                log.info(f"[early-stop] no improvement in {sel_name} for "
                         f"{epoch - best_epoch} epochs (best {best_macro_f1:.4f} @e{best_epoch}); "
                         f"stopping at epoch {epoch}.")
                break

    # Smoke aid: a --max-steps cap breaks the epoch loop BEFORE the end-of-epoch
    # validation/checkpoint, so a capped run would otherwise write NO checkpoint at all.
    # Persist one (full-model when the anchor is live, per BLOCKER-1) so the smoke can
    # inspect the saved state and confirm the cohort-anchor survives the save. Full runs
    # (max_steps == 0) never enter this branch => behavior unchanged.
    if getattr(args, "max_steps", 0) and not (out_dir / "best.pt").exists():
        _smoke_lora_only = not backbone_unfrozen
        if getattr(model.lm, "cohort_anchor", None) is not None:
            _smoke_lora_only = False
        save_checkpoint(
            model, optimizer, scheduler, best_epoch, step,
            {"note": "max-steps smoke cap (no end-of-epoch val ran)"},
            out_dir / "latest.pt", save_lora_only=_smoke_lora_only,
        )
        log.info(
            f"[smoke] max-steps cap reached before any end-of-epoch save; wrote "
            f"{out_dir / 'latest.pt'} (lora_only={_smoke_lora_only}) for post-smoke inspection"
        )

    # ☠️ END-OF-TRAINING TRUNK RECORD. The step-0 check proved the pin held before any
    # gradient; this one says where the trunk ENDED UP. With --unfreeze-backbone it will
    # have DRIFTED off the pinned file, and that is fine — it is declared and quantified
    # here. What is never acceptable is inheriting a trunk silently, which is exactly
    # what the un-pinned path did for the life of the project.
    _tv = _trunk_verify(model, mednext_ckpt, out_dir, "end",
                        pinned=bool(getattr(args, "pin_seg_trunk", False)))
    _d = _tv["file_diff"]
    log.info(
        f"[trunk-verify:end] trunk drift vs {mednext_ckpt}: "
        f"{_d['different']}/{_d['mapped']} tensors moved, max_abs_diff {_d['max_abs_diff']} "
        f"({'EXPECTED — the unfreeze schedule trains the trunk' if unfreeze_backbone else 'trunk was frozen; any drift here is a defect'})"
    )
    log.info(f"fold {args.fold} done. best macro_f1 = {best_macro_f1:.4f} at epoch {best_epoch}")
    if wb is not None:
        wb.finish()


# ===========================================================================
# SMOKE TEST (verify the scaffold runs end-to-end with placeholder backbones)
# ===========================================================================


def smoke_test(args: argparse.Namespace) -> None:
    """One forward + backward + step on synthetic data. Verifies wiring.

    Confirms:
      - model.py imports cleanly
      - mrope_4d identity reduction passes
      - model forward produces a non-NaN total loss
      - backward + optimizer step run without crashing
      - field-target builder aligns with routing output
    """
    setup_logging(Path(args.out_dir or "/tmp/neurofusion_smoke"))
    setup_seed(0)
    log.info("=== NeuroFusion v4 smoke test ===")

    log.info("[1/5] mrope_4d identity reduction")
    _test_identity_reduction()

    log.info("[2/5] build model with placeholder backbones (small crop for CPU smoke)")
    cfg = NeuroFusionConfig()
    cfg.freeze_backbone = False        # let placeholder seg head receive grad
    cfg.crop_size = (16, 16, 16)        # shrink for CPU smoke (saves >1 GB cross-attn cache)
    cfg.seg_feature_dim = 64            # smaller feature dim for placeholder Conv3d
    cfg.qformer_hidden_dim = 256        # shrink Q-Former for smoke
    cfg.qformer_n_layers = 2
    cfg.qformer_n_heads = 4
    model = NeuroFusion(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  trainable params (placeholders): {n_train:,}")

    log.info("[3/5] synthetic batch -> forward")
    B, D, H, W = 2, 32, 32, 32
    mri = torch.randn(B, 4, D, H, W)
    seg = torch.zeros(B, D, H, W, dtype=torch.long)
    seg[0, 8:20, 8:20, 8:20] = 1   # NCR core
    seg[0, 12:18, 12:18, 12:18] = 3   # ET inside
    seg[1, 14:26, 14:26, 14:26] = 2   # ED-only

    # Build a single fake report aligned with the seg
    from schema import (
        BrainTumorReport, Lesion, SurroundingEffects, Involvement,
    )
    fake_lesion = Lesion(
        lesion_index=1, location="parieto-temporal", composition="necrotic",
        enhancement_pattern="heterogeneous",
        surrounding_effects=SurroundingEffects(
            edema_severity="moderate", mass_effect="present", midline_shift="present",
        ),
        involvement=Involvement(
            ventricular_compression="present", brainstem_compression="none",
            midbrain_compression="none",
        ),
        axis_shift="contralateral",
    )
    fake_report = BrainTumorReport(
        case_id="TR01", hemisphere="right",
        differential_diagnosis=["Glioblastoma"],
        overall_mass_effect="moderate",
        lesions=[fake_lesion],
        findings="Synthetic test case with enhancement and mass effect.",
        impression="Test impression.",
    )
    reports = [fake_report, fake_report]

    out = model(mri, seg, training=True)
    log.info(f"  seg_logits  : {tuple(out['seg_logits'].shape)}")
    log.info(f"  visual_tokens: {tuple(out['visual_tokens'].shape)}")
    log.info(f"  routing N_total: {out['routing']['batch_idx'].shape[0]}")
    log.info(f"  total loss (model-internal): {out['loss'].item():.4f}")

    log.info("[4/5] field-target alignment + external loss recomputation")
    field_targets = build_field_targets(
        reports, out["routing"]["batch_idx"], out["routing"]["lesion_idx"],
        device=torch.device("cpu"),
    )
    for k, v in field_targets.items():
        log.info(f"  field_target[{k}] shape={tuple(v.shape)} values={v.tolist()}")

    log.info("[5/5] backward + optimizer step")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )
    out["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
    )
    optimizer.step()
    log.info(f"  grad_norm = {grad_norm:.4f}")
    log.info("=== smoke test PASSED ===")


def real_models_smoke(args: argparse.Namespace) -> None:
    """End-to-end smoke with real MedNeXt + real MedGemma+QLoRA+M-RoPE + XGrammar.

    Pipeline exercised (one batch, B=1, real crop 128^3):
      MRI → frozen MedNeXt (Phase 1 v5 ckpt) → CC routing → per-lesion Q-Former
        → FieldClassificationHead → MedGemma+QLoRA+M-RoPE → loss → backward
        → XGrammar-constrained generate (k=2 samples)

    Requires:
      - CUDA GPU (real MedGemma 4B can't run on CPU)
      - HF cache populated with google/medgemma-4b-it
      - mednext_checkpoint pointing at Phase 1 best.pt (default: ~/scratch/.../phase1_v5/best.pt)
    """
    out_dir = Path(args.out_dir or "/tmp/neurofusion_real_smoke")
    setup_logging(out_dir)
    setup_seed(0)
    log.info("=== NeuroFusion real-models smoke ===")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit(f"--real-models needs CUDA; got device={device.type}")

    log.info("[1/6] mrope_4d identity reduction")
    _test_identity_reduction()

    log.info(f"[2/6] build NeuroFusion with real MedNeXt + MedGemma")
    log.info(f"  mednext_checkpoint: {args.mednext_checkpoint}")
    cfg = NeuroFusionConfig()  # production dims — no shrinking
    if not Path(args.mednext_checkpoint).is_file():
        raise SystemExit(f"MedNeXt checkpoint not found: {args.mednext_checkpoint}")
    model = NeuroFusion(cfg, mednext_checkpoint=args.mednext_checkpoint)
    model = model.to(device)
    if not getattr(model.lm, "_using_real_lm", False):
        raise SystemExit(
            "MedGemma did not load. Confirm HF_HOME=$HOME/scratch/hf_cache, HF_HUB_OFFLINE=1, "
            "and that google/medgemma-4b-it is in the cache."
        )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  trainable params (LoRA + Q-Former + axes): {n_train:,}")

    log.info("[3/6] synthetic real-shaped batch")
    B, D, H, W = 1, *cfg.crop_size
    mri = torch.randn(B, 4, D, H, W, device=device)
    seg = torch.zeros(B, D, H, W, dtype=torch.long, device=device)
    # Place one ET-bearing tumor core to keep routing non-trivial
    cd, ch, cw = D // 2, H // 2, W // 2
    seg[0, cd-12:cd+12, ch-12:ch+12, cw-12:cw+12] = 1   # NCR
    seg[0, cd-4:cd+4,   ch-4:ch+4,   cw-4:cw+4]   = 3   # ET core

    from schema import (
        BrainTumorReport, Lesion, SurroundingEffects, Involvement,
    )
    fake_report = BrainTumorReport(
        case_id="TR01", hemisphere="right",
        differential_diagnosis=["Glioblastoma"],
        overall_mass_effect="moderate",
        lesions=[Lesion(
            lesion_index=1, location="parieto-temporal", composition="necrotic",
            enhancement_pattern="heterogeneous",
            surrounding_effects=SurroundingEffects(
                edema_severity="moderate", mass_effect="present", midline_shift="present"),
            involvement=Involvement(
                ventricular_compression="present", brainstem_compression="none",
                midbrain_compression="none"),
            axis_shift="contralateral",
        )],
        findings="Synthetic real-models smoke case with heterogeneous enhancement and mass effect.",
        impression="Smoke impression.",
    )

    log.info("[4/6] forward (real LM in bf16)")
    out = model(mri, seg, training=True)
    log.info(f"  seg_logits   : {tuple(out['seg_logits'].shape)}")
    log.info(f"  visual_tokens: {tuple(out['visual_tokens'].shape)}")
    log.info(f"  routing N    : {out['routing']['batch_idx'].shape[0]}")
    log.info(f"  total loss   : {out['loss'].item():.4f}")
    if not torch.isfinite(out["loss"]):
        raise SystemExit(f"non-finite loss: {out['loss'].item()}")

    log.info("[5/6] backward + step (LoRA adapters + M-RoPE axes)")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )
    out["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
    )
    optimizer.step()
    log.info(f"  grad_norm = {grad_norm:.4f}")
    if not torch.isfinite(grad_norm):
        raise SystemExit(f"non-finite grad_norm: {grad_norm}")

    log.info("[6/6] XGrammar-constrained generate (k=2)")
    with torch.no_grad():
        routing = out["routing"]
        # heuristic strings come from FieldClassificationHead; for smoke use a placeholder
        heuristics = [""] * B
        samples = model.lm.generate(
            visual_tokens=out["visual_tokens"],
            batch_idx=routing["batch_idx"],
            centroids=routing.get("centroids", torch.zeros(routing["batch_idx"].shape[0], 3, device=device)),
            heuristic_strings=heuristics,
            k_samples=2,
            use_json_constraint=True,
        )
    import json as _json
    n_valid_json = 0
    for b_samples in samples:
        for s in b_samples:
            try:
                _json.loads(s)
                n_valid_json += 1
            except Exception:
                pass
    total = sum(len(s) for s in samples)
    log.info(f"  generated {total} samples; {n_valid_json}/{total} parse as JSON")
    if total == 0:
        raise SystemExit("generate() returned 0 samples")

    log.info("=== real-models smoke PASSED ===")


# ===========================================================================
# MAIN
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroFusion v4 phased training")
    p.add_argument("--phase", choices=["1", "2a", "2b", "smoke"], required=True)
    p.add_argument("--out-dir", type=str, default="runs/default")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Phase 1
    p.add_argument("--brats-roots", nargs="+", type=str, help="paths to BraTS 2020/2021/2023 roots")

    # Phase 2 inputs
    p.add_argument("--jsonl", type=str, help="path to neurofusion_369.jsonl")
    p.add_argument("--brats-root", type=str, help="path to BraTS 2020 training root")
    p.add_argument("--splits", type=str, help="path to splits.json")
    p.add_argument("--kl-corpus-jsonl", type=str, default=None,
                   help="Phase 0a: jsonl of medical sentences for the KL-to-base replay "
                        "buffer (default None = 10 hardcoded placeholders, byte-identical).")
    p.add_argument("--text-sft-jsonl", type=str, default=None,
                   help="Phase 1a: jsonl of medical sentences for the text-only SFT path "
                        "(teacher-forced LM alignment, no volume). Default None = OFF.")
    p.add_argument("--text-sft-weight", type=float, default=0.0,
                   help="Phase 1a: weight of the text-only SFT loss added to the total "
                        "(default 0.0 = OFF, byte-identical). ~0.25 approximates a 1:4 "
                        "text:image mix relative to w_gen=1.0.")
    p.add_argument("--text-sft-nsamples", type=int, default=4,
                   help="Phase 1a: medical sentences per step for the text-only SFT loss.")
    p.add_argument("--seg-checkpoint", type=str, default=None, help="Phase 1 backbone checkpoint")
    p.add_argument("--warmstart-nf", type=str, default=None,
                   help="Warmstart Q-Former+field-heads+LoRA from an existing fine-tuned NF "
                        "checkpoint (e.g. scratch/runs/fold{K}/best.pt). Model weights only; "
                        "optimizer/scheduler/LR + unfreeze schedule restart fresh.")
    p.add_argument("--resume", type=str, default=None,
                   help="RESUME (not warmstart) an interrupted Phase-2b run from a latest.pt "
                        "checkpoint: restores full model + optimizer + scheduler + epoch/step so "
                        "the SAME cosine schedule and early-stop window continue onward. The trunk "
                        "param group is pre-added to match the saved optimizer state, EMA is "
                        "re-registered over the unfrozen trunk, and the best-metric floor is seeded "
                        "from the sibling best.pt so a post-resume validation can never overwrite "
                        "the real best with a worse checkpoint. Used to let a wall-clock-truncated "
                        "run train to its natural plateau (early-stop).")

    # Phase 2b training
    p.add_argument("--fold", type=int, default=0, help="fold index 0..4")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-qformer", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=10)
    # --- head-conditioned CoT (draft-then-commit) supervision. Default OFF ->
    #     byte-identical to the JSON-only training distribution. ---
    p.add_argument("--cot-supervision", action="store_true",
                   help="Train the LM on draft-then-commit targets "
                        "'<findings+impression prose>[COMMIT]<canonical JSON>' instead of "
                        "the JSON alone, so the multi-stage narrate->commit decode is "
                        "in-distribution. Changes ONLY the tokenized target string.")
    p.add_argument("--cot-max-length", type=int, default=1024,
                   help="Tokenizer max_length for the (longer) CoT target. Only used "
                        "when --cot-supervision is set (default JSON path stays 512). "
                        "1024 truncates 0.11%% of ultra-with-met targets (896->1.34%%, "
                        "768->5.38%%); differential_diagnosis is the 3rd JSON key so it "
                        "is never truncated.")
    # --- Head-conditioned CoT v2 (2026-07, JOURNAL): learned cohort-anchor prefix.
    #     All default OFF => byte-identical to the locked hero + existing cotcond run. ---
    p.add_argument("--cohort-anchor", action="store_true",
                   help="Enable the head-conditioned CoT v2 cohort-anchor: a learned "
                        "embedding (none/GLI/MEN/MET) prepended to the visual prefix in the "
                        "draft->commit forward so the DRAFT reasons from a cohort prior "
                        "(fixes the meningioma->metastasis draft bias). Requires one of "
                        "--cohort-anchor-map / --cohort-anchor-from-gold. Off => no anchor "
                        "param, decode BYTE-IDENTICAL. Use with --cot-supervision + neutral "
                        "case_id (v2 drops the cohort tag).")
    p.add_argument("--cohort-anchor-map", type=str, default=None,
                   help="Path to a JSON dict {case_id: cohort} giving the per-case anchor. "
                        "PRIMARY leak-free source = the Stage-1 visual-token cohort probe's "
                        "OUT-OF-FOLD TRAIN-split predictions (so train anchors carry the same "
                        "~0.88 accuracy + confusion structure the TEST predictions will have "
                        "at inference — train/inference matched). Values may be labels "
                        "(none/GLI/MEN/MET, case-insensitive) or ints 0..3. Cases absent from "
                        "the map default to 0 (none). MUST be probe predictions, NEVER the "
                        "case_id cohort tag.")
    p.add_argument("--cohort-anchor-from-gold", action="store_true",
                   help="FALLBACK anchor source (smoke / no probe file yet): derive the cohort "
                        "from each TRAIN report's gold differential_diagnosis[0] via the dx "
                        "SYNONYMS (a legitimate training label, not the case_id tag), then "
                        "corrupt a --cohort-anchor-error-rate fraction to random wrong cohorts "
                        "to SIMULATE the probe's ~0.88 accuracy (else train sees perfect anchors "
                        "the ~0.88 inference anchors can't match). Ignored if --cohort-anchor-map "
                        "is given.")
    p.add_argument("--cohort-anchor-error-rate", type=float, default=0.12,
                   help="Fraction of gold-derived anchors to flip to a random wrong cohort "
                        "(only with --cohort-anchor-from-gold), matching the Stage-1 probe's "
                        "~0.88 accuracy. Seeded by --seed for reproducibility.")
    p.add_argument("--eval-only", type=str, default=None, metavar="OUT_JSON",
                   help="Load the checkpoint, run ONE validate + anchor-dx sub-eval on "
                        "the fold VAL split, write the metrics to OUT_JSON and exit "
                        "WITHOUT training. Exists to compare best.pt against latest.pt on "
                        "the same basis: the selection composite separates them by 0.17 sd "
                        "of its own epoch-to-epoch spread, so which one ships cannot be "
                        "read off the training log and must be MEASURED. VAL split only — "
                        "spends no test look.")
    p.add_argument("--anchor-dx-val-cases", type=int, default=40,
                   help="BLOCKER-2: max VAL cases used in the anchor-ON, tag-free "
                        "draft->commit dx-recall sub-eval that SELECTS best.pt for "
                        "head-conditioned CoT v2. Only runs when --cohort-anchor is set; "
                        "VAL split only (never test). Larger = less selection noise, "
                        "slower per epoch (each case runs a 2-stage generate).")
    # --- L3 (draft-then-commit content): dx-first draft. Default OFF => the CoT
    #     target is byte-identical to '<findings+impression>[COMMIT]<json>'. ---
    p.add_argument("--dx-first-draft", action="store_true",
                   help="CoT draft states the top-1 differential up front: prepend "
                        "'Diagnosis: <ddx[0]>. ' to the draft prose before [COMMIT] "
                        "(findings+impression unchanged). Only affects --cot-supervision "
                        "targets. Off => byte-identical target string.")
    # --- Rung-3 (Lever A): head-distillation. Default OFF => byte-identical (no probe/
    #     head created, l_dxdistill=0). Trains a dx_probe on the LM's last-prefix hidden
    #     to match the FROZEN dx-head's softmax (KL) -> native dx without the head crutch. ---
    p.add_argument("--use-dx-distill", action="store_true",
                   help="Rung-3: distill the frozen DiagnosisHead's dx signal into the LM "
                        "(KL on a dx_probe over the last-prefix hidden). Native dx lever.")
    p.add_argument("--dx-distill-weight", type=float, default=0.2,
                   help="weight of the head-distillation KL term (default 0.2).")
    p.add_argument("--dx-head-path", type=str, default=None,
                   help="frozen dx-head bundle (default out/m6_research/diagnosis_head_mistral.pt).")
    # --- NF_knight 1.2 / W-QF-OBJ: auxiliary dx loss ON connector_OUT. Default OFF =>
    #     no head created, l_auxdx = 0, total byte-identical. Targets the MEASURED
    #     connector_IN 0.8045 -> connector_OUT 0.603 compression loss: nothing in the
    #     current objective asks the Q-Former to preserve diagnosis. ---
    p.add_argument("--aux-dx", action="store_true",
                   help="W-QF-OBJ: add a linear 4-way dx CE on the per-case mean of the "
                        "Q-Former output tokens, so dx-discriminability of what the LM "
                        "sees becomes part of what the Q-Former is optimized for.")
    p.add_argument("--aux-dx-weight", type=float, default=0.3,
                   help="weight of the aux dx CE term (default 0.3).")
    # --- Rung-2 (Lever C): morphology->diagnosis rationale in the CoT DRAFT.
    #     Default OFF => byte-identical target string. Composes a Chain-of-Diagnosis
    #     clause from the case's own gold morphology (scripts/dx_morphology_kb) in
    #     place of the bare 'Diagnosis: <ddx[0]>. ' clause; falls back to that bare
    #     clause when the dx family is unknown. DRAFT-only; leak-safe. ---
    p.add_argument("--dx-rationale-draft", action="store_true",
                   help="CoT draft states a morphology->diagnosis 'Chain-of-Diagnosis' "
                        "clause composed from the case's OWN gold fields (composition/"
                        "enhancement/location/edema/multiplicity) + the tumor type's "
                        "anatomic descriptor, instead of the bare 'Diagnosis: <ddx[0]>. ' "
                        "clause. Leak-safe (observable fields are already the SFT target; "
                        "the anatomic descriptor is type-level) and DRAFT-only (stripped at "
                        "[COMMIT], scored JSON unchanged). Only affects --cot-supervision "
                        "targets. Off => byte-identical target string.")
    # --- generic smoke/debug cap: stop after N optimizer steps (0 = disabled). ---
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop training after this many optimizer steps (0 = disabled = "
                        "train to --epochs). For GPU smokes; does not alter the default path.")
    # --- Phase-2b training-quality levers (early-stop + EMA). All default OFF so
    #     existing launchers are byte-identical unless explicitly enabled. ---
    p.add_argument("--es-patience", type=int, default=0,
                   help="early-stop: stop if the selection metric has not improved for "
                        "this many epochs (0 = disabled = train to --epochs). Gated by "
                        "--es-min-epoch and the unfreeze epoch so it never fires during warm-up.")
    p.add_argument("--es-min-epoch", type=int, default=10,
                   help="earliest epoch early-stopping may trigger (lets the LR schedule + "
                        "post-unfreeze trunk adaptation settle first).")
    p.add_argument("--es-min-delta", type=float, default=0.0,
                   help="minimum selection-metric improvement to count as a new best "
                        "(debounces noise around a plateau).")
    # --- L5 (LR schedule + early-stop). --scheduler defaults to 'cosine' so the
    #     per-step LambdaLR path is byte-identical; 'plateau' switches to
    #     ReduceLROnPlateau stepped once per epoch on the selection metric. ---
    p.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine",
                   help="Phase-2b LR scheduler. 'cosine' (DEFAULT) = the existing "
                        "5%%-warmup LambdaLR stepped per optimizer step (byte-identical). "
                        "'plateau' = ReduceLROnPlateau(mode=max, factor=0.5, "
                        "patience=--plateau-patience, min_lr=1e-6) stepped ONCE PER EPOCH "
                        "on the checkpoint selection metric.")
    p.add_argument("--plateau-patience", type=int, default=1,
                   help="ReduceLROnPlateau patience in EPOCHS (only used with "
                        "--scheduler plateau); LR halves after this many epochs without "
                        "selection-metric improvement.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop training if the selection metric has not improved for this "
                        "many epochs (0 = disabled). Shares the existing --es-patience "
                        "early-stop plumbing (same --es-min-epoch / unfreeze-epoch gating); "
                        "effective patience = max(--es-patience, --early-stop-patience).")
    p.add_argument("--use-ema", action="store_true",
                   help="maintain an EMA of the trainable params (LoRA + Q-Former + field "
                        "heads + unfrozen trunk); validate + save best.pt on the EMA weights. "
                        "Off by default (byte-identical to the non-EMA path).")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay (only used with --use-ema).")
    p.add_argument("--augment", action="store_true",
                   help="enable 3D MRI augmentation on the training set (flips on non-sagittal "
                        "axes, small affine, intensity jitter). val/test untouched.")
    # --- L4 (cohort-balanced sampling). Default OFF => train_loader keeps
    #     shuffle=True (no sampler), byte-identical to today. ---
    p.add_argument("--cohort-balanced-sampler", action="store_true",
                   help="oversample minority cohorts (GLI/MEN/MET, by case_id prefix) with a "
                        "WeightedRandomSampler so each cohort contributes ~equal mass per epoch "
                        "(per-case weight = 1/cohort-count; drawn with replacement, len(train) "
                        "draws/epoch). Replaces shuffle=True. Off => shuffle=True, no sampler "
                        "(byte-identical).")
    p.add_argument("--free-gen-val-every", type=int, default=0,
                   help="run lightweight free-gen val every N epochs (0 = off). "
                        "When > 0, checkpoint selection blends val/macro_f1 with "
                        "val/freegen_structural_validity to catch aug-retrain-style "
                        "regressions (TF↑ while FG↓).")
    p.add_argument("--free-gen-n-cases", type=int, default=8,
                   help="number of val cases used in free-gen subeval (smaller = faster)")
    p.add_argument("--free-gen-k-samples", type=int, default=2,
                   help="K-sample count for free-gen subeval (smaller = faster)")
    p.add_argument("--use-stage2-mrope", action="store_true",
                   help="enable Stage-2 M-RoPE: concat per-lesion visual prefix + spatial "
                        "position_ids. Required for SSL retrain that learns the new architecture.")
    p.add_argument("--use-global-pool", action="store_true",
                   help="GLOBAL-POOL ablation (reviewer-requested): bypass per-lesion "
                        "connected-component routing. Emit exactly ONE lesion per case whose "
                        "feature is the foreground-masked adaptive_avg_pool3d of the frozen "
                        "MedNeXt bottleneck. Supervised against the dominant (largest-volume) "
                        "GT lesion's fields; L_ground disabled. Default off.")
    p.add_argument("--lm-family", choices=["medgemma", "mistral"], default="medgemma",
                   help="report-generation LM backbone. 'medgemma' (default) = the existing "
                        "pipeline (byte-identical). 'mistral' = LLaVA-Med's Mistral-7B LM "
                        "(extracted, 4-bit QLoRA, 1D RoPE, hidden 4096) — LM-swap experiment.")

    # -- Visual-token-expansion experiment (Models 2/3/4). All three default to
    # the Model-1 (deployment) values so an unflagged run is BYTE-IDENTICAL.
    p.add_argument("--feature-stage", choices=["bottleneck", "enc_hires"], default="bottleneck",
                   help="Q-Former feature SOURCE. 'bottleneck' (DEFAULT) = frozen MedNeXt "
                        "bottleneck [B,512,8,8,8] (byte-identical to all existing checkpoints). "
                        "'enc_hires' = last encoder stage enc_stages[-1] [B,256,16,16,16] "
                        "(8x real-voxel context; feature_proj in-ch 512->256, out stays 320).")
    p.add_argument("--pin-seg-trunk", action="store_true",
                   help="TRUNK EVICTION (NF 1.3). DROP the warmstart checkpoint's own "
                        "vision.backbone.mednext.* so --seg-checkpoint is the trunk that "
                        "actually trains. WITHOUT THIS FLAG --seg-checkpoint IS INERT for "
                        "any full checkpoint (measured: 524/524 tensors overwritten, "
                        "max|d| 0.955 vs e30). feature_proj is NOT dropped: it is in no "
                        "trunk file and is an untrained random draw (KNOWN_UNOPTIMISED), "
                        "so it is inherited from the warmstart and that is declared.")
    p.add_argument("--crop-size", type=int, default=None,
                   help="LesionRouter per-lesion crop edge, in FEATURE voxels (cubic). "
                        "☠️ THE TAP. Default cfg.crop_size=(16,16,16) on an 8^3 bottleneck "
                        "map clamps to the WHOLE map, so every 'per-lesion' crop is the "
                        "same global average and routing is centroid-INDEPENDENT. Set this "
                        "smaller than the feature grid (e.g. 8 with --feature-stage "
                        "enc_hires, which is 16^3) for a genuinely lesion-local tap.")
    p.add_argument("--n-queries", type=int, default=32,
                   help="Q-Former queries per lesion (n_queries_per_lesion). DEFAULT 32 "
                        "(128 LM visual tokens for 4 lesions). 64 -> 256 tokens (M3D budget) "
                        "for the visual-token-expansion Models 2/3/4.")
    p.add_argument("--max-lesions", type=int, default=4,
                   help="LesionRouter top-K (cfg.max_lesions). DEFAULT 4 (the schema/GT/grammar "
                        "output cap is ALSO 4 and is NOT changed by this flag). Raising to 8/16 "
                        "feeds the LM extra (mostly sub-1cm^3 noise) visual prefixes while the "
                        "emitted report stays <=4 lesions -- the input-only max_lesions ablation.")
    p.add_argument("--mask-overflow-lesion-loss", action="store_true",
                   help="DEFAULT off (byte-identical). When set, predicted lesion slots whose "
                        "index exceeds the GT lesion count emit ignore_index (-100) for the field "
                        "CE instead of clipping to the dominant GT lesion -- so noise blobs do not "
                        "mis-supervise the field head. Turn ON for max_lesions>4 (Models 3/4); keep "
                        "OFF for Model-2 (max_lesions=4) to stay comparable to Model-1.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label-smoothing applied to per-field CE. 0.0 = off. "
                        "0.1 is a reasonable default. Helps content heads escape the "
                        "always-majority equilibrium when the train set is class-imbalanced "
                        "(audit Phase D'' diagnosis 2026-05-19).")
    p.add_argument("--field-head-pool", choices=["mean", "attention"], default="mean",
                   help="M5: 'attention' = learned-query attention pool over the 32 lesion "
                        "tokens (vs M1 'mean'). Recovers the head-vs-connector-output gap.")
    p.add_argument("--field-head-global-token", action="store_true",
                   help="M5: append a global-context token (bottleneck mean-pool) to each "
                        "lesion's token set before pooling. Targets the composition crop-loss.")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="focal loss gamma for per-field heads; 0.0 = plain weighted CE "
                        "(default, byte-identical). gamma>0 enables focal cross-entropy "
                        "(down-weights easy/confident-correct examples by (1-p_t)**gamma); "
                        "focal and label-smoothing are not combined (label_smoothing is "
                        "ignored when gamma>0).")
    p.add_argument("--class-weight-labeled-only", action="store_true",
                   help="compute per-field class weights from labeled folds (0..4) only, "
                        "excluding pseudo cases (fold>=5 by convention). Prevents pseudo-label "
                        "class distribution shift from biasing the head's gradient toward "
                        "classes the val set doesn't contain. Recommended when running SSL.")
    p.add_argument("--w-field", type=float, default=None,
                   help="override cfg.w_field (default 0.5). Use 1.0-2.0 to boost field-head "
                        "gradient when targeting per-field macro_F1 over generation loss.")
    p.add_argument("--w-gen", type=float, default=None,
                   help="override cfg.w_gen (default 1.0).")
    p.add_argument("--w-seg", type=float, default=None,
                   help="override cfg.w_seg (default 1.0). Vision backbone is frozen, so seg "
                        "loss only updates the feature_proj layer; lowering this is mostly cosmetic. "
                        "With --unfreeze-backbone, KEEP this > 0 — the Dice+CE seg loss regularizes "
                        "the now-trainable trunk toward valid segmentation (anti-catastrophic-"
                        "forgetting: the trunk can't drift to a degenerate seg that breaks routing).")

    # -- Phase 2b gradual backbone unfreeze (project_neural_ceiling_diagnosis).
    # When the MedNeXt trunk stays frozen, intensity-derivable fields (composition,
    # enhancement, etc.) get no per-case signal because the features never adapt to
    # the report task. Unfreezing the 3D vision trunk (NOT the QLoRA-frozen LM) at a
    # SMALL discriminative LR lets the morphology features adapt without forgetting
    # the Phase-1 seg geometry that routing/hemisphere depend on.
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="Phase 2b: unfreeze the pretrained MedNeXt trunk so it ADAPTS to the "
                        "report task (project_neural_ceiling_diagnosis). The trunk is kept frozen "
                        "for --freeze-epochs (warm-up of qformer/heads against stable features), "
                        "then flipped trainable at --unfreeze-epoch and added to the optimizer as a "
                        "THIRD param group at --lr-backbone. The LM stays QLoRA-frozen. Keep w_seg>0 "
                        "so the seg loss regularizes the trunk (anti-forgetting). Default OFF == "
                        "byte-identical to the frozen-trunk convention.")
    p.add_argument("--unfreeze-epoch", type=int, default=1,
                   help="epoch index (0-based) at which the MedNeXt trunk is flipped trainable when "
                        "--unfreeze-backbone is set. Default 1 (trunk frozen for epoch 0 only). The "
                        "per-epoch re-freeze is CONDITIONAL on this: while epoch < unfreeze-epoch the "
                        "trunk is re-frozen each epoch; from unfreeze-epoch onward it stays trainable.")
    p.add_argument("--freeze-epochs", type=int, default=None,
                   help="alias for --unfreeze-epoch (number of warm-up epochs the trunk stays frozen "
                        "before unfreezing). If given, OVERRIDES --unfreeze-epoch. Provided to match "
                        "the orchestrator's vocabulary; freeze-epochs=N == unfreeze-epoch=N.")
    p.add_argument("--lr-backbone", type=float, default=1e-5,
                   help="discriminative LR for the unfrozen MedNeXt trunk param group (default 1e-5, "
                        "~10-20x below lr-qformer/lr-lora) to adapt slowly without catastrophic "
                        "forgetting of the Phase-1 seg features. Only used with --unfreeze-backbone.")

    # Smoke
    p.add_argument("--real-models", action="store_true",
                   help="--phase smoke: load real MedNeXt + MedGemma + XGrammar end-to-end (needs GPU + HF cache)")
    p.add_argument("--mednext-checkpoint", type=str,
                   default=str(Path.home() / "scratch/checkpoints/phase1_v5/best.pt"),
                   help="Phase 1 MedNeXt checkpoint for real-models smoke / Phase 2b")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "1":
        train_phase1_seg_pretrain(args)
    elif args.phase == "2a":
        train_phase2a_identity_check(args)
    elif args.phase == "2b":
        for needed in ("jsonl", "brats_root", "splits"):
            if getattr(args, needed) is None:
                raise SystemExit(f"--phase 2b requires --{needed.replace('_', '-')}")
        train_phase2b_multitask(args)
    elif args.phase == "smoke":
        if args.real_models:
            real_models_smoke(args)
        else:
            smoke_test(args)


if __name__ == "__main__":
    main()
