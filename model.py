"""
model.py — NeuroFusion v4 architecture skeleton.

Changes from v3:
  - 4-axis M-RoPE retrofit into LM attention (see mrope_4d.py)
  - Classifier heuristic channel: field predictions rendered as text prefix for LM
  - HiResCAM integration point on the field classifier head
  - KL-to-base regularizer (anti-forgetting) in the loss
  - Qwen2.5-VL-7B parallel-pilot variant (stub)

This is a SKELETON. Placeholder backbones run end-to-end on CPU for shape checks;
real MedNeXt / MedGemma / Qwen2.5-VL / XGrammar integrations are marked TODO and
slotted in during Week 2 engineering.

Pipeline stages (A-H, matches diagram 01_main_architecture.svg):
  A. Input             mri: [B, 4, D, H, W]
  B. MedNeXt 3D        seg logits + penultimate features (FROZEN after Phase 1)
  C. Lesion routing    connected-components -> per-lesion crops + centroids
  D. Q-Former          32 queries/lesion + factorized 3D PE -> visual tokens
  E. Field classifier  per-lesion 10-head softmax -> rendered as text prefix
  F. LM (4D M-RoPE)    MedGemma 1.5 4B + QLoRA + retrofitted 4-axis M-RoPE
  G. Calibration       semantic entropy + conformal abstention (inference only)
  H. Output            typed JSON + narrative + HiResCAM citations

Losses:
  L = L_seg + 0.5 L_field + L_gen + 0.3 L_ground + 0.05 L_kl_base
"""
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from mrope_4d import MRope4DConfig, MRope4DAngleTable, apply_mrope_4d, build_position_ids

log = logging.getLogger("model")


# ==========================================================================
# Config
# ==========================================================================

# The three LM weight formats. "nf4" is NF 1.3 native (what ships today), "int4awq" is
# NF 1.3 Marlin, and "fp16" is the merged-but-unquantised attribution anchor that makes the
# native-vs-Marlin delta attributable. Kept as a module-level constant so a test can assert
# the set is exactly this and a fourth option cannot be added without the tests noticing.
_LM_QUANT_CHOICES = frozenset({"nf4", "fp16", "int4awq"})


@dataclass
class NeuroFusionConfig:
    # -- segmentation backbone
    seg_backbone: str = "mednext-medium"
    seg_feature_dim: int = 320
    seg_n_classes: int = 4
    freeze_backbone: bool = True
    # -- backbone unfreeze (Phase-2b trunk adaptation; default OFF == byte-identical
    # to existing frozen checkpoints). These are HINTS read by train.py; the model
    # itself stays agnostic and only acts on requires_grad via set_backbone_trainable.
    # WHY: project_neural_ceiling_diagnosis / project_headline_artifact — when the
    # MedNeXt trunk is frozen, intensity-derivable fields (composition, etc.) get no
    # per-case signal because the features never adapt to the report task. Setting
    # unfreeze_backbone=True lets train.py flip the trunk trainable mid-run and add
    # its params to the optimizer at a SMALL discriminative LR (backbone_lr) so the
    # Phase-1 seg morphology features adapt without catastrophic forgetting. The LM
    # base stays QLoRA-frozen regardless; only the 3D vision trunk is affected.
    # NB: when unfreeze_backbone is True the launcher is expected to also set
    # freeze_backbone=False so SegmentationBackbone.__init__ leaves the trunk
    # requires_grad=True from construction and the gradient-checkpointing/eval
    # paths stay consistent; the gradual schedule (warm-up frozen epochs, then
    # flip via set_backbone_trainable) is owned by train.py.
    unfreeze_backbone: bool = False
    backbone_lr: float = 1e-5
    ensemble_size: int = 3
    # Spatial shape (D, H, W) the volumes are resized to before MedNeXt.
    # Must match dataset.NeuroFusionDataset(target_shape=...). Used by the
    # bilateral-prefill code to compute midline from the actual resized frame
    # rather than hardcoding 64.0 (which silently broke if target_shape changed).
    target_shape: tuple[int, int, int] = (128, 128, 128)

    # -- lesion routing
    # ☠️ RAISING max_lesions INVALIDATES THE CHAIN TOKEN BOUND (FIX 3, 2026-08-08).
    # The proposed `max_new_tokens ~1400` for CoT stages 3-4 was sized from a MEASURED
    # distribution taken at max_lesions=4: n=21 cases, legitimate mass 816-1188 tokens,
    # with the two 1554/2263 outliers identified as LOOPS because they were the SIMPLEST
    # cases (n_lesions=1) while the 4-lesion case used only 944 tokens. A report describing
    # more lesions can legitimately need more tokens, so raising this constant moves the
    # legitimate ceiling and the bound must be RE-MEASURED, not scaled by intuition.
    # See CURRENT_STATE_REGISTER.md "TASK B — FIX 3 SIZED FROM THE DATA".
    max_lesions: int = 4
    min_lesion_volume_cm3: float = 1.0
    # -- overflow-lesion field-loss masking (visual-token-expansion Models 3/4).
    # DEFAULT False == byte-identical: train.build_field_targets clips a predicted
    # lesion slot k whose index exceeds the GT lesion count to the last GT lesion's
    # labels. When max_lesions > the schema/GT cap of 4 (Models 3/4), those extra
    # slots are sub-1cm^3 NOISE blobs and clipping teaches the field head that
    # noise = the dominant tumor (mis-supervision). Set True for max_lesions>4 runs
    # to emit ignore_index (-100) for any slot k-1 >= GT lesion count, so noise
    # slots contribute NO field-head gradient -- making the ablation a clean
    # "does extra capacity hurt" test rather than "does mis-supervision hurt".
    # Keep False for Model-2 (max_lesions=4) to stay comparable to Model-1.
    mask_overflow_lesion_loss: bool = False
    # -- global-pool ablation (reviewer-requested). DEFAULT FALSE — when off the
    # routing path and all downstream outputs are byte-identical to the per-lesion
    # router. When True, LesionRouter bypasses 3D connected-components and emits
    # exactly ONE "lesion" per case whose feature = adaptive_avg_pool3d of the
    # frozen MedNeXt bottleneck features, masked to the segmentation foreground
    # (any tumor class, argmax>0). N_lesions==1 always; everything downstream
    # (Q-Former, field head, prefix builder, generative target) runs unchanged
    # with N=1. The single global prediction is supervised against the DOMINANT
    # (largest-volume) GT lesion's per-lesion fields: train.build_field_targets
    # indexes report.lesions[lesion_idx-1], and the router emits lesion_idx=1,
    # which (per schema.py: lesions ordered largest→smallest) is the dominant
    # lesion. L_ground is disabled (no per-lesion crops to ground).
    use_global_pool: bool = False
    # Feature-space per-lesion crop. For 128^3 input MedNeXt-L bottleneck is 8^3, so
    # 16^3 covers it plus margin; the old 48^3 default ballooned Q-Former cross-attn
    # memory to 110K positions and caused OOM on real data at B=1.
    crop_size: tuple[int, int, int] = (16, 16, 16)

    # -- Q-Former feature SOURCE (visual-token-expansion experiment, 2026-06).
    # Gates WHICH MedNeXt tensor _MONAIMedNeXtWrap projects into the per-lesion
    # visual tokens. DEFAULT "bottleneck" == BYTE-IDENTICAL to every existing
    # checkpoint (the deployment Model-1 stays unchanged with all new flags at
    # defaults). "enc_hires" swaps the source to the last ENCODER stage for 8x
    # the real-voxel context (256-token budget matches M3D), and is the only
    # value that changes feature_proj's in-channels:
    #   "bottleneck": hook self.mednext.bottleneck -> [B, 512, 8, 8, 8]
    #                 (512 = init_filters * 2^4); feature_proj = Conv3d(512, 320, 1).
    #   "enc_hires":  hook self.mednext.enc_stages[-1] -> [B, 256, 16, 16, 16]
    #                 (256 = init_filters * 2^3, 4096 real voxels = 8x context);
    #                 feature_proj = Conv3d(256, 320, 1).
    # In BOTH cases the out-channels stay seg_feature_dim (320) so the Q-Former
    # input_proj is untouched. The frozen path uses a register_forward_hook on
    # the chosen module; the unfrozen per-stage-checkpointed path mirrors it by
    # stashing enc_outputs[-1] (see _mednext_forward_checkpointed). crop_size
    # stays (16,16,16) -- on the enc_hires 16^3 grid that is now the FULL real
    # grid, not padding.
    feature_stage: str = "bottleneck"

    # -- Q-Former
    n_queries_per_lesion: int = 32
    qformer_n_layers: int = 6
    qformer_hidden_dim: int = 768
    qformer_n_heads: int = 12
    use_3d_pe_on_queries: bool = True
    qformer_warm_start: str | None = "Salesforce/instructblip-vicuna-7b"

    # -- LM
    # lm_family selects the report-generation LM backbone. This is the single
    # gate for the MedGemma-vs-Mistral swap (LM-swap experiment, 2026-06):
    #   "medgemma" (default): google/medgemma-4b-it, 4-axis M-RoPE retrofit,
    #       hidden 2560, head_dim 256. BYTE-IDENTICAL to the pre-swap pipeline —
    #       every Mistral-specific branch is gated off when lm_family=="medgemma".
    #   "mistral": LLaVA-Med's Mistral-7B-Instruct-v0.2 LM (extracted, vision
    #       stripped), 4-bit QLoRA, STANDARD 1D RoPE (no M-RoPE patch — its
    #       measured benefit is negligible; see project_mrope_conditional_benefit),
    #       hidden 4096. Q-Former->LM projection auto-resizes to 4096.
    lm_family: str = "medgemma"
    lm_name: str = "google/medgemma-4b-it"
    lm_hidden_dim: int = 2560
    lm_head_dim: int = 256
    # Local extracted Mistral LM checkpoint (lm_state_dict.safetensors + config.json
    # + tokenizer); produced by scripts/extract_llava_med_mistral_lm.py. Only used
    # when lm_family == "mistral".
    #
    # ☠️ THIS WAS A HARDCODED NARVAL PATH AND IT SILENTLY BROKE THE CONTAINER.
    # `cloud-run/download_weights.py:100` pulls the LM to `/weights/llava_med_mistral_lm`
    # (or $NF_MISTRAL_BASE), but neither `nf_infer.py` nor `server.py` ever set
    # `mistral_lm_ckpt` -- ZERO hits for it in either file. So the image downloaded ~13.5
    # GiB and then `_load_real_mistral` looked for it under /home/khawaja1/scratch/, which
    # does not exist in the image: the pull was wasted and the load failed. A developer
    # absolute path is not a default, it is a machine-specific assumption.
    # Env first (container), Narval literal as the fallback so every research run and every
    # existing launcher is BYTE-IDENTICAL.
    mistral_lm_ckpt: str = field(default_factory=lambda: os.environ.get(
        "NF_MISTRAL_BASE",
        "/home/khawaja1/scratch/checkpoints/llava_med_mistral_lm_fast"))

    # ★★ THE NF 1.3 native / NF 1.3 Marlin SWITCH. One codebase, one flag, two builds --
    # deliberately NOT two forks, because a fork guarantees the arms drift and the comparison
    # stops meaning anything. Everything upstream of the LM (MedNeXt trunk, router, Q-Former)
    # is bit-identical across all three settings BY CONSTRUCTION.
    #
    #   "nf4"     NF 1.3 native  -- bitsandbytes NF4 + live LoRA adapters. What ships today.
    #                              Data-free round-to-nearest; dequantises on every forward.
    #   "fp16"    the ATTRIBUTION ANCHOR, not a deliverable. Merged fp16, no quantisation.
    #   "int4awq" NF 1.3 Marlin  -- AWQ-calibrated int4 over the merged fp16 weights.
    #
    # ☠️ THE MERGED ARMS TAKE NO LoRA AND MUST SKIP `lm.*` ON CHECKPOINT LOAD. Their adapter
    # is already baked into the weights, so re-applying it would double-count the update; and
    # the shipped checkpoint's LM keys are PEFT-wrapped names over an NF4 base, which do not
    # correspond to anything in a merged model. _lm_weights_are_external() below is the single
    # predicate every loader must consult, so this cannot be half-applied at one call site.
    #
    # ⚠️ CONFOUND, RECORDED NOT HIDDEN: the LoRA was fitted on top of NF4(W), so merging into
    # fp16 is not identity w.r.t. the trained function. That is exactly why "fp16" exists as a
    # third arm -- without it, any native-vs-Marlin delta is unattributable between "the
    # quantiser changed" and "the base the adapter sits on changed".
    lm_quant: str = "nf4"
    # Directory holding the merged (and possibly quantised) LM. Required unless lm_quant=="nf4".
    lm_quant_ckpt: str | None = None

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # -- M-RoPE Stage-2: concat per-lesion visual prefix + spatial position_ids.
    # Default False (backwards-compatible with the original mean-pool prefix +
    # 1D RoPE that all existing checkpoints were trained under). Set True for
    # the SSL retrain that learns the new architecture from scratch.
    use_stage2_mrope: bool = False

    # -- ★ Stage D prereq ① (2026-08-06): PREFIX LAYOUT, decoupled from M-RoPE.
    #
    # ☠️ `use_stage2_mrope` conflated TWO independent decisions, and the conflation is why
    # a whole family of banked results measured nothing:
    #     (1) HOW THE PREFIX IS BUILT -- concat per-lesion, or MEAN-POOL ACROSS LESIONS
    #     (2) whether visual tokens get SPATIAL M-RoPE position ids
    # With the flag False (what every logged run used) the prefix is mean-pooled across
    # lesions, so emitting per-lesion tokens and then averaging them is a NO-OP BY
    # CONSTRUCTION. That is why the max_lesions 4->8->16 sweep changed prefix length by
    # exactly zero and read as "null": it was never a per-lesion capacity test. And turning
    # the flag on to fix (1) drags in (2), which changes attention math against pretrained
    # 1D RoPE and needs a dedicated SSL retrain -- so the fix was unaffordable for the wrong
    # reason.
    #
    # `visual_prefix` controls (1) ALONE. "per_lesion" gives Stage D a real per-lesion
    # prefix while M-RoPE stays identity, which is the configuration NF-Connector-v2 needs.
    # ★ It also unblocks the cohort anchor: that raises under use_stage2_mrope because the
    # SPATIAL stash counts visual tokens, and with identity stashing there is nothing to
    # break -- so the anchor and a per-lesion prefix can now coexist.
    #
    # ⚠️ "mean_pool" is the default and reproduces every existing checkpoint bit-for-bit.
    # use_stage2_mrope=True still implies per_lesion, so no old config changes meaning.
    visual_prefix: str = "mean_pool"          # "mean_pool" | "per_lesion"

    # -- Head-conditioned CoT v2 (2026-07, JOURNAL/next-paper; NOT in locked MLCN hero).
    # Prepend a small learned cohort-anchor embedding as the FIRST soft prefix token
    # in the draft->commit forward + generate_draft_commit, so the DRAFT (and hence the
    # committed JSON) reasons from a cohort prior — fixing the meningioma->metastasis
    # draft-stage bias (22/24 MEN misses have "metastatic tumor" in the draft prose).
    # The anchor index per case is the Stage-1 LEAK-FREE visual-token cohort probe's
    # prediction (train: OOF train-split preds; inference: test preds) — NEVER the
    # case_id cohort tag. Default False => NO anchor param is created and the
    # forward/decode is BYTE-IDENTICAL to the locked hero + the existing cotcond run.
    use_cohort_anchor: bool = False
    n_cohort_anchor: int = 4   # anchor slots: 0=none/unknown, 1=GLI, 2=MEN, 3=MET

    # -- Rung-3 head-distillation (native dx). Default OFF => no probe/head created,
    # forward + loss BYTE-IDENTICAL. A tiny dx_probe reads the LM's last-prefix hidden
    # (its next-token IS dd0 under --dx-first-draft) and is trained to match the FROZEN
    # DiagnosisHead's softmax over routing["lesion_features"] (KL). Transfers the head's
    # stable dx signal into the LM's native readout (no head crutch at inference).
    use_dx_distill: bool = False
    dx_distill_weight: float = 0.2
    dx_head_path: str | None = None    # None => out/m6_research/diagnosis_head_mistral.pt

    # -- NF_knight 1.2 / W-QF-OBJ: auxiliary diagnosis loss ON connector_OUT.
    #
    # MEASURED PROBLEM (out/m6_research, gate_c_mistral_probe): a LogReg on connector_IN
    # (the DOMINANT lesion's pooled router features ++ n_lesions, 321-d) reads cohort at
    # 0.8045, while the SAME probe on connector_OUT -- the 32xD_lm visual tokens the LM
    # actually consumes -- reads it at 0.603. The Q-Former discards ~0.20 of accuracy that
    # was present at its input, and NOTHING in the current objective penalizes that: the
    # Q-Former is trained only through the field heads (10 geometry/appearance fields) and
    # the LM's next-token loss. Diagnosis is never a target of the compression.
    #
    # THE INTERVENTION: a linear 4-way head on the per-case mean of connector_OUT, trained
    # with CE against the cohort label. It is deliberately LINEAR and deliberately reads the
    # MEAN -- the point is not to build a good classifier (the dx head already exists and is
    # better) but to make dx-discriminability of the POOLED OUTPUT part of what the Q-Former
    # is optimized for. A deeper head could satisfy the loss while leaving the tokens the LM
    # sees just as uninformative.
    #
    # ⚠️ WHAT THIS CANNOT FIX, stated up front: if the -0.20 is STRUCTURAL -- if a randomly
    # initialized Q-Former also scores ~0.60, i.e. the loss is what mean-pooling 32 queries
    # over a cross-attended crop costs regardless of weights -- then no reweighting of the
    # objective recovers it and the answer is an architecture change (a connector_IN skip
    # token), not this. That is precisely what the deciding probe measures, and this flag
    # must not be turned on before it reports.
    #
    # Default OFF => no head is created, `aux_dx_logits` is absent, and both the forward
    # pass and the assembled total are byte-identical to NF_knight 1.0/1.1.
    use_aux_dx: bool = False
    aux_dx_weight: float = 0.3

    # -- 4-axis M-RoPE (v4 primary novelty)
    use_4d_mrope: bool = True
    mrope_rotary_dim: int = 128
    mrope_axis_channels: tuple[int, int, int, int] = (16, 16, 16, 16)

    # -- custom Q-Former (exposes cross-attention weights for L_ground IoU)
    use_custom_qformer: bool = True   # False → fall back to nn.TransformerDecoder (no L_ground)

    # -- classifier heuristic channel (v4 reinstated)
    use_classifier_heuristic: bool = True     # render field preds as text prefix for LM
    classifier_confidence_threshold: float = 0.6  # only render fields with softmax >= this
    # -- M5 field-head upgrades (config-gated; defaults == byte-identical to Model-1).
    #   field_head_pool: "mean" (AdaptiveAvgPool over the 32 lesion tokens, M1 default)
    #     | "attention" (a learned query attention-pools the tokens -- targets the
    #     ~0.11 head-vs-connector-output gap measured by the connector probe).
    #   field_head_global_token: append a global-context token (mean-pool of the whole
    #     frozen MedNeXt bottleneck, projected to D_lm) to each lesion's token set before
    #     pooling -- targets the composition global(0.495)->crop(0.266) "Loss A". Both add
    #     fresh params that warmstart partial-load (strict=False) leaves at init.
    field_head_pool: str = "mean"
    field_head_global_token: bool = False

    # -- JSON-constrained decoding
    use_xgrammar: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    n_samples_for_uq: int = 8

    # -- anti-forgetting
    use_kl_to_base: bool = True
    kl_to_base_weight: float = 0.05
    kl_to_base_replay_buffer_size: int = 2000  # small set of clinical-text samples

    # -- loss weights
    w_seg: float = 1.0
    w_field: float = 0.5
    w_gen: float = 1.0
    w_ground: float = 0.3

    # -- field classification heads
    field_heads: dict[str, int] = field(default_factory=lambda: {
        "location": 16, "composition": 5, "enhancement_pattern": 7,
        "edema_severity": 4, "mass_effect": 2, "midline_shift": 2,
        "ventricular_compression": 2, "brainstem_compression": 2,
        "midbrain_compression": 2, "axis_shift": 3,
    })

    def __post_init__(self) -> None:
        # LM-swap (2026-06): when the Mistral LM is selected, auto-derive the LM
        # geometry so the Q-Former->LM projection (output_proj) and the
        # field-head input dim resize to Mistral's hidden size, and disable the
        # MedGemma-only 4-axis M-RoPE patch (Mistral uses standard 1D RoPE).
        # Only applied if the caller left the MedGemma defaults in place, so an
        # explicit override (e.g. a deliberate experiment) is never clobbered.
        if self.lm_family == "mistral":
            if self.lm_hidden_dim == 2560:      # MedGemma default untouched
                self.lm_hidden_dim = 4096       # Mistral-7B hidden size
            if self.lm_head_dim == 256:         # MedGemma default untouched
                self.lm_head_dim = 128          # 4096 / 32 heads
            # Mistral path does NOT use the MedGemma M-RoPE retrofit.
            self.use_4d_mrope = False
        elif self.lm_family != "medgemma":
            raise ValueError(
                f"Unknown lm_family={self.lm_family!r}; expected 'medgemma' or 'mistral'"
            )

        # ---- NF 1.3 native / Marlin switch: validate FAIL-CLOSED --------------------------
        # A typo'd lm_quant must raise here, not silently fall through to the nf4 branch and
        # produce a "Marlin" run that is actually native. That failure mode would be invisible
        # in every downstream number, which is precisely how a build gets mislabelled.
        if self.lm_quant not in _LM_QUANT_CHOICES:
            raise ValueError(
                f"Unknown lm_quant={self.lm_quant!r}; expected one of {sorted(_LM_QUANT_CHOICES)}"
            )
        if self.lm_quant != "nf4":
            if self.lm_family != "mistral":
                raise ValueError(
                    f"lm_quant={self.lm_quant!r} is a Mistral-only path "
                    f"(got lm_family={self.lm_family!r})")
            if not self.lm_quant_ckpt:
                raise ValueError(
                    f"lm_quant={self.lm_quant!r} requires lm_quant_ckpt (the merged/quantised "
                    "LM directory). Refusing to fall back to the nf4 weights, which would "
                    "silently label a native build as Marlin.")

    def _lm_weights_are_external(self) -> bool:
        """True when the LM weights come from lm_quant_ckpt rather than the NF checkpoint.

        ★ THE SINGLE PREDICATE. Every loader must consult this one function instead of
        re-deriving the condition, so the 'skip lm.* on load' rule cannot be applied at one
        call site and forgotten at another -- the exact defect class that produced the
        `feature_proj` bug (protected from freezing at one site, never wired to an optimiser
        at another) and the C12 metric bug (fixed at the producer, missed at the consumer).
        """
        return self.lm_quant != "nf4"


# ==========================================================================
# 1. Segmentation backbone (MedNeXt)
# ==========================================================================


@contextlib.contextmanager
def seg_tf32_off():
    """Run the SEGMENTATION TRUNK in true fp32 — TF32 OFF — and restore on exit.

    ☠️☠️ WHY. `model.py:3071` routes on `seg_logits.argmax(dim=1)` at eval, and
    `LesionRouter` then sorts connected components by EXACT INTEGER voxel count and
    slices to `max_lesions=4`. A one-voxel change at a lesion boundary can therefore
    change WHICH lesions are routed. TF32 has a 10-bit mantissa, so an A100 conv and a
    CPU conv do not agree to the last bit, and the argmax flips on boundary voxels.

    MEASURED, job 542913 (`out/NFDX_ROUTER_DEVICE_DIVERGENCE_542913.json`, n=60 fit-basis
    cases, 4 arms/case):
        a100(TF32 default) vs cpu     routed SET changed  5/60 = 8.3%  [3.6, 18.1]
        a100 rerun vs a100 rerun      routed SET changed  0/60        <- not nondeterminism
        a100(TF32 OFF)  vs cpu        routed SET changed  0/60        <- THE FIX
        a100(TF32) vs a100(TF32 OFF)  routed SET changed  5/60 = 8.3% <- the same 5 cases
    The decomposition arm is what makes this a FIX and not a mitigation: with TF32 off the
    A100 reproduces the CPU routed set EXACTLY (min matched IoU 1.0, identical kept-voxel
    and component counts) on all 60/60. The cause is reduced COMPUTE PRECISION, not the
    device and not kernel reduction order.

    SCOPE — deliberately the trunk only, NOT the LM. TF32 is doing useful work in the
    Mistral path and no comparable defect was measured there: the LM consumes visual
    tokens, not an argmax, so it has no integer cliff to fall off.

    ★ WHICH KNOB ACTUALLY CARRIES THE DEFECT: `cudnn`. The trunk is convolutions end to
    end — including the 512->320 `feature_proj`, which is a `nn.Conv3d(..., kernel_size=1)`
    (model.py:564), NOT a matmul. `cuda.matmul.allow_tf32` is therefore belt-and-braces
    here and is flipped only so that a future Linear inside the trunk cannot reintroduce
    the defect through the one knob nobody was watching.
    ☠️ A USEFUL CONSEQUENCE, not a coincidence: the process defaults on this cluster are
    cudnn=True / matmul=False, so the banked dx caches — extracted under those defaults —
    sit in exactly the same numerical condition as an arm that sets BOTH to True. The
    before/after arms in scripts/nfdx_seg_tf32_verify.py are comparable to the caches for
    that reason, and the reason is checkable rather than assumed.

    ☠️ THESE ARE PROCESS-GLOBAL FLAGS, NOT THREAD-LOCAL. That is safe HERE and the reason
    is structural, not lucky: the only concurrency on the serving path is
    `nf_dx_arm.Struct43Job`, which is pure numpy/scipy on the CPU and touches no CUDA
    kernel, and the LM decode runs in the SAME thread strictly AFTER this block exits.
    A future caller that runs a CUDA matmul on another thread DURING a trunk forward
    would silently get fp32 for its own work — slower, never wrong. If this module ever
    grows a real concurrent CUDA path, this must become a per-op precision setting.

    ☠️ NO ENV VAR AND NO ARGUMENT. A knob that re-enables TF32 here would be an escape
    hatch that reintroduces a measured correctness defect for a latency gain nobody has
    asked for. Measurement code that needs the OLD behaviour monkeypatches this symbol
    explicitly and says so in its output (scripts/nfdx_seg_tf32_verify.py does exactly
    that for its BEFORE arm) — visible in the harness, absent from the service.
    """
    if not torch.cuda.is_available():
        # CPU-only: the flags do not exist to matter, and touching them on a build
        # without CUDA is a needless failure mode.
        yield
        return
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul


def seg_tf32_probe() -> dict:
    """Observe whether `seg_tf32_off()` ACTUALLY changes the flags IN THIS PROCESS.

    ☠️ WHY THIS IS A SHARED HELPER AND NOT THREE INLINE COPIES. Three cache builders
    (`bprime_trunk_extractor`, `f3p_llt_features`, `bprime_dump_probs`) each have to stamp
    the numeric condition their cache was written under. The docstring of `seg_tf32_off`
    already argues that four copies of a guard means the one that got missed is the one
    that serves; the same argument applies to the STAMP, because a cache whose stamp
    drifted from the guard is worse than an unstamped one — it is a false provenance
    record, and this repo has already paid for two of those (the leak-5 trunk under a
    matching file sha; `gpu_name`/`max_lesions` unstamped on the banked dx caches).

    ☠️ THIS MEASURES, IT DOES NOT ASSUME. `guard_changed_flags` is the answer to "was the
    guard load-bearing here?", and it can legitimately be False in two very different
    ways, which is why both are reported rather than collapsed into one boolean:
      * `cuda=False` — CPU-only build. `seg_tf32_off` is a deliberate no-op there and an
        unconditional assert would be a FALSE GATE.
      * `cuda=True` and the flags were already off — the guard changed nothing because
        something upstream had already turned TF32 off. The cache is still in the right
        numeric condition, but the stamp must not claim the guard is what put it there.
    ★ A probe that reports ABSENCE must first be able to report PRESENCE: on this cluster
    the process default is cudnn=True, so `outside` is the evidence that this probe CAN
    see TF32 on. If `outside` is already (False, False) on a CUDA build, the probe has
    not been shown to discriminate in this process and `inside_is_off` alone is trusted.
    """
    cuda = torch.cuda.is_available()
    outside = (bool(torch.backends.cudnn.allow_tf32),
               bool(torch.backends.cuda.matmul.allow_tf32))
    with seg_tf32_off():
        inside = (bool(torch.backends.cudnn.allow_tf32),
                  bool(torch.backends.cuda.matmul.allow_tf32))
    return {
        "cuda": cuda,
        "tf32_outside_guard": {"cudnn": outside[0], "matmul": outside[1]},
        "tf32_inside_guard": {"cudnn": inside[0], "matmul": inside[1]},
        # the condition the cache must be written under
        "inside_is_off": (inside == (False, False)),
        # was the guard load-bearing in THIS process (i.e. did it flip anything)?
        "guard_changed_flags": bool(cuda and inside != outside),
        # ★ could this probe have SEEN tf32 on at all? If not, a False above is
        #   uninformative rather than reassuring.
        "probe_could_observe_tf32_on": bool(cuda and (outside[0] or outside[1])),
    }


def seg_tf32_refuse_if_not_off(where: str) -> dict:
    """`seg_tf32_probe()` + the fail-closed assert. Returns the probe for stamping.

    ☠️ FAIL-CLOSED ON CUDA ONLY. On a CUDA build, if the guard does not leave BOTH flags
    off, the cache would not be in the served numeric condition and writing it would bank
    a file that looks fine and is wrong. On CPU there is nothing to turn off and refusing
    would be a gate that fires for a non-reason.
    """
    p = seg_tf32_probe()
    if p["cuda"] and not p["inside_is_off"]:
        raise RuntimeError(
            f"☠️ REFUSING ({where}): seg_tf32_off() did not turn both TF32 flags off on a "
            f"CUDA build (inside={p['tf32_inside_guard']}). The cache would not be in the "
            f"numeric condition the service segments in.")
    return p


class SegmentationBackbone(nn.Module):
    """MedNeXt 3D. Frozen after BraTS pretraining, exposes penultimate features.

    TRAINING:  caller passes GT segmentation mask (from BraTS .nii.gz).
    INFERENCE: caller passes None; this module runs MedNeXt forward to produce mask.
    This distinction is enforced by NeuroFusion.forward() via the `use_gt_seg` flag.
    """

    def __init__(self, cfg: NeuroFusionConfig, checkpoint_path: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._using_real_mednext = False
        self._feature_hook_output: torch.Tensor | None = None

        if checkpoint_path is not None:
            try:
                self.backbone = _load_real_mednext(cfg, checkpoint_path)
                self._using_real_mednext = True
                self._register_feature_hook()
                log.info(f"Loaded real MedNeXt from {checkpoint_path}")
            except Exception as e:
                log.warning(f"MedNeXt load failed ({e}); falling back to placeholder")
                self.backbone = _PlaceholderMedNeXt(cfg)
        else:
            log.info("No MedNeXt checkpoint provided; using placeholder backbone. "
                     "Run scripts/download_mednext.py to obtain pretrained weights.")
            self.backbone = _PlaceholderMedNeXt(cfg)

        # Training uses GT masks. Inference uses MedNeXt predictions.
        if cfg.freeze_backbone:
            # When using the MONAI wrap, only freeze the pretrained MedNeXt trunk —
            # the bottleneck→feature projection (added on top) must stay trainable
            # since it has no pretrained weights.
            if self._using_real_mednext and hasattr(self.backbone, "mednext"):
                for p in self.backbone.mednext.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False

    def _register_feature_hook(self) -> None:
        """Hook the last encoder conv to capture penultimate features."""
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv3d):
                last_conv = m
        if last_conv is not None:
            def _hook(module, inp, out):
                self._feature_hook_output = out
            last_conv.register_forward_hook(_hook)
            self._last_conv_layer = last_conv

    def forward(self, mri: torch.Tensor) -> dict[str, torch.Tensor]:
        self._feature_hook_output = None
        # ☠️ THE GUARD LIVES HERE, at the ONE choke point every trunk forward passes
        # through: `NeuroFusion.forward` (model.py -> `self.vision(mri)`), the
        # sliding-window `_SegNet` adapter in `nf_infer._segment_and_forward`, and
        # `nf_infer.dx_probe_forward` all call THIS method. Putting it at the call sites
        # instead would be four copies, and the one that got missed would be the one that
        # serves. `features` is captured by a hook INSIDE `self.backbone`, so the routed
        # mask AND the 320-ch feature map the router/Q-Former consume are both covered.
        with seg_tf32_off():
            out = self.backbone(mri)
        logits = out["logits"]
        features = out.get("features") if out.get("features") is not None else \
                   (self._feature_hook_output if self._feature_hook_output is not None else logits)
        return {"seg_logits": logits, "features": features}


class _PlaceholderMedNeXt(nn.Module):
    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.seg_head = nn.Conv3d(4, cfg.seg_n_classes, 1)
        self.feat_head = nn.Conv3d(4, cfg.seg_feature_dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.seg_head(x), "features": self.feat_head(x)}


class _MONAIMedNeXtWrap(nn.Module):
    """Wraps MONAI MedNeXt to expose the {logits, features} dict that
    SegmentationBackbone.forward() expects.

    - logits  : top-resolution segmentation head output  [B, n_classes, D, H, W]
    - features: chosen-stage output projected to cfg.seg_feature_dim
                  feature_stage="bottleneck": [B, F, D/16, H/16, W/16] (8^3 @ 128^3)
                  feature_stage="enc_hires" : [B, F, D/8,  H/8,  W/8 ] (16^3 @ 128^3)

    feature_stage gates the Q-Former feature SOURCE (visual-token-expansion
    experiment). "bottleneck" (DEFAULT) is byte-identical to all existing
    checkpoints. "enc_hires" hooks the last encoder stage (enc_stages[-1]) for
    8x the real-voxel context, with feature_proj in-channels 512->256 and the
    out-channels unchanged (320). The attribute the forward reads is _feat_out
    in BOTH cases; for the unfrozen per-stage-checkpointed path the stashed feat
    is set from enc_outputs[-1] (see _mednext_forward_checkpointed) so the hook
    and the checkpointed path agree on the same tensor.
    """

    def __init__(
        self,
        mednext: nn.Module,
        feat_ch: int,
        feature_dim: int,
        feature_stage: str = "bottleneck",
    ) -> None:
        super().__init__()
        self.mednext = mednext
        self.feature_stage = feature_stage
        self.feature_proj = nn.Conv3d(feat_ch, feature_dim, kernel_size=1)
        # Single attribute both code paths read from. Kept as _bottleneck_out too
        # so any external reader of the old name still works; they alias.
        self._feat_out: torch.Tensor | None = None
        if feature_stage == "bottleneck":
            self.mednext.bottleneck.register_forward_hook(self._stash_feat)
        elif feature_stage == "enc_hires":
            # Hook the LAST encoder stage (16^3 / 256-ch). In the frozen MONAI
            # stock forward this fires normally; in the unfrozen checkpointed
            # forward we ALSO set self.mednext._nf_enc_hires_out from
            # enc_outputs[-1] (mirrored in _mednext_forward_checkpointed) so both
            # paths agree. The _nf_enc_hires_enabled flag tells the checkpointed
            # forward to do that stash (bottleneck path never sets it -> stays
            # byte-identical).
            self.mednext.enc_stages[-1].register_forward_hook(self._stash_feat)
            self.mednext._nf_enc_hires_enabled = True
        else:
            raise ValueError(
                f"unknown feature_stage={feature_stage!r} (expected 'bottleneck' or 'enc_hires')"
            )

    # Back-compat alias: external code that referenced _bottleneck_out keeps working.
    @property
    def _bottleneck_out(self):
        return self._feat_out

    @_bottleneck_out.setter
    def _bottleneck_out(self, v):
        self._feat_out = v

    def _stash_feat(self, mod, inp, out):
        self._feat_out = out

    # ------------------------------------------------------------------ decoder taps
    # ★ Stage D prereq ③ (2026-08-06). OPT-IN, off by default; when unused nothing is
    # registered and every existing path is byte-identical.
    #
    # ☠️ WHY THESE EXIST. At the shipped bottleneck (8^3) a median MET lesion occupies
    # 1.23 feature voxels and 46.7% of MET cases have their WHOLE tumour inside less than
    # ONE voxel, so a masked pool there is arithmetically a pool over background. Geometry
    # (init_filters=32, blocks_down=[3,4,8,8], 128^3 input):
    #     dec_stages[0] -> 16^3 / 256ch      dec_stages[1] -> 32^3 / 128ch
    #     dec_stages[2] -> 64^3 /  64ch      dec_stages[3] -> 128^3 /  32ch
    # E1 probes [1] and [2]. No per-lesion descriptor claim is admissible until it has been
    # measured at a tap where a lesion is resolvable at all.
    #
    # ☠️☠️ A FORWARD HOOK, NOT AN EDIT TO `_mednext_forward_checkpointed`. That patched
    # forward is monkeypatched in ONLY by set_backbone_trainable(True); the FROZEN path --
    # which is what E1, every probe and the shipped service run -- uses MONAI's stock
    # forward. A tap written into the patched forward would therefore be DEAD CODE in
    # exactly the configuration it was built to measure, and would have reported nothing
    # while looking like it worked. That is the no-op class this codebase is convicted of
    # nine times. A module hook fires under BOTH forwards, which is the whole point.
    def enable_dec_taps(self, indices) -> None:
        """Register forward hooks on the named decoder stages. Idempotent."""
        idx = sorted({int(i) for i in indices})
        n = len(self.mednext.dec_stages)
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            raise ValueError(f"decoder stage(s) {bad} out of range (have {n})")
        if not hasattr(self, "_nf_dec_handles"):
            self._nf_dec_handles = {}
        self._nf_dec_out: dict[int, torch.Tensor] = {}
        for i in idx:
            if i in self._nf_dec_handles:
                continue

            def _mk(j):
                def _hook(mod, inp, out):
                    self._nf_dec_out[j] = out
                return _hook
            self._nf_dec_handles[i] = self.mednext.dec_stages[i].register_forward_hook(_mk(i))

    def dec_tap(self, i: int) -> torch.Tensor:
        """Return the stashed output of decoder stage `i`, or RAISE.

        ☠️ Raises rather than returning None. A tap that silently yields nothing would let
        a probe score a zero/absent descriptor and report it as a result -- the failure
        mode the enable_dec_taps docstring exists to prevent.
        """
        out = getattr(self, "_nf_dec_out", {}).get(i)
        if out is None:
            raise RuntimeError(
                f"decoder tap {i} did not fire. Call enable_dec_taps([{i}]) BEFORE the "
                f"forward pass, and read it AFTER. Registered: "
                f"{sorted(getattr(self, '_nf_dec_handles', {}))}")
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Plain pass-through so the chosen-stage forward-hook + feature projection
        # behave normally. When the trunk is UNFROZEN for Phase-2b adaptation,
        # NeuroFusion.set_backbone_trainable installs PER-STAGE gradient checkpointing
        # on self.mednext (see _mednext_forward_checkpointed) so the unfrozen 128^3
        # backbone's activations fit a 40 GB A100; that happens inside self.mednext(x)
        # below, transparently to this wrapper.
        self._feat_out = None
        out = self.mednext(x)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        if self.feature_stage == "enc_hires":
            # The checkpointed forward stashes enc_outputs[-1] HERE (its hook fires
            # under recompute too, which is harmless, but the explicit stash is the
            # authoritative one for the in-graph tensor).
            stashed = getattr(self.mednext, "_nf_enc_hires_out", None)
            if stashed is not None:
                self._feat_out = stashed
                self.mednext._nf_enc_hires_out = None
        assert self._feat_out is not None, (
            f"feature hook did not fire (feature_stage={self.feature_stage})"
        )
        features = self.feature_proj(self._feat_out)
        return {"logits": logits, "features": features}


def _ckpt_call(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a MedNeXt stage under gradient checkpointing when grad is enabled (training
    with the trunk unfrozen), else run it directly (eval / no_grad — nothing to save)."""
    if torch.is_grad_enabled():
        return _grad_checkpoint(mod, x, use_reentrant=False)
    return mod(x)


def _mednext_forward_checkpointed(self, x: torch.Tensor):
    """Drop-in replacement for monai.networks.nets.mednext.MedNeXt.forward that
    gradient-checkpoints each encoder/decoder STAGE + up/down block individually.

    WHY per-stage (not one big region): a single whole-trunk checkpoint does not help
    — its backward recomputes the entire forward AT ONCE, so peak activation memory is
    unchanged and an unfrozen 128^3 MedNeXt-L still OOMs a 40 GB A100 (observed in
    dec_stage). Per-stage checkpointing materializes ONE stage's activations at a time
    during backward, cutting peak to ~(largest stage) + persistent.

    Installed via monkeypatch by NeuroFusion.set_backbone_trainable(True) ONLY when the
    trunk is unfrozen, so the frozen path keeps MONAI's stock forward. stem / bottleneck
    / out heads are left un-checkpointed: they are small, and the bottleneck forward-hook
    (_MONAIMedNeXtWrap._stash_bottleneck) must fire under normal grad to feed feature_proj.
    Mirrors MONAI's forward exactly; module structure & state_dict keys are untouched.
    """
    x = self.stem(x)

    enc_outputs = []
    for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
        x = _ckpt_call(enc_stage, x)
        enc_outputs.append(x)
        x = _ckpt_call(down_block, x)

    # enc_hires feature source (visual-token expansion): stash the LAST encoder
    # stage output (16^3 / 256-ch) so _MONAIMedNeXtWrap.forward reads the same
    # in-graph tensor the bottleneck path's forward-hook would have. We only set
    # this when the wrap is in enc_hires mode (the attribute is created by the
    # hook registration); a getattr-guarded assignment keeps the bottleneck path
    # byte-identical (the attribute stays absent and the wrap's hook fires as before).
    if hasattr(self, "_nf_enc_hires_out") or getattr(self, "_nf_enc_hires_enabled", False):
        self._nf_enc_hires_out = enc_outputs[-1]

    x = self.bottleneck(x)

    if self.do_ds:
        ds_outputs = []

    for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
        if self.do_ds and i < len(self.out_blocks):
            ds_outputs.append(self.out_blocks[i](x))
        x = _ckpt_call(up_block, x)
        x = x + enc_outputs[-(i + 1)]
        x = _ckpt_call(dec_stage, x)

    x = self.out_0(x)

    if self.do_ds and self.training:
        return (x, *ds_outputs[::-1])
    else:
        return x


def _load_real_mednext(cfg: NeuroFusionConfig, checkpoint_path: str) -> nn.Module:
    """Load Phase 1 v5 MedNeXt-L (MONAI implementation) from a NeuroFusion checkpoint.

    Phase 1 v5 was trained with monai.networks.nets.mednext.MedNeXt; see
    scripts/pretrain_phase1_v5.py:build_mednext_l for the canonical config.
    Saved checkpoint layout: {"model": <state_dict>, "ema": ..., "args": ..., ...}.

    Returns a _MONAIMedNeXtWrap that yields {"logits", "features"} for the
    downstream Q-Former + routing pipeline.
    """
    import os
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"MedNeXt checkpoint not found: {checkpoint_path}")

    try:
        from monai.networks.nets.mednext import MedNeXt as _MONAIMedNeXt
    except ImportError as e:
        raise ImportError(
            f"MONAI MedNeXt not importable ({e}). pip install 'monai[all]'"
        ) from e

    base = _MONAIMedNeXt(
        spatial_dims=3,
        init_filters=32,
        in_channels=4,
        out_channels=cfg.seg_n_classes,
        encoder_expansion_ratio=[3, 4, 8, 8, 8],
        decoder_expansion_ratio=[8, 8, 8, 4, 3],
        bottleneck_expansion_ratio=8,
        kernel_size=3,
        deep_supervision=True,
        use_residual_connection=True,
        blocks_down=[3, 4, 8, 8],
        blocks_bottleneck=8,
        blocks_up=[8, 8, 4, 3],
        norm_type="group",
        global_resp_norm=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("network_weights", ckpt.get("state_dict", ckpt)))
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"MedNeXt: {len(missing)} missing keys")
    if unexpected:
        log.warning(f"MedNeXt: {len(unexpected)} unexpected keys")
    log.info(f"MedNeXt backbone loaded from {checkpoint_path}")

    BOTTLENECK_CH = 32 * (2 ** 4)  # init_filters × 2^(n_downsamples) = 512 for MedNeXt-L
    ENC_HIRES_CH = 32 * (2 ** 3)   # last encoder stage = init_filters × 2^(n_downsamples-1) = 256
    feature_stage = getattr(cfg, "feature_stage", "bottleneck")
    feat_ch = ENC_HIRES_CH if feature_stage == "enc_hires" else BOTTLENECK_CH
    return _MONAIMedNeXtWrap(
        base, feat_ch=feat_ch, feature_dim=cfg.seg_feature_dim, feature_stage=feature_stage
    )


# ==========================================================================
# 2. Lesion routing — 3D connected-components
# ==========================================================================


class LesionRouter(nn.Module):
    """3D connected-components + per-lesion feature crops. Emits voxel centroid (d, h, w)."""

    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        features: torch.Tensor,              # [B, C_feat, D', H', W']
        seg: torch.Tensor,                   # [B, D, H, W] (GT at train, predicted at eval)
        training: bool = True,
    ) -> dict[str, Any]:
        if self.cfg.use_global_pool:
            return self._forward_global_pool(features, seg)

        from scipy.ndimage import center_of_mass, label as cc_label

        B = features.shape[0]
        per_lesion_feats: list[torch.Tensor] = []
        per_lesion_gt: list[torch.Tensor] = []
        batch_idx: list[int] = []
        within_case_idx: list[int] = []
        centroids: list[tuple[float, float, float]] = []   # (d, h, w) in image space

        for b in range(B):
            seg_np = seg[b].detach().cpu().numpy()
            binary = (seg_np > 0).astype("uint8")
            labeled, n = cc_label(binary)

            volumes = [(i, int((labeled == i).sum())) for i in range(1, n + 1)]
            volumes.sort(key=lambda x: -x[1])

            kept = volumes[: self.cfg.max_lesions]
            for rank, (lesion_id, _) in enumerate(kept, start=1):
                mask = (labeled == lesion_id)
                cz, cy, cx = center_of_mass(mask)
                crop = self._crop_feature(features[b], (cz, cy, cx), seg_shape=seg_np.shape)
                per_lesion_feats.append(crop)
                # Per-lesion GT mask in the SAME feature-crop frame (for L_ground IoU).
                # Downsample this lesion's binary mask to bottleneck resolution, then
                # apply the same crop+pad bounds as _crop_feature did to the features.
                lesion_mask_img = torch.from_numpy(mask.astype("float32")).to(features.device)
                gt_feat = self._crop_gt_mask(lesion_mask_img, features[b].shape[-3:], (cz, cy, cx))
                per_lesion_gt.append(gt_feat)
                batch_idx.append(b)
                within_case_idx.append(rank)
                centroids.append((cz, cy, cx))

        # ☠️ B1 (2026-08-02): this branch FABRICATES a lesion, and every downstream
        # consumer has been unable to tell. It emits zeros at centroid (0,0,0) labelled
        # "lesion 1", so:
        #   * nf_infer.py's `n_lesions == 0` guard could NEVER fire (batch_idx is never
        #     empty) -- the "No tumor detected" path was unreachable dead code;
        #   * the hemisphere rule below sees d0 = 0.0 < midline and stamps "right";
        #   * the measurement module would emit a volume and a diameter for a lesion
        #     that does not exist, on exactly the cases where the model saw nothing.
        # We must NOT return an empty routing -- `lesion_features` feeds straight into
        # the Q-Former in the same forward, so torch.stack([]) raises and kills the
        # TRAIN path. Instead: keep the placeholder, and LABEL it, so callers can gate.
        # Shipped dummy counts on the canonical 174: GLI 0 / MEN 2 / MET 10.
        n_dummy = 0
        if not per_lesion_feats:
            # Each entry should be 4D [C, dz, dy, dx] so torch.stack gives 5D.
            # Old code used a leading (1,) which made the stack 6D and crashed
            # Q-Former.input_proj (conv3d wants 5D). Only fires when predicted
            # seg has no positive voxels — i.e. validation on MedNeXt argmax.
            dummy = torch.zeros((features.shape[1], *self.cfg.crop_size), device=features.device)
            per_lesion_feats = [dummy]
            per_lesion_gt = [torch.zeros(self.cfg.crop_size, device=features.device)]
            batch_idx = [0]
            within_case_idx = [1]
            centroids = [(0.0, 0.0, 0.0)]
            n_dummy = 1

        is_dummy = torch.zeros(len(per_lesion_feats), dtype=torch.bool,
                               device=features.device)
        if n_dummy:
            is_dummy[:] = True

        return {
            "lesion_features": torch.stack(per_lesion_feats),
            "lesion_gt_feat": torch.stack(per_lesion_gt),   # [N_total, dz, dy, dx]
            "batch_idx": torch.tensor(batch_idx, device=features.device),
            "lesion_idx": torch.tensor(within_case_idx, device=features.device),
            "centroids": torch.tensor(centroids, device=features.device),
            # B1: True where the row is a FABRICATED placeholder, not a routed lesion.
            # Never infer this from `torch.any(lesion_features)` -- a descriptor's
            # geometry channels are non-zero by construction for an empty mask, so that
            # test breaks silently the moment the connector changes (§CORE-T.7 trap 2).
            "is_dummy": is_dummy,
        }

    def _forward_global_pool(
        self,
        features: torch.Tensor,   # [B, C_feat, Df, Hf, Wf] — frozen MedNeXt bottleneck features
        seg: torch.Tensor,        # [B, D, H, W] (GT at train, predicted argmax at eval)
    ) -> dict[str, Any]:
        """GLOBAL-POOL ablation (reviewer-requested). Bypass per-lesion routing
        entirely: produce exactly ONE "lesion" per case whose feature is the
        adaptive_avg_pool3d of the frozen bottleneck features, masked to the
        segmentation foreground (any tumor class, argmax>0), pooled over the
        whole volume to cfg.crop_size — the same crop shape the per-lesion
        router emits, so all downstream modules run unchanged with N=1.

        Supervision: the single global prediction is supervised against the
        DOMINANT (largest-volume) GT lesion's per-lesion fields. We emit
        lesion_idx=1; train.build_field_targets indexes report.lesions[0],
        which (schema.py: lesions ordered largest→smallest by volume) is the
        dominant lesion. Case-level fields (hemisphere, overall_mass_effect)
        are unchanged — they live in the generative target / prefill, not here.

        L_ground (per-lesion mask IoU) is disabled: there are no per-lesion
        crops to ground, so we emit a zeros lesion_gt_feat. compute_grounding_loss
        on an all-zero GT mask yields a degenerate (and meaningless) IoU; to keep
        the ablation clean the grounding signal is zeroed at the source here.
        """
        B = features.shape[0]
        C_feat = features.shape[1]
        device = features.device
        dz, dy, dx = self.cfg.crop_size
        Df, Hf, Wf = features.shape[-3:]

        per_lesion_feats: list[torch.Tensor] = []
        batch_idx: list[int] = []
        within_case_idx: list[int] = []
        centroids: list[tuple[float, float, float]] = []
        dummy_flags: list[bool] = []   # B1: True where this case had NO foreground

        for b in range(B):
            # Foreground mask (any tumor class) at image resolution, downsampled
            # to feature resolution via max-pool (preserves presence).
            fg_img = (seg[b] > 0).float()
            fg_feat = F.adaptive_max_pool3d(
                fg_img.unsqueeze(0).unsqueeze(0), (Df, Hf, Wf)
            ).squeeze(0).squeeze(0)                                  # [Df, Hf, Wf]

            feat_b = features[b]                                      # [C_feat, Df, Hf, Wf]
            if fg_feat.sum() > 0:
                # Foreground-masked global average over the whole volume:
                # sum(feature * mask) / sum(mask), broadcast over channels.
                m = fg_feat.unsqueeze(0)                              # [1, Df, Hf, Wf]
                masked_sum = (feat_b * m).sum(dim=(-3, -2, -1))       # [C_feat]
                pooled_vec = masked_sum / fg_feat.sum().clamp(min=1e-6)
            else:
                # No foreground (e.g. predicted seg with no positive voxels):
                # fall back to a plain global average so the pipeline still runs.
                pooled_vec = feat_b.mean(dim=(-3, -2, -1))            # [C_feat]
            # B1: flag the no-foreground case. NOTE this branch pools the WHOLE
            # volume, so the feature is real (brain, not tumour) rather than zeros —
            # which is why `torch.any(features)` cannot detect it and an explicit
            # flag is required.
            dummy_flags.append(bool(fg_feat.sum() == 0))

            # Broadcast the single pooled vector to the crop_size grid the
            # per-lesion router would have emitted, so Q-Former.input_proj
            # (Conv3d) and everything downstream see an identical tensor shape.
            pooled = pooled_vec.view(C_feat, 1, 1, 1).expand(C_feat, dz, dy, dx).contiguous()
            per_lesion_feats.append(pooled)

            # Centroid of the foreground at IMAGE resolution (matches the
            # per-lesion router's centroid convention so the hemisphere prefill
            # in generate() — which reads centroids[:, 0] — still works).
            from scipy.ndimage import center_of_mass
            seg_np = seg[b].detach().cpu().numpy()
            fg_np = (seg_np > 0)
            if fg_np.any():
                cz, cy, cx = center_of_mass(fg_np.astype("uint8"))
            else:
                cz = cy = cx = 0.0
            centroids.append((float(cz), float(cy), float(cx)))

            batch_idx.append(b)
            within_case_idx.append(1)   # single dominant "lesion" per case

        return {
            "lesion_features": torch.stack(per_lesion_feats),                  # [B, C_feat, dz, dy, dx]
            # L_ground disabled in global-pool: zeros GT mask (no per-lesion crop).
            "lesion_gt_feat": torch.zeros((B, dz, dy, dx), device=device),
            "batch_idx": torch.tensor(batch_idx, device=device),
            "lesion_idx": torch.tensor(within_case_idx, device=device),
            "centroids": torch.tensor(centroids, device=device),
            "is_dummy": torch.tensor(dummy_flags, dtype=torch.bool, device=device),
        }

    def _forward_global_pool(
        self,
        features: torch.Tensor,   # [B, C_feat, Df, Hf, Wf] — frozen MedNeXt bottleneck features
        seg: torch.Tensor,        # [B, D, H, W] (GT at train, predicted argmax at eval)
    ) -> dict[str, Any]:
        """GLOBAL-POOL ablation (reviewer-requested). Bypass per-lesion routing
        entirely: produce exactly ONE "lesion" per case whose feature is the
        adaptive_avg_pool3d of the frozen bottleneck features, masked to the
        segmentation foreground (any tumor class, argmax>0), pooled over the
        whole volume to cfg.crop_size — the same crop shape the per-lesion
        router emits, so all downstream modules run unchanged with N=1.

        Supervision: the single global prediction is supervised against the
        DOMINANT (largest-volume) GT lesion's per-lesion fields. We emit
        lesion_idx=1; train.build_field_targets indexes report.lesions[0],
        which (schema.py: lesions ordered largest→smallest by volume) is the
        dominant lesion. Case-level fields (hemisphere, overall_mass_effect)
        are unchanged — they live in the generative target / prefill, not here.

        L_ground (per-lesion mask IoU) is disabled: there are no per-lesion
        crops to ground, so we emit a zeros lesion_gt_feat. compute_grounding_loss
        on an all-zero GT mask yields a degenerate (and meaningless) IoU; to keep
        the ablation clean the grounding signal is zeroed at the source here.
        """
        B = features.shape[0]
        C_feat = features.shape[1]
        device = features.device
        dz, dy, dx = self.cfg.crop_size
        Df, Hf, Wf = features.shape[-3:]

        per_lesion_feats: list[torch.Tensor] = []
        batch_idx: list[int] = []
        within_case_idx: list[int] = []
        centroids: list[tuple[float, float, float]] = []

        for b in range(B):
            # Foreground mask (any tumor class) at image resolution, downsampled
            # to feature resolution via max-pool (preserves presence).
            fg_img = (seg[b] > 0).float()
            fg_feat = F.adaptive_max_pool3d(
                fg_img.unsqueeze(0).unsqueeze(0), (Df, Hf, Wf)
            ).squeeze(0).squeeze(0)                                  # [Df, Hf, Wf]

            feat_b = features[b]                                      # [C_feat, Df, Hf, Wf]
            if fg_feat.sum() > 0:
                # Foreground-masked global average over the whole volume:
                # sum(feature * mask) / sum(mask), broadcast over channels.
                m = fg_feat.unsqueeze(0)                              # [1, Df, Hf, Wf]
                masked_sum = (feat_b * m).sum(dim=(-3, -2, -1))       # [C_feat]
                pooled_vec = masked_sum / fg_feat.sum().clamp(min=1e-6)
            else:
                # No foreground (e.g. predicted seg with no positive voxels):
                # fall back to a plain global average so the pipeline still runs.
                pooled_vec = feat_b.mean(dim=(-3, -2, -1))            # [C_feat]

            # Broadcast the single pooled vector to the crop_size grid the
            # per-lesion router would have emitted, so Q-Former.input_proj
            # (Conv3d) and everything downstream see an identical tensor shape.
            pooled = pooled_vec.view(C_feat, 1, 1, 1).expand(C_feat, dz, dy, dx).contiguous()
            per_lesion_feats.append(pooled)

            # Centroid of the foreground at IMAGE resolution (matches the
            # per-lesion router's centroid convention so the hemisphere prefill
            # in generate() — which reads centroids[:, 0] — still works).
            from scipy.ndimage import center_of_mass
            seg_np = seg[b].detach().cpu().numpy()
            fg_np = (seg_np > 0)
            if fg_np.any():
                cz, cy, cx = center_of_mass(fg_np.astype("uint8"))
            else:
                cz = cy = cx = 0.0
            centroids.append((float(cz), float(cy), float(cx)))

            batch_idx.append(b)
            within_case_idx.append(1)   # single dominant "lesion" per case

        return {
            "lesion_features": torch.stack(per_lesion_feats),                  # [B, C_feat, dz, dy, dx]
            # L_ground disabled in global-pool: zeros GT mask (no per-lesion crop).
            "lesion_gt_feat": torch.zeros((B, dz, dy, dx), device=device),
            "batch_idx": torch.tensor(batch_idx, device=device),
            "lesion_idx": torch.tensor(within_case_idx, device=device),
            "centroids": torch.tensor(centroids, device=device),
        }

    def _crop_gt_mask(
        self,
        lesion_mask_img: torch.Tensor,    # [D, H, W] float, single-lesion binary at image res
        feat_shape: tuple[int, int, int],  # (Df, Hf, Wf) — MedNeXt bottleneck shape
        center_img: tuple[float, float, float],
    ) -> torch.Tensor:
        """Downsample a single-lesion mask to feature resolution and apply the same
        crop+pad bounds as _crop_feature, so L_ground IoU is spatially aligned."""
        dz, dy, dx = self.cfg.crop_size
        Df, Hf, Wf = feat_shape
        # max-pool image-space mask to feature resolution (preserves lesion presence)
        m_feat = F.adaptive_max_pool3d(
            lesion_mask_img.unsqueeze(0).unsqueeze(0), (Df, Hf, Wf)
        ).squeeze(0).squeeze(0)
        Di, Hi, Wi = lesion_mask_img.shape
        cz = int(center_img[0] / Di * Df)
        cy = int(center_img[1] / Hi * Hf)
        cx = int(center_img[2] / Wi * Wf)
        z0, z1 = max(0, cz - dz // 2), min(Df, cz - dz // 2 + dz)
        y0, y1 = max(0, cy - dy // 2), min(Hf, cy - dy // 2 + dy)
        x0, x1 = max(0, cx - dx // 2), min(Wf, cx - dx // 2 + dx)
        patch = m_feat[z0:z1, y0:y1, x0:x1]
        patch = F.pad(patch, (0, dx - patch.shape[-1], 0, dy - patch.shape[-2], 0, dz - patch.shape[-3]))
        return patch

    def _crop_feature(
        self,
        feat: torch.Tensor,
        center_img: tuple[float, float, float],
        seg_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        dz, dy, dx = self.cfg.crop_size
        Df, Hf, Wf = feat.shape[-3:]
        Di, Hi, Wi = seg_shape
        cz = int(center_img[0] / Di * Df)
        cy = int(center_img[1] / Hi * Hf)
        cx = int(center_img[2] / Wi * Wf)
        z0, z1 = max(0, cz - dz // 2), min(Df, cz - dz // 2 + dz)
        y0, y1 = max(0, cy - dy // 2), min(Hf, cy - dy // 2 + dy)
        x0, x1 = max(0, cx - dx // 2), min(Wf, cx - dx // 2 + dx)
        patch = feat[:, z0:z1, y0:y1, x0:x1]
        patch = F.pad(patch, (0, dx - patch.shape[-1], 0, dy - patch.shape[-2], 0, dz - patch.shape[-3]))
        return patch


# ==========================================================================
# 3. Q-Former with factorized 3D positional embeddings
# ==========================================================================


class PerLesionQFormer(nn.Module):
    """32 queries per lesion + lesion-index embedding + optional factorized 3D PE."""

    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.qformer_hidden_dim

        self.queries = nn.Parameter(torch.randn(cfg.n_queries_per_lesion, D) * 0.02)
        self.lesion_idx_embed = nn.Embedding(cfg.max_lesions + 1, D)
        self.input_proj = nn.Conv3d(cfg.seg_feature_dim, D, 1)

        # Factorized 3D positional embeddings (Mechanism 4 in diagram 3)
        if cfg.use_3d_pe_on_queries:
            # 256 buckets per axis is plenty for 48^3 crops even with bilinear
            self.pe_d = nn.Embedding(256, D)
            self.pe_h = nn.Embedding(256, D)
            self.pe_w = nn.Embedding(256, D)
        else:
            self.pe_d = self.pe_h = self.pe_w = None

        if cfg.use_custom_qformer:
            # Custom decoder exposes per-layer cross-attention weights for L_ground IoU.
            from qformer_custom import CustomQFormerDecoder
            self.decoder = CustomQFormerDecoder(
                d_model=D, nhead=cfg.qformer_n_heads,
                num_layers=cfg.qformer_n_layers,
                dim_feedforward=D * 4, dropout=0.1, norm_first=True,
            )
            self._is_custom_decoder = True
        else:
            layer = nn.TransformerDecoderLayer(
                d_model=D, nhead=cfg.qformer_n_heads, dim_feedforward=D * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.qformer_n_layers)
            self._is_custom_decoder = False

        self.output_proj = nn.Linear(D, cfg.lm_hidden_dim)
        self._last_attn: torch.Tensor | None = None
        self._feat_dhw: tuple[int, int, int] | None = None   # used by L_ground reshape

    def forward(
        self,
        lesion_features: torch.Tensor,       # [N_total, C_feat, d, h, w]
        lesion_idx: torch.Tensor,            # [N_total]
        centroids: torch.Tensor | None = None,  # [N_total, 3] voxel (d, h, w)
    ) -> torch.Tensor:
        n = lesion_features.shape[0]
        x = self.input_proj(lesion_features)                # [N, D, d, h, w]
        # Cache feature-grid dims so L_ground can reshape the flat cross-attn back to 3D
        self._feat_dhw = tuple(int(v) for v in x.shape[-3:])
        x = x.flatten(2).transpose(1, 2)                    # [N, d*h*w, D]

        q = self.queries.unsqueeze(0).expand(n, -1, -1)      # [N, 32, D]
        q = q + self.lesion_idx_embed(lesion_idx).unsqueeze(1)

        if self.pe_d is not None and centroids is not None:
            # Add factorized 3D PE based on lesion centroid (shared across all 32 queries of that lesion)
            d_idx = centroids[:, 0].long().clamp(0, 255)
            h_idx = centroids[:, 1].long().clamp(0, 255)
            w_idx = centroids[:, 2].long().clamp(0, 255)
            pe = self.pe_d(d_idx) + self.pe_h(h_idx) + self.pe_w(w_idx)  # [N, D]
            q = q + pe.unsqueeze(1)                                        # broadcast over queries

        out = self.decoder(q, x)                              # [N, 32, D]
        if self._is_custom_decoder:
            self._last_attn = self.decoder.last_cross_attn    # [N, n_q, d*h*w] (head-mean)
        return self.output_proj(out)                          # [N, 32, D_lm]


# ==========================================================================
# 4. Field classification head (with heuristic-text rendering)
# ==========================================================================


class FieldClassificationHead(nn.Module):
    """Per-lesion 10-head classifier. Also renders predictions as text prefix for LM."""

    # Inverse enum maps (index -> string) — mirrors schema.py. Keep in sync!
    ENUM_STRINGS: dict[str, list[str]] = {
        "location": ["frontal", "temporal", "parietal", "occipital", "insular", "cerebellar",
                      "brainstem", "parieto-temporal", "temporo-parietal", "fronto-parietal",
                      "fronto-temporal", "temporo-occipital", "parieto-occipital", "thalamic",
                      "basal-ganglia", "corpus-callosum"],
        "composition": ["solid", "cystic", "necrotic", "necrotic-cystic", "hemorrhagic"],
        "enhancement_pattern": ["none", "homogeneous", "heterogeneous", "ring",
                                 "thick-walled-ring", "nodular", "peripheral-nodular"],
        "edema_severity": ["none", "mild", "moderate", "marked"],
        "mass_effect": ["none", "present"],
        "midline_shift": ["none", "present"],
        "ventricular_compression": ["none", "present"],
        "brainstem_compression": ["none", "present"],
        "midbrain_compression": ["none", "present"],
        "axis_shift": ["none", "contralateral", "ipsilateral"],
    }

    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pool_mode = getattr(cfg, "field_head_pool", "mean")
        self.use_global = getattr(cfg, "field_head_global_token", False)
        if self.pool_mode == "attention":
            # learned query attention-pools the (lesion [+ global]) tokens; recovers the
            # head-vs-connector-output gap that mean-pool leaves (connector probe).
            self.attn_query = nn.Parameter(torch.randn(cfg.lm_hidden_dim) * 0.02)
            self.attn_key = nn.Linear(cfg.lm_hidden_dim, cfg.lm_hidden_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
        if self.use_global:
            # bottleneck-mean global-context token -> D_lm, appended per lesion.
            self.global_proj = nn.Linear(cfg.seg_feature_dim, cfg.lm_hidden_dim)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(cfg.lm_hidden_dim, n_classes) for name, n_classes in cfg.field_heads.items()}
        )

    def forward(self, lesion_tokens: torch.Tensor,
                global_feat: torch.Tensor | None = None,
                batch_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Args: lesion_tokens [N_total, 32, D_lm]; optional global_feat [B, seg_feature_dim]
        + batch_idx [N_total] (mapping each lesion to its case) for the global-context token.
        Returns: {field: [N_total, n_classes] logits}."""
        tokens = lesion_tokens
        if self.use_global and global_feat is not None and batch_idx is not None:
            g = self.global_proj(global_feat)                       # [B, D_lm]
            g_per = g[batch_idx.long()].unsqueeze(1)                # [N_total, 1, D_lm]
            tokens = torch.cat([lesion_tokens, g_per], dim=1)       # [N_total, 33, D_lm]
        if self.pool_mode == "attention":
            k = self.attn_key(tokens)                               # [N, T, D_lm]
            scores = (k @ self.attn_query) / (self.cfg.lm_hidden_dim ** 0.5)  # [N, T]
            attn = torch.softmax(scores, dim=1).unsqueeze(-1)       # [N, T, 1]
            pooled = (attn * tokens).sum(dim=1)                     # [N, D_lm]
        else:
            pooled = self.pool(tokens.transpose(1, 2)).squeeze(-1)
        return {name: head(pooled) for name, head in self.heads.items()}

    def render_heuristic_text(
        self,
        field_logits: dict[str, torch.Tensor],
        batch_idx: torch.Tensor,
        lesion_idx: torch.Tensor,
    ) -> list[str]:
        """Render per-case classifier predictions as a text prefix for the LM.

        Only fields with softmax-max >= cfg.classifier_confidence_threshold are included,
        so that low-confidence heuristics don't poison the LM prompt.
        """
        batch_size = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1
        out: list[list[str]] = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for field_name, logits in field_logits.items():
                if field_name not in self.ENUM_STRINGS:
                    continue
                probs = F.softmax(logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                for i in range(logits.shape[0]):
                    if conf[i].item() < self.cfg.classifier_confidence_threshold:
                        continue
                    b = int(batch_idx[i].item())
                    L = int(lesion_idx[i].item())
                    val = self.ENUM_STRINGS[field_name][int(idx[i].item())]
                    out[b].append(f"lesion_{L}.{field_name}={val}")

        return [" ; ".join(lst) if lst else "(no high-confidence fields)" for lst in out]


def field_logits_to_per_lesion_softmax(
    field_logits: dict[str, torch.Tensor],
    batch_idx: torch.Tensor,
    lesion_idx: torch.Tensor,
    enum_strings: dict[str, list[str]] = FieldClassificationHead.ENUM_STRINGS,
) -> list[dict[int, dict[str, dict[str, float]]]]:
    """Convert FieldClassificationHead's flat per-lesion logits into the nested
    {batch_slot: {lesion_index: {field_name: {class_str: prob}}}} structure that
    xgrammar_decoder.PerFieldLogitFuser.register_case expects.

    Args:
        field_logits: {field_name: [N_lesions_total, n_classes]}.
        batch_idx:    [N_lesions_total] — which batch slot each lesion belongs to.
        lesion_idx:   [N_lesions_total] — 1-based lesion index within its case.

    Returns:
        list of length B; element b is {lesion_index_int: {field_name: {class: prob}}}.
        The lesion-index here matches the lesion_index emitted in the generated JSON,
        which the fuser uses to pull the right per-lesion distribution at decode time.
    """
    B = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1
    out: list[dict[int, dict[str, dict[str, float]]]] = [dict() for _ in range(B)]

    if not field_logits:
        return out

    # Softmax each field's logits once across all lesions.
    softmaxed: dict[str, torch.Tensor] = {
        name: F.softmax(logits, dim=-1).detach().float().cpu()
        for name, logits in field_logits.items()
    }

    bi_list = batch_idx.detach().cpu().tolist()
    li_list = lesion_idx.detach().cpu().tolist()
    for i, (b, L) in enumerate(zip(bi_list, li_list)):
        b = int(b)
        L = int(L)
        per_field: dict[str, dict[str, float]] = {}
        for fname, probs in softmaxed.items():
            classes = enum_strings.get(fname)
            if classes is None:
                continue
            row = probs[i].tolist()
            # Trim to len(classes) — head may be sized for n_classes >= len(classes)
            # (e.g. location has a learned 17-way head with the 17th = OTHER fallback).
            d = {c: float(row[j]) for j, c in enumerate(classes) if j < len(row)}
            per_field[fname] = d
        out[b][L] = per_field
    return out



# ==========================================================================
# 5. LM wrapper with 4-axis M-RoPE retrofit
# ==========================================================================


# Draft-then-commit ("head-conditioned CoT") delimiter. Single source of truth:
# train.build_cot_target_strings joins the draft prose and the JSON with this
# literal, and generate_draft_commit / scripts/predict_draft_commit.py split on
# "[COMMIT]" to recover the two stages. A plain-text marker (not a special token)
# so no tokenizer/vocab change is needed.
COT_COMMIT_MARKER = "\n[COMMIT]\n"


class LMWithMRope4D(nn.Module):
    """Report-generation LM wrapper. Two config-gated backbones (cfg.lm_family):

    "medgemma" (default): MedGemma 1.5 4B + QLoRA + 4-axis M-RoPE retrofit.
      Loading order:
        1. AutoModelForCausalLM + 4-bit BnB quantisation  (bitsandbytes)
        2. PEFT LoraConfig (r=16, alpha=32, target q/k/v/o)
        3. patch_medgemma_with_mrope_4d() → monkey-patches every attention layer's
           rotary computation with 4-axis (t, d, h, w) position encoding
        4. Tokenizer: left-padded for batch generation
      Frozen: base MedGemma weights. Trained: LoRA + M-RoPE axis tables.

    "mistral" (LM-swap, 2026-06): LLaVA-Med's Mistral-7B-Instruct-v0.2 LM
      (vision stripped, extracted to a local ckpt) + 4-bit QLoRA + STANDARD 1D
      RoPE (NO M-RoPE patch — measured benefit negligible). Hidden 4096; the
      Q-Former->LM projection auto-resizes via cfg.__post_init__. See
      _load_real_mistral. The class name (…MRope4D) is retained for backward
      compatibility; the Mistral path simply leaves the M-RoPE machinery unused.

    Both paths fall back to _PlaceholderLM if the LM is unavailable (no GPU /
    missing weights), so CPU shape-checks still run end-to-end.
    """

    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._using_real_lm = False
        self.tokenizer = None

        # Head-conditioned CoT v2 cohort-anchor (default OFF => None => byte-identical).
        # A tiny learned embedding table indexed by the Stage-1 cohort probe's
        # prediction (none/GLI/MEN/MET); prepended as the first soft prefix token in
        # the draft->commit forward + generate_draft_commit. Created HERE (before the
        # family branches) so it exists on every path — the mistral early-`return`,
        # the medgemma path, and the placeholder fallback. When disabled it is None
        # and no parameter is added, so the optimizer/state_dict/decode are unchanged.
        if getattr(cfg, "use_cohort_anchor", False):
            self.cohort_anchor = nn.Embedding(cfg.n_cohort_anchor, cfg.lm_hidden_dim)
            nn.init.normal_(self.cohort_anchor.weight, mean=0.0, std=0.02)
        else:
            self.cohort_anchor = None

        # Rung-3 dx-distill probe (default OFF => None => no param, no state_dict key).
        # Reads the last-prefix hidden [B, D_lm] -> 4 dx logits [GLI,MEN,MET,OTHER];
        # trained (via the LoRA optimizer group: it is under model.lm + requires_grad)
        # to match the frozen DiagnosisHead. NOT used at inference (LM free-generates dd0).
        if getattr(cfg, "use_dx_distill", False):
            self.dx_probe = nn.Linear(cfg.lm_hidden_dim, 4)
            nn.init.normal_(self.dx_probe.weight, std=0.02)
            nn.init.zeros_(self.dx_probe.bias)
        else:
            self.dx_probe = None

        # LM-swap branch (2026-06): Mistral path is config-gated and additive.
        # It loads LLaVA-Med's Mistral LM in 4-bit QLoRA with STANDARD 1D RoPE
        # (no M-RoPE patch). The MedGemma path below is untouched.
        if cfg.lm_family == "mistral":
            self.mrope_table = None      # Mistral uses 1D RoPE; no 4-axis table
            self._mrope_cfg = None
            try:
                self._load_real_mistral(cfg)
                self._using_real_lm = True
            except Exception as e:
                log.warning(f"Mistral load failed ({e}); using placeholder LM. "
                            "Ensure GPU + extracted Mistral ckpt are available.")
                self.lm = _PlaceholderLM(cfg)
            return

        if cfg.use_4d_mrope:
            # cfg.mrope_rotary_dim / axis_channels are HINTS only — the patcher derives the
            # actual rotary_dim and inv_freq from HF's per-layer rotary_emb modules so that
            # sliding_attention (base 10K) and full_attention (base 1M, linear×8.0) get
            # bit-correct rotations. Legacy axis_channels=(16,16,16,16) is auto-promoted
            # to full-rotary inside patch_medgemma_with_mrope_4d.
            mrope_cfg = MRope4DConfig(
                head_dim=cfg.lm_head_dim,
                rotary_dim=cfg.mrope_rotary_dim,
                axis_channels=cfg.mrope_axis_channels,
                low_freq_axis="none",
            )
            self.mrope_table = None        # built by patch_medgemma_with_mrope_4d
            self._mrope_cfg = mrope_cfg
        else:
            self.mrope_table = None
            self._mrope_cfg = None

        try:
            self._load_real_medgemma(cfg)
            self._using_real_lm = True
        except Exception as e:
            log.warning(f"MedGemma load failed ({e}); using placeholder LM. "
                        "Ensure GPU + HuggingFace token are available for full training.")
            self.lm = _PlaceholderLM(cfg)

    def _load_real_medgemma(self, cfg: NeuroFusionConfig) -> None:
        """Load MedGemma 4B with 4-bit QLoRA and retrofit 4-axis M-RoPE."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from mrope_patch import patch_medgemma_with_mrope_4d

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Prefer instruction-tuned variant; fall back to base pretraining checkpoint
        model_names = [cfg.lm_name, "google/medgemma-4b-it", "google/medgemma-4b-pt"]
        base = None
        for name in model_names:
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
                log.info(f"MedGemma loaded: {name}")
                break
            except Exception as exc:
                log.debug(f"  tried {name}: {exc}")

        if base is None:
            raise RuntimeError(
                "Could not load any MedGemma variant. "
                "Run: huggingface-cli login  (token must accept google/medgemma usage policy)"
            )

        # Prepare for 4-bit + LoRA training:
        #  - freezes all base params
        #  - casts layer norms to fp32 (numerical stability under 4-bit matmul)
        #  - casts lm_head output to fp32 (removes the bf16/fp32 mismatch in generate)
        #  - enables gradient checkpointing (essential — saves ~30 GB activation memory
        #    on 4B MedGemma at batch 1, target_text_ids ~1k tokens)
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.lm = get_peft_model(base, lora_cfg)

        # Retrofit 4-axis M-RoPE into every attention layer.
        # patch_medgemma_with_mrope_4d builds per-layer-type tables (sliding vs full) from
        # HF's rotary_emb modules and wraps each Gemma3DecoderLayer.forward so the dispatcher
        # picks the right table at call time. verify_identity_reduction runs in train.py --phase 2a.
        if self._mrope_cfg is not None:
            self.lm, self.mrope_table = patch_medgemma_with_mrope_4d(
                self.lm, self._mrope_cfg, verify=False,
            )

        # Left-padding tokenizer for batch generation
        for tok_name in [cfg.lm_name, "google/medgemma-4b-it"]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
                break
            except Exception:
                pass
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"MedGemma ready — trainable params: "
                 f"{sum(p.numel() for p in self.lm.parameters() if p.requires_grad):,}")

    def _load_real_mistral(self, cfg: NeuroFusionConfig) -> None:
        """Load LLaVA-Med's Mistral-7B LM in 4-bit QLoRA (LM-swap experiment).

        Differences vs the MedGemma path (all deliberate simplifications):
          - NO 4-axis M-RoPE retrofit. Mistral keeps its standard 1D RoPE
            (rope_theta=1e6). Our own measurement found the M-RoPE benefit
            negligible (project_mrope_conditional_benefit_2026-05-26), so the
            Mistral path skips the patch entirely. self.lm therefore has NO
            `_mrope_dispatcher`; every M-RoPE stash call in forward/generate is
            already guarded by `hasattr(self.lm, "_mrope_dispatcher")` and is a
            no-op here.
          - Weights come from the LOCAL extracted checkpoint (vision stripped),
            loaded into a vanilla MistralForCausalLM(config), then wrapped with
            BnB 4-bit + LoRA. We instantiate the fp16 model from the extracted
            state_dict and re-quantize via from_pretrained on a temp dir... but
            to avoid a disk round-trip we load the state_dict into a quantizable
            model directly (see below).

        get_input_embeddings() resolves to model.model.embed_tokens for a PEFT-
        wrapped MistralForCausalLM — same API the inputs_embeds path already uses.
        """
        from pathlib import Path

        import torch
        from safetensors.torch import load_file
        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
            MistralConfig,
            MistralForCausalLM,
        )
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        # ---- NF 1.3 Marlin / fp16 anchor: pre-merged weights, NO bnb, NO LoRA -------------
        # ☠️ NO get_peft_model HERE, AND THAT IS THE WHOLE POINT. The adapter is already baked
        # into these weights by scripts/merge_lora_to_fp16.py. Wrapping again would (a) add a
        # second, untrained adapter on top of an already-merged one and (b) rename every
        # module to PEFT's `base_model.model.*` layout, so a later load of the NF checkpoint
        # would appear to match and would re-apply the SAME update a second time. The caller
        # must skip `lm.*` on checkpoint load -- gated by cfg._lm_weights_are_external().
        if cfg._lm_weights_are_external():
            qdir = Path(cfg.lm_quant_ckpt)
            if not qdir.is_dir():
                raise RuntimeError(f"lm_quant_ckpt not found: {qdir}")
            if not any((qdir / n).is_file() for n in
                       ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")):
                raise RuntimeError(f"lm_quant_ckpt has no weights file: {qdir}")
            log.info(f"Mistral LM: lm_quant={cfg.lm_quant} from {qdir} "
                     f"(pre-merged; no bitsandbytes, no LoRA wrapper)")
            base = MistralForCausalLM.from_pretrained(
                str(qdir), dtype=torch.bfloat16, device_map="auto")
            # Weights are frozen for both arms: there is nothing trainable in a merged or
            # quantised LM, and leaving them trainable would let a stray optimiser touch them.
            for p in base.parameters():
                p.requires_grad_(False)
            self.lm = base
            for tok_name in [str(qdir), cfg.mistral_lm_ckpt,
                             "mistralai/Mistral-7B-Instruct-v0.2"]:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)
                    break
                except Exception:
                    pass
            if self.tokenizer is not None:
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            n_tr = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
            log.info(f"Mistral ready (lm_quant={cfg.lm_quant}) — trainable params: {n_tr:,}")
            return

        ckpt_dir = Path(cfg.mistral_lm_ckpt)
        if not (ckpt_dir / "lm_state_dict.safetensors").is_file():
            raise RuntimeError(
                f"Extracted Mistral LM checkpoint not found at {ckpt_dir}. "
                "Run scripts/extract_llava_med_mistral_lm.py first."
            )

        mistral_cfg = MistralConfig.from_pretrained(str(ckpt_dir))

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load the extracted LM weights into a fp16 MistralForCausalLM, then save
        # to a (cached) HF dir so BitsAndBytes can 4-bit quantize on load. We
        # materialize a one-time HF snapshot under the ckpt dir so subsequent
        # builds (and every fold of Phase 2b) reuse it without re-saving.
        hf_dir = ckpt_dir / "hf_fp16"
        if not (hf_dir / "config.json").is_file():
            log.info(f"Materializing fp16 HF snapshot of extracted Mistral LM at {hf_dir}")
            # Instantiate the empty skeleton directly in fp16 to halve transient
            # CPU memory (fp32 init would peak ~28 GB before the .to(fp16) cast).
            _prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            try:
                tmp = MistralForCausalLM(mistral_cfg)
            finally:
                torch.set_default_dtype(_prev)
            state = load_file(str(ckpt_dir / "lm_state_dict.safetensors"))
            load_res = tmp.load_state_dict(state, strict=False)
            if load_res.missing_keys or load_res.unexpected_keys:
                log.warning(
                    f"Mistral LM load_state_dict (snapshot): missing={load_res.missing_keys} "
                    f"unexpected={load_res.unexpected_keys}"
                )
            tmp = tmp.to(torch.float16)
            tmp.save_pretrained(str(hf_dir), safe_serialization=True)
            del tmp, state

        # --- stage the fp16 snapshot to node-local storage before the 4-bit load ---
        # from_pretrained(device_map="auto", 4-bit) does per-tensor mmap reads that are
        # pathologically slow on Lustre for this 14.5 GB file (observed ~0.6 MB/s/tensor
        # -> ETA hours; the smoke timed out mid-load). Copying the snapshot to
        # $SLURM_TMPDIR (node-local NVMe) first turns it into one sequential copy
        # (minutes) + a fast local quant-load. Falls back to Lustre if staging fails.
        import os as _os
        import shutil as _shutil
        import time as _time
        _localroot = _os.environ.get("SLURM_TMPDIR") or _os.environ.get("LOCAL_SCRATCH")
        if _localroot and Path(_localroot).is_dir():
            _staged = Path(_localroot) / "mistral_hf_fp16"
            try:
                if not (_staged / "config.json").is_file():
                    log.info(f"Staging Mistral fp16 snapshot {hf_dir} -> {_staged} (node-local; avoids slow Lustre mmap)")
                    _t0 = _time.time()
                    _shutil.copytree(str(hf_dir), str(_staged))
                    log.info(f"staged in {_time.time()-_t0:.0f}s")
                hf_dir = _staged
                log.info(f"Loading Mistral 4-bit from node-local {hf_dir}")
            except Exception as _e:
                log.warning(f"node-local staging failed ({_e}); loading from Lustre {hf_dir}")

        base = MistralForCausalLM.from_pretrained(
            str(hf_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
        log.info("Mistral (LLaVA-Med) LM loaded in 4-bit")

        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,   # q/k/v/o_proj exist in Mistral too
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.lm = get_peft_model(base, lora_cfg)
        # No M-RoPE patch → no dispatcher. The stash helpers are guarded by
        # hasattr(self.lm, "_mrope_dispatcher") so they no-op cleanly.

        # Tokenizer: Mistral's own (from the extracted ckpt; falls back to the
        # cached base). Left-padded for batch generation, same as MedGemma path.
        for tok_name in [str(ckpt_dir), "mistralai/Mistral-7B-Instruct-v0.2"]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)
                break
            except Exception:
                pass
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Mistral ready — trainable params: "
                 f"{sum(p.numel() for p in self.lm.parameters() if p.requires_grad):,}")

    def _format_prompt(
        self,
        schema_prompt: str,
        heuristic: str,
        use_heuristics: bool = True,
        json_prefill: str | None = None,
    ) -> str:
        """Build one LM prompt string, gated on lm_family.

        MedGemma (default): the historical plain-text marker format —
        `<SYS>..</SYS>\\n[CLASSIFIER HEURISTICS] ..\\n[TASK] ..\\n[JSON]` — kept
        BYTE-IDENTICAL so existing checkpoints/training distribution match.

        Mistral: the same content wrapped in the Mistral-Instruct chat template
        `[INST] .. [/INST]`. The visual prefix tokens are still prepended at the
        embedding level (outside this string); the JSON prefill, when present,
        is appended after `[/INST]` as the start of the assistant turn. This is
        a CONTENT-PRESERVING port — only the instruction delimiters change.
        """
        heur_line = (
            f"[CLASSIFIER HEURISTICS] {heuristic}\n" if use_heuristics else ""
        )
        if self.cfg.lm_family == "mistral":
            body = (
                f"{schema_prompt}\n"
                f"{heur_line}"
                "Generate a structured JSON radiology report."
            )
            text = f"[INST] {body} [/INST]"
            if json_prefill is not None:
                text = text + "\n" + json_prefill
            return text
        # MedGemma plain-text format (unchanged).
        text = (
            f"<SYS>{schema_prompt}</SYS>\n"
            f"{heur_line}"
            "[TASK] Generate a structured JSON radiology report.\n"
            "[JSON]"
        )
        if json_prefill is not None:
            text = text + "\n" + json_prefill
        return text

    def build_prompt(
        self,
        schema_prompt: str,
        heuristic_strings: list[str],
        target_jsons: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble the LM input for one batch.

        Prompt structure (per item):
            <SYS> {schema_prompt} </SYS>
            [CLASSIFIER HEURISTICS] {heuristic_string}
            [TASK] Generate a structured JSON radiology report.
            [JSON]

        Visual tokens are injected at the model's forward level (as prefix
        embeddings), not here — this method only handles text tokenization.

        Returns dict with 'input_ids', 'attention_mask' for training;
        for generation, omit target portion.
        """
        if self.tokenizer is None:
            raise RuntimeError("build_prompt requires a loaded tokenizer (real MedGemma only)")

        pieces = []
        for i, heuristic in enumerate(heuristic_strings):
            text = self._format_prompt(schema_prompt, heuristic, use_heuristics=True)
            if target_jsons is not None:
                text += " " + target_jsons[i]
            pieces.append(text)

        encoded = self.tokenizer(
            pieces,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_new_tokens + 512,
        )
        return encoded

    def _prepend_cohort_anchor(
        self,
        prefix_embeds: torch.Tensor,
        visual_mask: torch.Tensor,
        cohort_anchor_idx: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend the learned cohort-anchor embedding as the FIRST prefix token.

        Head-conditioned CoT v2. Given the mean-pool visual prefix `prefix_embeds`
        [B, P, D] and its `visual_mask` [B, P], look up a per-case anchor embedding
        (index in [0, n_cohort_anchor): 0=none, 1=GLI, 2=MEN, 3=MET — the Stage-1
        cohort probe's prediction) and concat it in front, returning
        ([B, 1+P, D], [B, 1+P]). The extra token is always attended (mask=1).

        No-op contract: returns the SAME tensor objects unchanged when the anchor is
        disabled (self.cohort_anchor is None) or no index is supplied
        (cohort_anchor_idx is None) — so the anchor-OFF path is BYTE-IDENTICAL.
        Stage-1 mean-pool prefix only (callers guard against use_stage2_mrope, whose
        spatial M-RoPE stash counts visual tokens and would need a matching offset).
        """
        if getattr(self, "cohort_anchor", None) is None or cohort_anchor_idx is None:
            return prefix_embeds, visual_mask
        B = prefix_embeds.shape[0]
        assert len(cohort_anchor_idx) == B, (
            f"cohort_anchor_idx length {len(cohort_anchor_idx)} != batch {B}"
        )
        device = prefix_embeds.device
        idx = torch.tensor([int(i) for i in cohort_anchor_idx], dtype=torch.long, device=device)
        anchor = self.cohort_anchor(idx).unsqueeze(1).to(prefix_embeds.dtype)  # [B, 1, D]
        prefix_embeds = torch.cat([anchor, prefix_embeds], dim=1)
        anchor_mask = torch.ones(B, 1, device=device, dtype=visual_mask.dtype)
        visual_mask = torch.cat([anchor_mask, visual_mask], dim=1)
        return prefix_embeds, visual_mask

    def _prefix_is_per_lesion(self) -> bool:
        """★ Stage D prereq ①: the ONE place that decides the prefix layout.

        ☠️ Every call site used to test `cfg.use_stage2_mrope` directly, which is what fused
        "concat per-lesion" with "apply spatial M-RoPE". There are four such sites, and a
        change that updates three of them is a silent train/eval architecture mismatch --
        the -56.4pp free-generation signature this project has already paid for once. So the
        decision lives here and the sites ask.

        `use_stage2_mrope=True` still implies per-lesion, so no existing config changes
        meaning; M-RoPE remains gated on that flag alone.
        """
        mode = getattr(self.cfg, "visual_prefix", "mean_pool")
        if mode not in ("mean_pool", "per_lesion"):
            raise ValueError(
                f"visual_prefix={mode!r} (expected 'mean_pool' or 'per_lesion'). Failing "
                f"closed: a typo must not silently select the legacy mean-pool layout, "
                f"because that layout makes a per-lesion connector a no-op and the run "
                f"would look healthy while measuring nothing.")
        return bool(self.cfg.use_stage2_mrope or mode == "per_lesion")

    def _build_per_lesion_prefix(
        self, visual_tokens: torch.Tensor, batch_idx: torch.Tensor,
        centroids: torch.Tensor, B: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Concat per-lesion Q-Former queries into a per-batch visual prefix.

        Stage-2 architectural change (2026-05-16): previously visual_tokens
        of shape [N_lesions, Q, D] were MEAN-pooled across lesions to give
        prefix [B, Q, D] — collapsing spatial identity and forcing the LM to
        guess hemisphere/location from a single pooled query stream. The audit
        flagged this as root cause of bilateral over-prediction AND of weak
        per-lesion field accuracy (location, axis_shift, etc.).

        New behavior: concat per-lesion queries into [B, max_lesions*Q, D]
        (zero-padded for batch items with fewer lesions). Each visual token
        thereby keeps a stable association to its source lesion's centroid,
        enabling M-RoPE Stage-2 (spatial position encoding) to distinguish
        them. Returns also the per-token centroid + valid-mask for downstream
        position-id construction.

        Returns:
            prefix_embeds:    [B, max_n_visual, D]   (zero-padded)
            tok_centroids:    [B, max_n_visual, 3]  (zero-filled padding)
            visual_mask:      [B, max_n_visual]      (1 valid, 0 pad)
            n_visual_per_b:   length-B list of unpadded visual lengths
        """
        Q = self.cfg.n_queries_per_lesion
        D = self.cfg.lm_hidden_dim
        n_lesions_per_b = [0] * B
        for b in batch_idx.tolist():
            n_lesions_per_b[b] += 1
        n_visual_per_b = [n * Q for n in n_lesions_per_b]
        max_n_visual = max(max(n_visual_per_b), Q)  # at least Q tokens (dummy-lesion case)

        prefix_embeds = torch.zeros(B, max_n_visual, D, device=device, dtype=torch.bfloat16)
        tok_centroids = torch.zeros(B, max_n_visual, 3, device=device, dtype=torch.long)
        visual_mask = torch.zeros(B, max_n_visual, device=device, dtype=torch.long)
        write_offset = [0] * B
        for i, b in enumerate(batch_idx.tolist()):
            off = write_offset[b]
            prefix_embeds[b, off:off + Q] = visual_tokens[i].to(torch.bfloat16)
            # All Q queries of one lesion share its centroid → same per-axis
            # position. This lets M-RoPE rotate same-lesion queries identically
            # (so they self-attend cleanly) while distinguishing different lesions.
            c = centroids[i].long().clamp(min=0)
            tok_centroids[b, off:off + Q] = c.unsqueeze(0).expand(Q, -1)
            visual_mask[b, off:off + Q] = 1
            write_offset[b] = off + Q
        return prefix_embeds, tok_centroids, visual_mask, n_visual_per_b

    def _stash_4axis_pids(
        self, seq_len: int, B: int, device: torch.device,
        n_visual_per_b: list[int] | None = None,
        tok_centroids: torch.Tensor | None = None,
        stage: str = "identity",
    ) -> None:
        """Build 4-axis position_ids and stash for the next forward call.

        Two stages:
          - stage="identity" (default): all tokens get text-position (p, p, p, p).
            M-RoPE reduces exactly to 1D RoPE — used to verify the patch path
            is exercised without changing model outputs.
          - stage="spatial": visual tokens get (modality=0, d, h, w) from their
            source lesion centroid; text tokens (before, between, after visual)
            get cumulative text positions (p, p, p, p). This makes M-RoPE
            actually encode per-lesion spatial identity and is the SSL-retrain
            architecture. NB: changes attention math → REQUIRES RETRAINING.

        Args:
            seq_len: total sequence length (B-shared)
            B: batch size
            device: torch device
            n_visual_per_b: per-batch number of valid visual tokens at the head
                of the sequence. Only used when stage="spatial".
            tok_centroids: [B, max_n_visual, 3] long, per visual token (d, h, w).
                Padding positions can hold anything (won't be attended via the
                attention mask). Only used when stage="spatial".
            stage: "identity" or "spatial".
        """
        if not (hasattr(self.lm, "_mrope_dispatcher") and self.lm._mrope_dispatcher is not None):
            return  # patch not installed (e.g. placeholder LM)
        # Build [4, B, T] pids tensor. For now we assume B==1 spatial layouts
        # are shared across the batch (true for both training and inference at
        # batch_size=1, which is our running config).
        if stage == "spatial" and n_visual_per_b is not None and tok_centroids is not None:
            assert B == 1, "spatial M-RoPE pids currently assume batch_size==1"
            n_v = n_visual_per_b[0]
            n_text = seq_len - n_v
            pids = torch.zeros(4, seq_len, device=device, dtype=torch.long)
            # Visual block: axis 0 = modality (default 0 since single visual stream),
            # axes 1..3 = (cz, cy, cx) per token. NB the routing's "cz"-named first
            # column is actually sagittal (X) — semantically harmless for M-RoPE
            # since the axes are exchangeable rotations.
            if n_v > 0:
                pids[0, :n_v] = 0
                pids[1, :n_v] = tok_centroids[0, :n_v, 0]
                pids[2, :n_v] = tok_centroids[0, :n_v, 1]
                pids[3, :n_v] = tok_centroids[0, :n_v, 2]
            # Text block: cumulative text positions starting at 0.
            if n_text > 0:
                t_pos = torch.arange(n_text, device=device, dtype=torch.long)
                pids[0, n_v:] = t_pos
                pids[1, n_v:] = t_pos
                pids[2, n_v:] = t_pos
                pids[3, n_v:] = t_pos
            pids_3d = pids.unsqueeze(1).expand(4, B, seq_len).contiguous()
        else:
            # Identity-reduction: all 4 axes = token index, matches 1D RoPE exactly.
            pos1d = torch.arange(seq_len, device=device, dtype=torch.long)
            pids_3d = pos1d.unsqueeze(0).expand(4, seq_len).unsqueeze(1).expand(4, B, seq_len).contiguous()
        self.lm._mrope_dispatcher._mrope_stash(pids_3d)

    def _clear_4axis_stash(self) -> None:
        if hasattr(self.lm, "_mrope_dispatcher") and self.lm._mrope_dispatcher is not None:
            self.lm._mrope_dispatcher._mrope_stash(None)

    def _stash_4axis_callable_for_generate(
        self, visual_pids: torch.Tensor, n_text_prefix: int, n_visual: int,
        device: torch.device,
    ) -> None:
        """Stash a callable that builds [4, T_q] pids on every dispatcher call.

        HF generate with KV cache calls apply_rotary_pos_emb at each decode
        step with q.shape[2]==1 (the new token only) and supplies the
        absolute position via HF's 1D position_ids. A static stash sized for
        the prefill would mismatch — the callable form computes the right
        4-axis pids dynamically.

        For a token at absolute position p (HF's position_ids value):
          - if p < n_visual: it's a visual prefix token → use
            visual_pids[:, p] (the per-token (modality, d, h, w))
          - else: it's a text token → row = (rel, rel, rel, rel) where
            rel = p - n_visual + n_text_prefix (cumulative text position).
            But for ABSOLUTE-position M-RoPE, simply (p, p, p, p) is the
            cleanest identity reduction and consistent with how forward()
            assigns text positions = [n_visual, n_visual+1, ...]. Use that.

        Returns nothing; the callable is stashed via the dispatcher.
        """
        if not (hasattr(self.lm, "_mrope_dispatcher") and self.lm._mrope_dispatcher is not None):
            return
        n_visual_local = int(n_visual)
        vis_pids = visual_pids  # [4, n_visual]

        def build_pids(hf_position_ids: torch.Tensor | None, T_q: int) -> torch.Tensor:
            # HF's position_ids is [B, T_q]. We need [4, T_q].
            if hf_position_ids is not None and hf_position_ids.dim() == 2:
                abs_pos = hf_position_ids[0]  # [T_q] for batch slot 0
            else:
                # No HF pids (rare; e.g., very old transformers). Assume
                # contiguous positions starting at 0.
                abs_pos = torch.arange(T_q, device=device, dtype=torch.long)
            pids = torch.zeros(4, T_q, dtype=torch.long, device=device)
            for ti in range(T_q):
                p = int(abs_pos[ti].item())
                if p < n_visual_local:
                    pids[:, ti] = vis_pids[:, p]
                else:
                    pids[:, ti] = p  # text token: (p, p, p, p)
            return pids

        self.lm._mrope_dispatcher._mrope_stash(build_pids)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        centroids: torch.Tensor,
        heuristic_strings: list[str],
        target_text_ids: torch.Tensor | None = None,
        target_text_mask: torch.Tensor | None = None,
        cohort_anchor_idx: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self._using_real_lm:
            lm_out = self.lm(visual_tokens, target_text_ids)
            return {"loss": lm_out.get("loss", torch.tensor(0.0, device=visual_tokens.device))}

        # Real LM forward: prepend visual tokens as soft prefix embeddings
        device = visual_tokens.device
        B = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1

        if self._prefix_is_per_lesion():
            # Concat per-lesion queries instead of mean-pooling. Each visual token keeps its
            # source-lesion centroid; whether those centroids become SPATIAL position ids is
            # a separate decision, made below and gated on use_stage2_mrope alone.
            prefix_embeds, tok_centroids, visual_mask, n_visual_per_b = \
                self._build_per_lesion_prefix(visual_tokens, batch_idx, centroids, B, device)
        else:
            # Stage-1 (legacy, backward-compatible with existing checkpoints):
            # mean-pool visual tokens across lesions → [B, Q, D].
            prefix_embeds = torch.zeros(B, self.cfg.n_queries_per_lesion,
                                        self.cfg.lm_hidden_dim, device=device,
                                        dtype=torch.bfloat16)
            counts = torch.zeros(B, device=device)
            for i, b in enumerate(batch_idx.tolist()):
                prefix_embeds[b] += visual_tokens[i].to(torch.bfloat16)
                counts[b] += 1
            prefix_embeds = prefix_embeds / counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
            tok_centroids = None
            visual_mask = torch.ones(B, prefix_embeds.shape[1], device=device, dtype=torch.long)
            n_visual_per_b = None

        # Head-conditioned CoT v2: prepend the cohort-anchor as the first prefix token
        # (default OFF => no-op). Grows prefix_embeds/visual_mask by one column; labels
        # (built below from prefix_embeds.shape[1]) and the identity M-RoPE stash
        # (seq_len from inputs_embeds) both follow automatically. Stage-1 mean-pool only.
        if getattr(self, "cohort_anchor", None) is not None and cohort_anchor_idx is not None:
            # ☠️ NARROWED 2026-08-06 (Stage D prereq ①). This used to refuse whenever the
            # prefix was per-lesion, because the two were the same flag. The actual conflict
            # is with the SPATIAL M-RoPE stash, which counts visual tokens and would be
            # thrown off by the extra anchor column. Under identity stashing there is
            # nothing to miscount, so a per-lesion prefix and the anchor now coexist -- which
            # is the point: the anchor is the only channel by which the dx head reaches the
            # draft, and it was collateral damage of the conflated flag.
            if self.cfg.use_stage2_mrope:
                raise NotImplementedError(
                    "cohort-anchor is incompatible with use_stage2_mrope specifically: the "
                    "SPATIAL M-RoPE stash counts visual tokens and the anchor adds one. It "
                    "IS compatible with visual_prefix='per_lesion' under identity stashing."
                )
            prefix_embeds, visual_mask = self._prepend_cohort_anchor(
                prefix_embeds, visual_mask, cohort_anchor_idx
            )

        if target_text_ids is None:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # Project visual prefix into LM embedding space.
        # get_input_embeddings() resolves to .language_model.embed_tokens for Gemma3 MM.
        word_embeds = self.lm.get_input_embeddings()(target_text_ids)
        inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        # Attention mask: visual padding (where visual_mask=0) must NOT be attended.
        text_mask = torch.ones(B, word_embeds.shape[1], device=device, dtype=torch.long)
        attention_mask = torch.cat([visual_mask, text_mask], dim=1)

        labels = torch.full(
            (B, prefix_embeds.shape[1]), -100, dtype=torch.long, device=device
        )
        labels = torch.cat([labels, target_text_ids], dim=1)

        # M-RoPE stash: Stage-2 uses spatial positions for visual tokens;
        # Stage-1 uses identity-reduction (text-position for all). Both exercise
        # the 4-axis dispatcher; Stage-2 changes attention math vs pretrained
        # 1D RoPE so requires the dedicated SSL retrain.
        # NB: Do NOT clear the stash in a finally — gradient checkpointing
        # re-runs attention during backward and needs the same pids (audit H3).
        self._stash_4axis_pids(
            seq_len=inputs_embeds.shape[1], B=B, device=device,
            n_visual_per_b=n_visual_per_b, tok_centroids=tok_centroids,
            stage="spatial" if self.cfg.use_stage2_mrope else "identity",
        )
        want_hidden = getattr(self, "dx_probe", None) is not None
        out = self.lm(inputs_embeds=inputs_embeds, labels=labels,
                      attention_mask=attention_mask,
                      output_hidden_states=want_hidden)
        res = {"loss": out.loss}
        if want_hidden:
            # last prefix-token position (index P-1): its next-token IS the first draft
            # token, which under --dx-first-draft is the diagnosis. Fixed position =>
            # tokenization-invariant, prefix-only => no text leak.
            P = prefix_embeds.shape[1]
            res["commit_hidden"] = out.hidden_states[-1][:, P - 1, :]   # [B, D_lm]
        return res

    def compute_debias_loss(
        self,
        visual_tokens: torch.Tensor,      # [N_lesions, Q, D] for the single case
        batch_idx: torch.Tensor,          # [N_lesions], all == 0 (B=1)
        centroids: torch.Tensor,          # [N_lesions, 3]
        heuristic_string: str,
        pinned_dx: list[str] | None,      # the dx-head's PREDICTED dx (NOT GT) for THIS case
        target_text_ids: torch.Tensor,    # [1, T] GT report tokens
    ) -> dict[str, torch.Tensor]:
        """Prompt-aware teacher-forced LM loss for the DE-BIAS LoRA (Step 5).

        Trains the LoRA to verbalize prose conditioned on the SAME prompt the M6 inference
        path builds — [visual prefix] + [schema + "CONFIRMED DIAGNOSIS: <dx>. findings/impression
        MUST be consistent with <dx>." + heuristic + json_prefill] — but with the dx-head's
        PREDICTED dx (so the LoRA learns to narrate the disease it will actually be told at
        inference, NOT the GT — else it fluently narrates the wrong disease on the ~23% wrong-pin
        cases). Loss = CE on the GT report tokens only (visual + prompt masked to -100).

        B is fixed to 1 (the trainer uses batch_size=1) so there is NO visual/prompt/target
        padding-alignment problem. Mirrors generate()'s prompt construction verbatim."""
        if not self._using_real_lm:
            return {"loss": torch.tensor(0.0, device=visual_tokens.device, requires_grad=True)}
        device = visual_tokens.device
        from schema import get_schema_prompt_block

        # 1. Visual prefix: Stage-1 mean-pool over the case's lesions -> [1, Q, D].
        prefix_embeds = visual_tokens.to(torch.bfloat16).mean(dim=0, keepdim=True)  # [1, Q, D]
        n_vis = prefix_embeds.shape[1]

        # 2. Hemisphere prefill (geometry, exactly as generate(): dim-0 sagittal vs midline,
        #    largest lesion first). centroids[:,0] is the misnamed-"cz" sagittal axis.
        midline = self.cfg.target_shape[0] / 2.0
        sag = centroids[:, 0].float().cpu().tolist() if len(centroids) else []
        hem = "right" if (sag and sag[0] < midline) else ("left" if sag else "right")
        cid = ""  # case_id is cosmetic in the prefill for training; the report tokens carry the GT
        json_prefill = f'{{"case_id": "{cid}", "hemisphere": "{hem}", '

        # 3. dx-pin injection (verbatim from generate()): condition on the PREDICTED dx.
        heur = heuristic_string or ""
        if pinned_dx:
            primary = pinned_dx[0] if isinstance(pinned_dx, (list, tuple)) and pinned_dx else str(pinned_dx)
            if isinstance(pinned_dx, (list, tuple)) and len(pinned_dx) > 1:
                ddx = ", ".join(map(str, pinned_dx))
                heur = (f"CONFIRMED DIAGNOSIS (most likely): {primary}; differential: {ddx}. "
                        f"The findings and impression MUST be consistent with {primary}. " + heur)
            else:
                heur = (f"CONFIRMED DIAGNOSIS: {primary}. The findings and impression MUST be "
                        f"consistent with {primary}. " + heur)
        prompt = self._format_prompt(get_schema_prompt_block(), heur,
                                     use_heuristics=True, json_prefill=json_prefill)

        # 4. Tokenize prompt + embed; embed the GT target. B=1 so no padding.
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        prompt_ids = enc["input_ids"]                                   # [1, P]
        emb = self.lm.get_input_embeddings()
        prompt_embeds = emb(prompt_ids).to(prefix_embeds.dtype)         # [1, P, D]
        target_embeds = emb(target_text_ids).to(prefix_embeds.dtype)    # [1, T, D]

        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
        attention_mask = torch.ones(1, inputs_embeds.shape[1], device=device, dtype=torch.long)
        # 5. Labels: predict ONLY the GT report tokens (visual + prompt -> -100).
        ignore = torch.full((1, n_vis + prompt_ids.shape[1]), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore, target_text_ids], dim=1)

        # M-RoPE position stash (MUST match the normal training forward, line ~1548): no-op for
        # Mistral (no _mrope_dispatcher), but for MedGemma's 4-axis M-RoPE this aligns the rotary
        # positions of THIS [visual+prompt+target] sequence. Without it the LoRA trains under
        # different positions than generate() deploys -> free-gen collapses while teacher-forced
        # loss stays fine (the exact -56.4pp signature the trap-guard caught).
        self._stash_4axis_pids(seq_len=inputs_embeds.shape[1], B=1, device=device,
                               n_visual_per_b=None, tok_centroids=None, stage="identity")
        # autocast(bf16): the trainable LoRA adapters are fp32 while the 4-bit base computes in
        # bf16; without this the first LoRA Linear hits "mat1 and mat2 dtype mismatch
        # (BFloat16 != float)". autocast casts both the bf16 activations and the fp32 LoRA weights
        # to bf16 for each matmul (standard QLoRA), and bf16 needs no GradScaler for backward.
        amp_dev = "cuda" if inputs_embeds.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, dtype=torch.bfloat16):
            out = self.lm(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)
        return {"loss": out.loss}

    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        centroids: torch.Tensor,
        heuristic_strings: list[str],
        k_samples: int = 8,
        use_json_constraint: bool = True,
        use_heuristics: bool = True,
        case_ids: list[str] | None = None,
        field_head_softmax_per_batch: list[dict[int, dict[str, dict[str, float]]]] | None = None,
        lambda_per_field: dict[str, float] | None = None,
        max_new_tokens: int | None = None,
        pinned_dx: list[list[str]] | None = None,
        neutral_case_id: bool = False,
        raw_prompt: bool = False,
    ) -> list[list[str]]:
        """Generate k JSON report strings per batch item, constrained by XGrammar.

        ── raw_prompt (FIX 1, 2026-08-08) ─────────────────────────────────────────
        When True, `heuristic_strings[i]` is sent to the LM VERBATIM (wrapped only in
        the family's chat delimiters) instead of being embedded in `_format_prompt`'s
        JSON-schema scaffold. This exists for the free-text verbalizer, which asks for
        prose and was being handed a prompt that OPENS with "You will generate a
        structured radiology report as JSON" and CLOSES, immediately before `[/INST]`,
        with "Generate a structured JSON radiology report." — so the LM reverted to
        JSON on ~92-100% of cases (BLOCKER-2) and every one of those reports was
        discarded by `_lm_verbalize`.

        ☠️ DEFAULT IS FALSE AND MUST STAY FALSE. Every existing caller — the 4-stage
        chain, `predict_m6`, `predict_draft_commit`, all banked numbers — keeps the
        byte-identical prompt it was measured with. This flag is opt-in per call, not
        a change of behaviour; that is the only reason it is landable before a freeze.
        `use_json_constraint` is independent: raw_prompt affects the PROMPT TEXT only,
        never the grammar, the prefill, or the logits processors.

        When `case_ids` is provided, the prompt is prefilled with
        `{"case_id": "<id>", ` per batch slot and the XGrammar matcher is
        advanced past that JSON prefix. This (a) aligns inference with the
        Phase 2b training distribution where case_id was always the first key,
        and (b) prevents case_id hallucination by supplying the value as
        context rather than asking the model to generate it.

        Late-fusion (ACL plan §Methodology). When `field_head_softmax_per_batch`
        is supplied, the constrained-decoding logits processor is wrapped with
        a `PerFieldLogitFuser` that biases the LM's choice of typed-field value
        tokens by `lambda_f * log p_head(c)` per enum class c. `lambda_per_field`
        controls the per-field fusion strength (default lambda=1.0 for all
        fields; 0.0 disables fusion for that field). This is the headline
        ACL paper mechanism and is strictly opt-in — None preserves the
        pre-fusion behaviour exactly.

        Ablation flags (paper §15):
          use_json_constraint=False -> A1 (no XGrammar / free generation)
          use_heuristics=False      -> A2 (drop the [CLASSIFIER HEURISTICS] line)

        Returns:
            list of length B; each element is a list of k JSON strings.
        """
        if not self._using_real_lm:
            return self._synthetic_reports(batch_idx, k_samples)

        import json
        from schema import get_decoding_json_schema

        device = visual_tokens.device
        B = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1

        if self._prefix_is_per_lesion():
            prefix_embeds, tok_centroids, visual_mask, n_visual_per_b = \
                self._build_per_lesion_prefix(visual_tokens, batch_idx, centroids, B, device)
        else:
            # Stage-1 (legacy): mean-pool prefix, no spatial stash. Matches
            # the architecture all existing checkpoints were trained under.
            prefix_embeds = torch.zeros(B, self.cfg.n_queries_per_lesion,
                                        self.cfg.lm_hidden_dim, device=device,
                                        dtype=torch.bfloat16)
            counts = torch.zeros(B, device=device)
            for i, b in enumerate(batch_idx.tolist()):
                prefix_embeds[b] += visual_tokens[i].to(torch.bfloat16)
                counts[b] += 1
            prefix_embeds = prefix_embeds / counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
            tok_centroids = None
            visual_mask = torch.ones(B, prefix_embeds.shape[1], device=device, dtype=torch.long)
            n_visual_per_b = None

        # XGrammar logits processor — schema-constrained JSON decoding (LoV3D 2026: 100% schema adherence by construction)
        logits_processors = []
        # Per-call reference so we can register/reset fuser state in the K-sample loop
        _field_logit_fuser = None  # type: ignore[var-annotated]
        if use_json_constraint and self.tokenizer is not None:
            try:
                from xgrammar_decoder import build_grammar_from_schema, XGrammarLogitsProcessor
                from schema import BrainTumorReport
                grammar = build_grammar_from_schema(BrainTumorReport)
                # Late-fusion: build the field-logit fuser if a softmax dict was
                # passed in. The fuser is reset per K-sample below.
                if field_head_softmax_per_batch is not None:
                    try:
                        from xgrammar_decoder import PerFieldLogitFuser
                        _field_logit_fuser = PerFieldLogitFuser(
                            tokenizer=self.tokenizer,
                            lambda_per_field=lambda_per_field or {},
                        )
                        log.debug(
                            "Late-fusion ACTIVE: PerFieldLogitFuser wired with "
                            f"lambda_per_field={lambda_per_field or {}}"
                        )
                    except Exception as e:
                        log.warning(f"PerFieldLogitFuser setup failed ({e}); proceeding without late-fusion")
                logits_processors.append(XGrammarLogitsProcessor(
                    grammar, self.tokenizer, field_logit_fuser=_field_logit_fuser,
                ))
                log.debug("XGrammar constrained decoding active")
                # RepetitionGuard AFTER XGrammar, deliberately: it must only ever REMOVE a
                # token from the grammar-legal set, never add one back. It targets the MEASURED
                # cause of schema-invalid reports -- 32% of generations loop on one token inside
                # a free-text string (where the grammar permits 31681 tokens), hit the 2048 cap,
                # and never emit findings/impression. Default ON via NF_REPETITION_GUARD; set to
                # 0 to disable. See repetition_guard.py for why the earlier EOS "fix" was a no-op.
                # ☠️ 2026-08-08: this import USED TO SIT BARE INSIDE THE OUTER `try`, and
                # `repetition_guard.py` was missing from the repo, so it raised ImportError
                # on EVERY generation. The outer handler then logged "xgrammar unavailable
                # ...falling back to unconstrained generation" — a message wrong about BOTH
                # its cause (the guard was missing, not xgrammar) AND its consequence
                # (XGrammarLogitsProcessor is appended ABOVE this line and the handler never
                # cleared the list, so constrained decoding stayed ACTIVE). An operator
                # reading that line would have drawn two false conclusions. Scope the import
                # to its own handler so a missing guard can never be reported as an xgrammar
                # failure, and say what is actually still running.
                if os.environ.get("NF_REPETITION_GUARD", "1") not in ("0", "false", ""):
                    try:
                        from repetition_guard import RepetitionGuard
                        logits_processors.append(RepetitionGuard())
                        log.debug("RepetitionGuard active (post-XGrammar)")
                    except ImportError as e:
                        log.warning(
                            f"RepetitionGuard UNAVAILABLE ({e}). XGrammar IS STILL ACTIVE "
                            f"({len(logits_processors)} logits processor(s) installed): JSON "
                            f"STRUCTURE remains constrained, but the free-text loop guard is "
                            f"NOT running, which is the measured cause of 2048-cap runaways.")

            except (ImportError, RuntimeError) as e:
                log.warning(f"xgrammar unavailable ({e}); falling back to unconstrained "
                            f"generation (logits_processors installed: {len(logits_processors)})")
            except Exception as e:
                log.warning(f"XGrammar setup failed ({e}); falling back to unconstrained "
                            f"generation (logits_processors installed: {len(logits_processors)})")

        # Generate k samples
        from schema import get_schema_prompt_block
        # JSON prefix per batch slot — fed verbatim to the LM as prompt context,
        # AND tokenized separately below for matcher advancement so XGrammar
        # knows we're past the first key-value pair.
        # Prefill is only meaningful under XGrammar: it aligns the matcher with
        # the Phase 2b training distribution (case_id always first key) AND
        # supplies the id as context. Under use_json_constraint=False (A1 ablation)
        # the prefill would force the model to begin every sample with
        # `{"case_id":"TRxx", ` and then free-generate — which guarantees
        # broken JSON and contaminates the no_xgrammar comparison. Gate on the
        # constraint flag so A1 measures true unconstrained generation.
        #
        # Bilateral-bias fix (2026-05-16): the visual encoder pools per-lesion
        # crops via Q-Former, losing laterality signal. As a result, the LM
        # over-predicts "bilateral" (~80% of test predictions vs ~2% in
        # training data; see project_phase3_findings_2026-05-16). Hemisphere
        # is deterministically derivable from lesion centroid x-coords in the
        # resized BraTS frame (W=128, midline=64; cx<midline → patient-right,
        # cx>midline → patient-left under BraTS RAS convention).
        #
        # Iteration: with PREDICTED seg (not GT) the contralateral side
        # often has tiny false-positive components, which under a naive
        # "any-voxel = bilateral" rule triggered bilateral inflation. The
        # LesionRouter sorts components largest-first, so we use ONLY the
        # largest lesion's centroid — robust to FP noise. Training-set
        # bilateral cases are 2.4%, test 0%, so the small accuracy loss on
        # truly-bilateral cases is acceptable vs the current 80% false
        # bilateral rate.
        hemispheres: list[str] = []
        if case_ids and use_json_constraint:
            # NB: LesionRouter unpacks `cz, cy, cx = center_of_mass(mask)`
            # which is misnamed — for BraTS NIfTI layout (X-sagittal, Y-coronal,
            # Z-axial), scipy.center_of_mass returns (d0, d1, d2) = (X, Y, Z).
            # The variable called "cz" in routing is actually X (sagittal), and
            # the one called "cx" is actually Z (axial). We need dim 0 (= the
            # misnamed "cz") for L-R classification.
            #
            # ☠️ B3 (2026-08-02) — THE "under BraTS RAS" CLAIM WAS FALSE, AND THE
            # COMMENT WAS THE HAZARD, NOT THE CODE. Under RAS, axis 0 increases
            # toward patient-RIGHT, which would make the rule below INVERTED. The
            # model is in fact served LPS UNCONDITIONALLY — dataset.py and
            # cloud-run/nf_infer.py both reorient via axcodes2ornt(("L","P","S")),
            # measured ('L','P','S') on 12/12 cases — so axis 0 increases toward
            # patient-LEFT and `d0 < midline -> patient-RIGHT` is CORRECT.
            # (The helper is now *named* `_canonical_lps` — renamed 2026-08-08
            # Stage-E orientation audit; it always executed LPS. Axis 1 is P,
            # not A. Derive anterior/posterior from the axcodes, not a name.)
            # A uniform sign error here is invisible because a mirrored volume
            # looks internally consistent, and this project has already shipped a
            # 96.5% L/R mirror once. Hence the runtime assert below rather than a
            # comment: n=5 was never enough evidence for a binary global property.
            # Earlier broken version used centroids[:, 2] (axial) which is
            # essentially random w.r.t. hemisphere (51% acc).
            # Midline of the sagittal axis in resized-volume voxel coords.
            # Derived from cfg.target_shape[0] (the sagittal dim — same axis
            # the routing's `cz`-named variable actually stores, per the
            # misnamed-unpack note above).
            midline = self.cfg.target_shape[0] / 2.0
            sag_all = centroids[:, 0].float().cpu().tolist()
            bi_list = batch_idx.cpu().tolist()
            for b in range(B):
                case_sag = [x for x, bi in zip(sag_all, bi_list) if bi == b]
                if not case_sag:
                    hemispheres.append("right")
                    continue
                largest_sag = case_sag[0]  # routing emits largest-first
                hemispheres.append("right" if largest_sag < midline else "left")
        # Ablation (default-OFF): neutral_case_id blanks the cohort tag
        # ("RG_MEN_.."/"RG_MET_..") out of the JSON prefill — verbatim to the
        # generate_draft_commit pattern — so free-generated differential_diagnosis
        # cannot be read off the id. Default False keeps `cids == case_ids`, so the
        # prefill (and the locked MLCN hero numbers) are BYTE-IDENTICAL.
        cids = (["case" for _ in range(B)] if neutral_case_id else (case_ids or [""] * B))
        # NF_STRIP_CASE_ID: omit the cohort-tagged id from the prefill entirely. `hemisphere`
        # stays because it is the model's OWN field-head prediction from the image (legitimate
        # self-conditioning), whereas case_id is external metadata that leaks the answer.
        _strip_cid = os.environ.get("NF_STRIP_CASE_ID") == "1"
        json_prefills = [
            (f'{{"hemisphere": "{hem}", ' if _strip_cid
             else f'{{"case_id": "{cid}", "hemisphere": "{hem}", ')
            for cid, hem in zip(cids, hemispheres)
        ] if (case_ids and use_json_constraint) else None
        prompts = []
        for i in range(B):
            heur = heuristic_strings[i] if i < len(heuristic_strings) else ""
            # M6 diagnosis-pin: ground the prose on the DiagnosisHead's dx (read off
            # connector_IN, Gate C 0.84). Prepended to the heuristic line so the LM's
            # findings/impression are consistent with the pinned diagnosis instead of
            # free-generating from a glioma-biased prior (the M5 0/60-meningioma failure).
            if pinned_dx is not None and i < len(pinned_dx) and pinned_dx[i]:
                dx_list = pinned_dx[i]
                primary = dx_list[0] if isinstance(dx_list, (list, tuple)) and dx_list else str(dx_list)
                if len(dx_list) > 1:
                    ddx = ", ".join(map(str, dx_list))
                    heur = (f"CONFIRMED DIAGNOSIS (most likely): {primary}; differential: {ddx}. "
                            f"The findings and impression MUST be consistent with {primary}. " + heur)
                else:
                    heur = (f"CONFIRMED DIAGNOSIS: {primary}. The findings and impression MUST be "
                            f"consistent with {primary}. " + heur)
            if raw_prompt:
                # FIX 1: send the caller's text verbatim. No schema block, no
                # "Generate a structured JSON radiology report." trailer — those are
                # exactly what made the free-text verbalizer emit JSON. Chat
                # delimiters only, matching _format_prompt's family handling so the
                # Mistral instruction format the LM was trained on still holds.
                base = (f"[INST] {heur} [/INST]" if self.cfg.lm_family == "mistral"
                        else heur)
            else:
                base = self._format_prompt(
                    get_schema_prompt_block(), heur,
                    use_heuristics=use_heuristics,
                    json_prefill=json_prefills[i] if json_prefills is not None else None,
                )
            prompts.append(base)

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        # Tokenize the JSON prefill alone (add_special_tokens=False so we don't
        # pick up BOS tokens) for matcher advancement. Per-batch token lists.
        prefill_token_ids: list[list[int]] | None = None
        if json_prefills is not None:
            prefill_enc = self.tokenizer(json_prefills, add_special_tokens=False)
            prefill_token_ids = [list(ids) for ids in prefill_enc["input_ids"]]
        # Use the standard HF input-embedding API — robust across model families.
        # For Gemma3 multimodal, this resolves to model.language_model.embed_tokens.
        word_embeds = self.lm.get_input_embeddings()(encoded["input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        # Attention mask: visual padding (visual_mask=0 where the per-lesion
        # concat had fewer than max_n_visual tokens) must NOT be attended.
        attention_mask = torch.cat([visual_mask, encoded["attention_mask"]], dim=1)

        # M-RoPE stash for generate. Two modes:
        #   Stage-2: stash a CALLABLE that builds [4, T_q] pids on-demand
        #     using HF's per-step 1D position_ids. Correct for every step
        #     (prefill AND each KV-cache decode token) — eliminates the
        #     decode-step train-infer mismatch.
        #   Stage-1 (legacy): identity-reduction static tensor — text-position
        #     for all, numerically same as 1D RoPE. Decode-step mismatches
        #     fall back to 1D via the dispatcher safety net.
        if self.cfg.use_stage2_mrope and tok_centroids is not None and n_visual_per_b is not None:
            n_v = n_visual_per_b[0]
            visual_pids = torch.zeros(4, n_v, device=device, dtype=torch.long)
            if n_v > 0:
                visual_pids[0] = 0
                visual_pids[1] = tok_centroids[0, :n_v, 0]
                visual_pids[2] = tok_centroids[0, :n_v, 1]
                visual_pids[3] = tok_centroids[0, :n_v, 2]
            self._stash_4axis_callable_for_generate(
                visual_pids=visual_pids,
                n_text_prefix=encoded["input_ids"].shape[1],
                n_visual=n_v,
                device=device,
            )
        else:
            self._stash_4axis_pids(
                seq_len=inputs_embeds.shape[1], B=B, device=device,
                n_visual_per_b=None, tok_centroids=None,
                stage="identity",
            )

        all_outputs: list[list[str]] = [[] for _ in range(B)]
        for k_iter in range(k_samples):
            # XGrammar matchers are stateful per-sequence and terminate when the JSON
            # is complete. Without reset, sample k+1 inherits the terminated state of
            # sample k and forces EOS immediately → empty output. Reset every loop.
            # Same applies to the late-fusion fuser — its field-state tracker must
            # restart for each new sample, and per-batch head softmax must be
            # re-registered so register_case takes effect this K-iteration.
            if _field_logit_fuser is not None and field_head_softmax_per_batch is not None:
                _field_logit_fuser.reset_case()
                for b in range(B):
                    if b < len(field_head_softmax_per_batch):
                        # Use the largest-lesion (lesion_index=1) head dist as a
                        # fallback when the matcher hasn't surfaced which lesion
                        # is currently being emitted. For multi-lesion cases the
                        # in-decoder lesion-tracking is a follow-up enhancement;
                        # the single-lesion-majority corpus makes this safe.
                        per_lesion = field_head_softmax_per_batch[b]
                        head_dist = per_lesion.get(1) or next(iter(per_lesion.values()), {})
                        if head_dist:
                            _field_logit_fuser.register_case(b, head_dist)
            for lp in logits_processors:
                if hasattr(lp, "reset"):
                    lp.reset(B)
                # DEBUG: log prefill_token_ids to verify they're what we expect
                if k_iter == 0 and prefill_token_ids is not None:
                    log.warning(
                        f"GENERATE-DEBUG k_iter=0: prefill_token_ids[0] "
                        f"(n={len(prefill_token_ids[0])}): {prefill_token_ids[0]}"
                    )
                    log.warning(
                        f"GENERATE-DEBUG json_prefills[0]: {json_prefills[0]!r}"
                    )
                # Advance each matcher through the prefilled JSON tokens so the
                # first masked step is "what comes after `{"case_id":"<id>", `"
                # — i.e. the second key (`"hemisphere"`), matching the Phase 2b
                # training distribution.
                if prefill_token_ids is not None and hasattr(lp, "_matchers"):
                    for b, ids in enumerate(prefill_token_ids):
                        m = lp._matchers[b]
                        for tok in ids:
                            if m.is_terminated():
                                break
                            # accept_token returns False (does NOT raise) on rejection.
                            # If the matcher state desyncs from generation, masking
                            # blows up and the model EOS-bails right after the prefill
                            # — surface this as a loud warning + break so the failure
                            # is visible in logs instead of silently producing empties.
                            try:
                                ok = bool(m.accept_token(int(tok)))
                            except Exception as e:
                                log.warning(
                                    f"matcher.accept_token raised on token {tok} "
                                    f"at batch {b}: {e}"
                                )
                                break
                            if not ok:
                                log.warning(
                                    f"matcher rejected prefill token {tok} at batch {b} "
                                    f"(grammar/prefill desync); aborting prefill advance"
                                )
                                break
            # QLoRA leaves lm_head in fp32 while the trunk runs in bf16; autocast handles
            # the mixed-precision matmul without us having to upcast lm_head manually.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out_ids = self.lm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=(max_new_tokens if max_new_tokens is not None
                                    else self.cfg.max_new_tokens),
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    logits_processor=logits_processors if logits_processors else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # HF's generate with inputs_embeds returns ONLY the new (generated)
            # tokens on modern transformers (4.36+). We use transformers 5.8,
            # so out_ids IS already the new tokens — do NOT subtract input_len.
            # See memory: feedback_lm_generate_slice — earlier mistaken slice
            # incorrectly dropped the first ~input_len of valid JSON tokens.
            new_ids = out_ids
            # Stage-0 instrumentation (NF v1.6): attribute a truncated JSON to its actual
            # cause. 6/156 hero drafts stopped mid-structure at the IDENTICAL point; the
            # length data already ruled out the token ceiling (they stop at ~1/4 the length
            # valid drafts reach), so the open question is whether the grammar matcher
            # reported is_terminated() early (forcing EOS at xgrammar_decoder.py:325-331).
            _cap = (max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens)
            for _b in range(new_ids.shape[0]):
                _n = int(new_ids[_b].numel())
                _last = int(new_ids[_b, -1].item()) if _n else -1
                _term = None
                for _lp in (logits_processors or []):
                    _ms = getattr(_lp, "_matchers", None)
                    if _ms and _b < len(_ms):
                        try:
                            _term = _ms[_b].is_terminated()
                        except Exception:  # noqa: BLE001
                            _term = "err"
                log.info(
                    f"[gen-stop] b={_b} n_new={_n}/{_cap} hit_cap={_n >= _cap} "
                    f"last_tok={_last} is_eos={_last == self.tokenizer.eos_token_id} "
                    f"matcher_terminated={_term}"
                )
            decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for b, text in enumerate(decoded):
                t = text.strip()
                # Re-attach the prefilled JSON prefix so the saved sample is
                # complete JSON (downstream parsers don't know about prefill).
                if json_prefills is not None:
                    t = json_prefills[b] + t
                all_outputs[b].append(t)

        return all_outputs

    @torch.no_grad()
    def generate_draft_commit(
        self,
        visual_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        centroids: torch.Tensor,
        case_ids: list[str] | None = None,
        max_draft_tokens: int = 160,
        max_json_tokens: int | None = None,
        use_json_constraint: bool = True,
        commit_only: bool = False,
        neutral_case_id: bool = False,
        cohort_anchor_idx: list[int] | None = None,
        dx_softmax_per_batch: list[dict] | None = None,
        dx_fusion_lambda: float = 2.0,
        commit_temperature: float | None = None,
        pinned_dx: list[list[str]] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Draft-then-commit decode matching the --cot-supervision training target.

        A CoT-supervised adapter learned to map the visual prefix ->
        '<draft prose>[COMMIT]<json>'. We reproduce that decode in two passes,
        feeding ONLY the visual prefix (+BOS) as context — NO schema prompt — so
        the context matches the TRAINING forward (LMWithMRope4D.forward:
        inputs_embeds = [visual_prefix, target_tokens], with no prompt in between).

          Stage 1 (draft):  free-generate up to `max_draft_tokens` from
                            [prefix][BOS]; cut at the [COMMIT] marker if emitted.
          Stage 2 (commit): append COT_COMMIT_MARKER (+ the same case_id/hemisphere
                            JSON prefill generate() uses) and XGrammar-constrained-
                            generate the JSON. Reuses generate()'s XGrammar
                            processor + matcher-advance machinery verbatim.

        Returns (json_strings, draft_strings): one committed JSON + its draft per
        batch item (single sample each). Stage-1 mean-pool prefix only (Model-1
        Mistral, use_stage2_mrope=False).

        dx_softmax_per_batch (L2, eval-time only): when provided, bias the Stage-2
        free-text differential_diagnosis[0] slot toward the leak-free probe's
        cohort, classifier-free-guidance style (lambda>1 amplifies), while leaving
        the prose free. It is a list[dict] of length B, one {canonical_dx_string:
        prob} softmax per case (e.g. {"Meningioma": 0.90, "Glioblastoma": 0.05,
        "Metastasis": 0.05}); the canonical strings MUST match
        xgrammar_decoder._DEFAULT_DX_COHORT_STRINGS / probe_anchor.CANON. Under the
        hood a PerFieldLogitFuser(enable_dx_fusion=True) is attached to the Stage-2
        XGrammar processor and only differential_diagnosis is fused. Default None =>
        NO fuser attached => byte-identical to the pre-fusion decode. generate() is
        untouched by this path.

        RED-TEAM: an over-strong dx_fusion_lambda can push a first token the
        matcher then cannot complete into valid JSON. The caller MUST validate the
        returned JSON (json.loads + schema) and fall back to the post-hoc override
        scripts/probe_anchor.py::write_override on any invalid/empty commit.
        """
        if not self._using_real_lm:
            jsons = [lst[0] for lst in self._synthetic_reports(batch_idx, 1)]
            return jsons, ["" for _ in jsons]
        # ☠️☠️ THE SHIPPED-PATH TRAP, WIDENED 2026-08-06 (Stage D prereq ①).
        # This is the inference path NF_knight 1.1 actually ships. It hardcodes the
        # mean-pool prefix. Before the visual_prefix decoupling it could only be reached in
        # the mean-pool configuration, so the narrow use_stage2_mrope guard was sufficient;
        # now `visual_prefix="per_lesion"` reaches it too, and WITHOUT this widening it
        # would silently build a MEAN-POOL prefix from per-lesion-trained weights.
        # That is a train/eval ARCHITECTURE mismatch that produces fluent output and a
        # quietly wrong number -- the -56.4pp free-generation signature this project has
        # already paid for once. Fail loudly instead: an unimplemented path must raise, not
        # improvise. Stage D implements the per-lesion decode here, in lockstep with the
        # training prefix, or the arms are not comparable.
        if self._prefix_is_per_lesion():
            raise NotImplementedError(
                "generate_draft_commit supports the mean-pool prefix ONLY, and this model "
                f"is configured per-lesion (visual_prefix="
                f"{getattr(self.cfg, 'visual_prefix', 'mean_pool')!r}, "
                f"use_stage2_mrope={self.cfg.use_stage2_mrope}). Silently mean-pooling here "
                "would decode a DIFFERENT ARCHITECTURE from the one that was trained. "
                "Implement the per-lesion prefix in this function in the same change that "
                "enables it for training."
            )
        device = visual_tokens.device
        B = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1
        emb = self.lm.get_input_embeddings()

        # Mean-pool visual prefix per case -> [B, Q, D] (verbatim from generate()).
        prefix_embeds = torch.zeros(B, self.cfg.n_queries_per_lesion,
                                    self.cfg.lm_hidden_dim, device=device, dtype=torch.bfloat16)
        counts = torch.zeros(B, device=device)
        for i, b in enumerate(batch_idx.tolist()):
            prefix_embeds[b] += visual_tokens[i].to(torch.bfloat16)
            counts[b] += 1
        prefix_embeds = prefix_embeds / counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
        visual_mask = torch.ones(B, prefix_embeds.shape[1], device=device, dtype=torch.long)

        # Head-conditioned CoT v2: prepend the cohort-anchor to the SHARED visual
        # prefix so BOTH the draft (Stage 1) and the commit (Stage 2) passes below
        # reason from the cohort prior (both concat this prefix_embeds). Default OFF
        # => no-op => byte-identical decode. use_stage2_mrope already raised above.
        if getattr(self, "cohort_anchor", None) is not None and cohort_anchor_idx is not None:
            prefix_embeds, visual_mask = self._prepend_cohort_anchor(
                prefix_embeds, visual_mask, cohort_anchor_idx
            )

        # B′9 (2026-08-08): generation-time dx pin — INJECT-AND-VERBALIZE on the
        # draft-commit path (GATE G-B′9; a Dx-FIELD-only override is a RECORDED
        # DEFECT). The pin sentence is the SAME string generate()/compute_debias_loss
        # inject; it is tokenized, embedded and APPENDED to the shared visual prefix,
        # so BOTH the Stage-1 draft and the Stage-2 commit condition on it (mirrors
        # _prepend_cohort_anchor's shared-prefix mechanism directly above).
        # ⚠️ Off-distribution by design: the CoT adapter trained with NO text between
        # prefix and target — hence the pre-registered A/B, never a silent default.
        # None, or an empty per-case list => byte-identical decode for that case.
        if pinned_dx is not None and any(pinned_dx):
            if B != 1:
                raise NotImplementedError(
                    "pinned_dx on the draft-commit path requires batch_size 1 "
                    "(per-case pin lengths differ; padding is not implemented)")
            dxl = pinned_dx[0]
            if dxl:
                primary = dxl[0] if isinstance(dxl, (list, tuple)) else str(dxl)
                if isinstance(dxl, (list, tuple)) and len(dxl) > 1:
                    ddx = ", ".join(map(str, dxl))
                    pin_txt = (f"CONFIRMED DIAGNOSIS (most likely): {primary}; differential: "
                               f"{ddx}. The findings and impression MUST be consistent with "
                               f"{primary}. ")
                else:
                    pin_txt = (f"CONFIRMED DIAGNOSIS: {primary}. The findings and impression "
                               f"MUST be consistent with {primary}. ")
                pin_ids = self.tokenizer(pin_txt, return_tensors="pt",
                                         add_special_tokens=False)["input_ids"].to(device)
                pin_embeds = emb(pin_ids).to(prefix_embeds.dtype)
                prefix_embeds = torch.cat([prefix_embeds, pin_embeds], dim=1)
                visual_mask = torch.cat(
                    [visual_mask,
                     torch.ones(1, pin_ids.shape[1], device=device, dtype=torch.long)], dim=1)

        # ---- Stage 1: draft (free-gen, no XGrammar) ---------------------------
        # Feed [visual_prefix][BOS]; training targets were tokenized with
        # add_special_tokens=True so the model saw BOS as the first target token.
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.eos_token_id
        bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=device)
        s1_embeds = torch.cat([prefix_embeds, emb(bos).to(prefix_embeds.dtype)], dim=1)
        s1_mask = torch.cat([visual_mask, torch.ones(B, 1, device=device, dtype=torch.long)], dim=1)
        self._stash_4axis_pids(seq_len=s1_embeds.shape[1], B=B, device=device,
                               n_visual_per_b=None, tok_centroids=None, stage="identity")
        if commit_only:
            # Ablation: no reasoning chain -- commit straight from the visual prefix
            # (empty draft), to test whether the draft causally sets the diagnosis.
            drafts = ["" for _ in range(B)]
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                draft_ids = self.lm.generate(
                    inputs_embeds=s1_embeds, attention_mask=s1_mask,
                    max_new_tokens=max_draft_tokens, do_sample=True,
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            drafts_raw = self.tokenizer.batch_decode(draft_ids, skip_special_tokens=True)
            # Cut at the marker if the model already emitted it inside the draft window.
            drafts = [d.split("[COMMIT]")[0].rstrip() for d in drafts_raw]

        # ---- hemisphere geometry prefill (verbatim logic from generate()) -----
        midline = self.cfg.target_shape[0] / 2.0
        sag_all = centroids[:, 0].float().cpu().tolist() if len(centroids) else []
        bi_list = batch_idx.cpu().tolist()
        hemispheres: list[str] = []
        for b in range(B):
            case_sag = [x for x, bi in zip(sag_all, bi_list) if bi == b]
            hemispheres.append("right" if (not case_sag or case_sag[0] < midline) else "left")
        # Ablation: neutral_case_id blanks the cohort tag ("RG_MEN_..") out of the
        # commit prefill, proving the diagnosis is read from the image, not the id.
        cids = (["case" for _ in range(B)] if neutral_case_id else (case_ids or [""] * B))
        # NF_STRIP_CASE_ID: omit the cohort-tagged id from the commit prefill entirely (matches
        # the generate() path at ~line 1911 and the strip-trained model's training prefill).
        _strip_cid = os.environ.get("NF_STRIP_CASE_ID") == "1"
        json_prefills: list[str | None] = (
            [(f'{{"hemisphere": "{hem}", ' if _strip_cid
              else f'{{"case_id": "{cid}", "hemisphere": "{hem}", ')
             for cid, hem in zip(cids, hemispheres)]
            if use_json_constraint else [None] * B
        )

        # ---- Stage 2: commit (XGrammar-constrained JSON) ----------------------
        commit_texts = [
            f"{drafts[b]}{COT_COMMIT_MARKER}{json_prefills[b] or ''}" for b in range(B)
        ]
        commit_enc = self.tokenizer(commit_texts, return_tensors="pt", padding=True,
                                    add_special_tokens=True).to(device)
        commit_embeds = emb(commit_enc["input_ids"]).to(prefix_embeds.dtype)
        s2_embeds = torch.cat([prefix_embeds, commit_embeds], dim=1)
        s2_mask = torch.cat([visual_mask, commit_enc["attention_mask"]], dim=1)

        # L2 dx-slot fusion (eval-time only): when a per-case dx softmax is
        # supplied, attach a PerFieldLogitFuser configured for ONLY
        # differential_diagnosis so the Stage-2 commit biases its top-1 dx toward
        # the leak-free probe cohort (classifier-free-guidance; lambda>1 amplifies).
        # dx_softmax_per_batch is None => _dx_fuser stays None =>
        # XGrammarLogitsProcessor(field_logit_fuser=None) is identical to the
        # pre-fusion call => byte-identical decode. (Mirrors generate() at the
        # field_head_softmax_per_batch wiring; generate() itself is untouched.)
        _dx_fuser = None  # type: ignore[var-annotated]
        logits_processors = []
        if use_json_constraint and self.tokenizer is not None:
            try:
                from xgrammar_decoder import build_grammar_from_schema, XGrammarLogitsProcessor
                from schema import BrainTumorReport
                grammar = build_grammar_from_schema(BrainTumorReport)
                if dx_softmax_per_batch is not None:
                    try:
                        from xgrammar_decoder import PerFieldLogitFuser
                        _dx_fuser = PerFieldLogitFuser(
                            tokenizer=self.tokenizer,
                            lambda_per_field={"differential_diagnosis": float(dx_fusion_lambda)},
                            field_enum_strings={},        # ONLY differential_diagnosis
                            enable_dx_fusion=True,
                        )
                        log.debug(
                            "L2 dx-fusion ACTIVE: PerFieldLogitFuser wired for "
                            f"differential_diagnosis, lambda={float(dx_fusion_lambda)}"
                        )
                    except Exception as e:
                        _dx_fuser = None
                        log.warning(f"dx PerFieldLogitFuser setup failed ({e}); commit without dx-fusion")
                logits_processors.append(XGrammarLogitsProcessor(
                    grammar, self.tokenizer, field_logit_fuser=_dx_fuser,
                ))
                # RepetitionGuard AFTER XGrammar, deliberately: it must only ever REMOVE a
                # token from the grammar-legal set, never add one back. It targets the MEASURED
                # cause of schema-invalid reports -- 32% of generations loop on one token inside
                # a free-text string (where the grammar permits 31681 tokens), hit the 2048 cap,
                # and never emit findings/impression. Default ON via NF_REPETITION_GUARD; set to
                # 0 to disable. See repetition_guard.py for why the earlier EOS "fix" was a no-op.
                # ☠️ Same defect as the generate() site — see the note there. Scoped so a
                # missing guard cannot masquerade as an XGrammar failure.
                if os.environ.get("NF_REPETITION_GUARD", "1") not in ("0", "false", ""):
                    try:
                        from repetition_guard import RepetitionGuard
                        logits_processors.append(RepetitionGuard())
                        log.debug("RepetitionGuard active (post-XGrammar)")
                    except ImportError as e:
                        log.warning(
                            f"RepetitionGuard UNAVAILABLE ({e}). XGrammar IS STILL ACTIVE "
                            f"({len(logits_processors)} processor(s)): commit-stage STRUCTURE "
                            f"is constrained; the free-text loop guard is NOT running.")

            except Exception as e:  # pragma: no cover - falls back to unconstrained
                log.warning(f"XGrammar setup failed ({e}); commit stage unconstrained "
                            f"(logits_processors installed: {len(logits_processors)})")

        prefill_token_ids: list[list[int]] | None = None
        if use_json_constraint and json_prefills[0] is not None:
            prefill_enc = self.tokenizer(list(json_prefills), add_special_tokens=False)
            prefill_token_ids = [list(ids) for ids in prefill_enc["input_ids"]]

        self._stash_4axis_pids(seq_len=s2_embeds.shape[1], B=B, device=device,
                               n_visual_per_b=None, tok_centroids=None, stage="identity")
        for lp in logits_processors:
            if hasattr(lp, "reset"):
                lp.reset(B)
            # Advance each matcher through the prefilled JSON tokens so the first
            # masked step is "what comes after the prefill" (2nd key onward).
            if prefill_token_ids is not None and hasattr(lp, "_matchers"):
                for b, ids in enumerate(prefill_token_ids):
                    m = lp._matchers[b]
                    for tok in ids:
                        if m.is_terminated():
                            break
                        try:
                            ok = bool(m.accept_token(int(tok)))
                        except Exception as e:
                            log.warning(f"matcher.accept_token raised on {tok} at b{b}: {e}")
                            break
                        if not ok:
                            log.warning(f"matcher rejected prefill token {tok} at b{b}; aborting advance")
                            break
        # Register each case's dx softmax + reset the fuser's per-case state
        # (XGrammar stateful-reset requirement: reset before each generation).
        # The fuser observes only GENERATED tokens — the case_id/hemisphere prefill
        # is advanced on the matcher, NOT fed to the fuser — so its recent-token
        # window starts clean and catches the generated `"differential_diagnosis":`
        # key. register_case expects {field: {class_label: prob}}, so wrap the
        # per-case {canonical_dx_string: prob} under the differential_diagnosis key.
        if _dx_fuser is not None and dx_softmax_per_batch is not None:
            _dx_fuser.reset_case()  # clear ALL per-batch state, then re-register
            for b in range(B):
                sm = dx_softmax_per_batch[b] if b < len(dx_softmax_per_batch) else None
                if sm:
                    _dx_fuser.register_case(b, {"differential_diagnosis": dict(sm)})
        # NF v1.5 Phase A: decouple the COMMIT (Stage-2) temperature from the DRAFT
        # (Stage-1) temperature. Best-of-k wants a HOT draft (diverse reasoning ->
        # diverse committed diagnosis) but a COOL/greedy commit (clean prose, so the
        # kept candidate's report body retains v1.3-full content quality). Faithfulness
        # flip-rate = 1.0 guarantees the draft's diversity still propagates to the
        # commit even when the commit itself is greedy. commit_temperature is None =>
        # byte-identical to the pre-Phase-A decode (do_sample, cfg.temperature/top_p).
        if commit_temperature is None:
            _s2_gen = dict(do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p)
        elif commit_temperature > 0:
            _s2_gen = dict(do_sample=True, temperature=float(commit_temperature), top_p=self.cfg.top_p)
        else:
            _s2_gen = dict(do_sample=False)   # greedy (XGrammar still masks -> valid JSON)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            json_ids = self.lm.generate(
                inputs_embeds=s2_embeds, attention_mask=s2_mask,
                max_new_tokens=(max_json_tokens if max_json_tokens is not None
                                else self.cfg.max_new_tokens),
                **_s2_gen,
                logits_processor=logits_processors if logits_processors else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        json_dec = self.tokenizer.batch_decode(json_ids, skip_special_tokens=True)
        out_jsons: list[str] = []
        for b in range(B):
            t = json_dec[b].strip()
            if json_prefills[b] is not None:
                t = json_prefills[b] + t  # re-attach prefix so it parses as full JSON
            out_jsons.append(t)
        return out_jsons, drafts

    def _synthetic_reports(self, batch_idx: torch.Tensor, k: int = 1) -> list[list[str]]:
        """Return valid synthetic JSON reports for smoke-testing without a real LM."""
        import json
        B = int(batch_idx.max().item()) + 1 if len(batch_idx) else 1
        template = {
            "case_id": "TR01",
            "hemisphere": "right",
            "differential_diagnosis": ["Glioblastoma multiforme (GBM)"],
            "overall_mass_effect": "marked",
            "lesions": [{
                "lesion_index": 1,
                "location": "parieto-temporal",
                "composition": "necrotic",
                "enhancement_pattern": "thick-walled-ring",
                "surrounding_effects": {
                    "edema_severity": "marked",
                    "mass_effect": "present",
                    "midline_shift": "present",
                },
                "involvement": {
                    "ventricular_compression": "present",
                    "brainstem_compression": "present",
                    "midbrain_compression": "none",
                },
                "axis_shift": "contralateral",
            }],
            "findings": (
                "Large irregular peripherally enhancing mass in the right parieto-temporal lobe. "
                "Thick-walled ring enhancement with central necrotic component and marked "
                "peritumoral edema. Mass effect with contralateral midline shift and "
                "compression on the right lateral ventricle and brainstem."
            ),
            "impression": "Findings most consistent with glioblastoma multiforme.",
        }
        json_str = json.dumps(template)
        return [[json_str] * k for _ in range(B)]


class _PlaceholderLM(nn.Module):
    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.lm_hidden_dim, 32000)

    def forward(self, visual_tokens: torch.Tensor, target_text_ids: torch.Tensor | None = None) -> dict:
        logits = self.proj(visual_tokens.mean(dim=1))
        return {"loss": logits.mean() * 0.0, "logits": logits}


# ==========================================================================
# 6. NeuroFusion top-level model
# ==========================================================================


class NeuroFusion(nn.Module):
    """End-to-end NeuroFusion v4: vision + routing + Q-Former + classifier + LM + losses."""

    def __init__(
        self,
        cfg: NeuroFusionConfig,
        mednext_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = SegmentationBackbone(cfg, checkpoint_path=mednext_checkpoint)
        self.router = LesionRouter(cfg)
        self.qformer = PerLesionQFormer(cfg)
        self.field_head = FieldClassificationHead(cfg)
        self.lm = LMWithMRope4D(cfg)

        # W-QF-OBJ aux dx head. Registered as an attribute of the MODEL, not of the
        # Q-Former, on purpose: `qformer_params` in train.py is
        # `list(model.qformer.parameters()) + list(model.field_head.parameters())`, so
        # hanging it off self.qformer would silently enrol it in that group under
        # lr_qformer, while hanging it here means train.py must add it explicitly and the
        # gradient path into the Q-Former is the only thing this loss can act through.
        self.aux_dx_head = None
        if getattr(cfg, "use_aux_dx", False):
            self.aux_dx_head = nn.Linear(cfg.lm_hidden_dim, 4)

        # Rung-3 dx-distill: load the FROZEN DiagnosisHead as the distillation teacher
        # (default OFF => None => nothing added; the head/buffers are NOT under
        # lm/qformer/field_head so they never enter the optimizer groups). Buffers move
        # with .to(device) and (harmlessly) ride the state_dict; the head is reloaded
        # fresh each run regardless.
        self.dx_head_module = None
        if getattr(cfg, "use_dx_distill", False):
            from pathlib import Path
            from scripts.predict_m6 import _load_diagnosis_head
            hp = cfg.dx_head_path or "out/m6_research/diagnosis_head_mistral.pt"
            _h = _load_diagnosis_head(Path(hp), torch.device("cpu"))
            self.dx_head_module = _h["module"]
            for p in self.dx_head_module.parameters():
                p.requires_grad = False
            self.dx_head_module.eval()
            self.register_buffer("dx_feat_mu", _h["feat_mu"].detach().cpu(), persistent=False)
            self.register_buffer("dx_feat_sd", _h["feat_sd"].detach().cpu(), persistent=False)
            self.dx_nl_mean = float(_h["nl_mean"])
            self.dx_nl_std = float(_h["nl_std"])
            self.dx_T = float(_h["temperature"])

    def set_backbone_trainable(self, trainable: bool) -> list[nn.Parameter]:
        """Flip requires_grad on the pretrained MedNeXt trunk and return the params
        that now require grad so train.py can add them to the optimizer as a new
        param group (at cfg.backbone_lr).

        This is the first-class hook for the gradual unfreeze (project_neural_
        ceiling_diagnosis). It changes WHAT trains, never WHICH data — leak safety
        is unaffected (the Phase-2b loader still only sees splits.json train folds).

        Behaviour:
          - Real MONAI wrap (vision.backbone has attr 'mednext'): set
            requires_grad=trainable on backbone.mednext.parameters() ONLY. The
            bottleneck->feature projection (backbone.feature_proj) is left as-is
            and stays trainable — it is NOT part of `mednext`, so it is never
            touched here and was already trainable under the frozen convention.
          - Placeholder / non-MONAI backbone (no 'mednext'): operate on the whole
            vision.backbone.parameters() (there is no separate feature_proj to
            preserve in that path).
          - The LM (QLoRA adapters + frozen MedGemma base) is NEVER touched — only
            the 3D vision trunk is (un)frozen.

        When trainable=True the trunk forward stays in the autograd graph (the
        forward() path runs vision() WITHOUT any torch.no_grad wrapper, so grads
        flow once requires_grad is set). LM gradient checkpointing is independent
        of this and is unaffected.

        Returns:
            list[nn.Parameter]: the trunk parameters whose requires_grad was just
            set to `trainable`. When trainable=False this is the now-frozen set
            (caller can ignore it); when trainable=True these are exactly the
            params to register in the optimizer's backbone param group. Returned
            in a stable order (named_parameters order) and de-duplicated.
        """
        backbone = getattr(self.vision, "backbone", None)
        if backbone is None:
            return []

        if hasattr(backbone, "mednext"):
            # Real MONAI MedNeXt wrap: flip only the pretrained trunk; keep
            # feature_proj trainable (it lives on the wrap, outside .mednext).
            trunk = backbone.mednext
            if hasattr(backbone, "feature_proj"):
                for p in backbone.feature_proj.parameters():
                    p.requires_grad = True
        else:
            # Placeholder / non-MONAI: no separate feature_proj to preserve.
            trunk = backbone

        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in trunk.parameters():
            p.requires_grad = trainable
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)

        # On unfreeze, install per-stage gradient checkpointing on the MONAI trunk so
        # the now-trainable 128^3 backbone's activations fit a 40 GB A100 (monkeypatch
        # only — module structure / state_dict keys unchanged; see
        # _mednext_forward_checkpointed). One-shot + MONAI-path only.
        if trainable and hasattr(backbone, "mednext"):
            import types
            mednext = backbone.mednext
            if not getattr(mednext, "_nf_stage_ckpt_installed", False):
                mednext.forward = types.MethodType(_mednext_forward_checkpointed, mednext)
                mednext._nf_stage_ckpt_installed = True
                log.info(
                    "backbone unfreeze: installed per-stage gradient checkpointing on "
                    "MedNeXt trunk (enc/dec stages + up/down blocks)"
                )
        return params

    def forward(
        self,
        mri: torch.Tensor,
        seg: torch.Tensor,
        field_targets: dict[str, torch.Tensor] | None = None,
        target_text_ids: torch.Tensor | None = None,
        target_text_mask: torch.Tensor | None = None,
        training: bool = True,
        cohort_anchor_idx: list[int] | None = None,
        routing_seg_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # A/B. Vision
        vis = self.vision(mri)
        seg_logits, features = vis["seg_logits"], vis["features"]

        # C. Lesion routing (GT at train, predicted at eval)
        #
        # `routing_seg` (the MASK that decides which lesions get routed) and `features` (what the
        # Q-Former / DiagnosisHead actually see) are separate arguments to the router. That
        # decoupling is what makes a DETECTION-ONLY auxiliary segmenter possible: supply a better
        # mask from a second, MET-fine-tuned segmenter while `features` still come from the
        # UNCHANGED hero trunk -> no feature drift, so no DiagnosisHead refit and no risk to the
        # content win.
        #
        # Until now nothing exercised that: `--mednext-checkpoint` swaps `self.vision`, which
        # produces seg_logits AND features from the SAME backbone, so the existing MET-FT result
        # moved the mask and the features together and cannot attribute the gain to detection.
        # `routing_seg_override` is the eval-only hook that separates them. Training is untouched
        # (it still routes on GT `seg`), so this cannot affect any training run.
        if routing_seg_override is not None:
            if training:
                raise ValueError("routing_seg_override is eval-only; training routes on GT seg")
            routing_seg = routing_seg_override
        else:
            # ☠️ THE INTEGER CLIFF. This argmax is the point where a floating-point
            # hair's-breadth becomes a DISCRETE routing decision: LesionRouter sorts
            # components by exact voxel count and keeps the top `max_lesions`. Measured
            # 8.3% (5/60) routed-set change A100-vs-CPU under default TF32, 0/60 with
            # TF32 off. The precision guard that closes that is `seg_tf32_off()`, applied
            # inside `SegmentationBackbone.forward` above — i.e. it is already in force on
            # `seg_logits` by the time control reaches this line, including on the
            # `routing_seg_override` branch, whose mask comes from the same guarded
            # forward.
            routing_seg = seg if training else seg_logits.argmax(dim=1)
        routing = self.router(features, routing_seg, training=training)

        # D. Q-Former with 3D PE
        visual_tokens = self.qformer(
            lesion_features=routing["lesion_features"],
            lesion_idx=routing["lesion_idx"],
            centroids=routing["centroids"],
        )

        # D2. W-QF-OBJ: 4-way dx logits on the per-case MEAN of connector_OUT.
        #
        # Pooling matches what the LM prefix effectively presents per case (mean over that
        # case's lesions, mean over the 32 queries), so the loss acts on the same quantity
        # the connector_OUT probe measures -- not on some other projection that could improve
        # while the LM-visible tokens do not.
        #
        # ☠️ DUMMY CASES ARE EXCLUDED, NOT SILENTLY LABELLED. When a case has no routed
        # lesion with positive voxels the LesionRouter substitutes a ZERO-FILLED dummy
        # (model.py:599). Its pooled token block is a CONSTANT, so including it would teach
        # the head "constant input -> this case's cohort" -- the exact free-accuracy shortcut
        # that made all 8 empty-mask MET cases score 8/8 correct in the detection study. The
        # valid-slot list is returned so train.py can align labels to it; a batch where every
        # slot is a dummy contributes NO aux loss rather than a confident wrong one.
        aux_dx_logits = None
        aux_dx_slots: list[int] = []
        if self.aux_dx_head is not None:
            bidx = routing["batch_idx"]
            pooled_rows = []
            for b in range(mri.shape[0]):
                rows = (bidx == b).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                if not torch.any(routing["lesion_features"][rows]):
                    continue                      # empty-seg dummy -> no supervision
                pooled_rows.append(visual_tokens[rows].mean(dim=(0, 1)))   # [D_lm]
                aux_dx_slots.append(b)
            if pooled_rows:
                aux_dx_logits = self.aux_dx_head(
                    torch.stack(pooled_rows).to(self.aux_dx_head.weight.dtype))

        # E. Field classifier + heuristic rendering
        # global-context token = mean-pool of the frozen MedNeXt bottleneck (matches the
        # 0.495 "bottleneck_global" probe reference); only computed when enabled.
        global_feat = features.mean(dim=(-3, -2, -1)) if self.field_head.use_global else None
        field_logits = self.field_head(visual_tokens, global_feat=global_feat,
                                       batch_idx=routing["batch_idx"])
        heuristic_strings = self.field_head.render_heuristic_text(
            field_logits, routing["batch_idx"], routing["lesion_idx"]
        )

        # F. LM with 4-axis M-RoPE
        lm_out = self.lm(
            visual_tokens=visual_tokens,
            batch_idx=routing["batch_idx"],
            centroids=routing["centroids"],
            heuristic_strings=heuristic_strings,
            target_text_ids=target_text_ids,
            target_text_mask=target_text_mask,
            cohort_anchor_idx=cohort_anchor_idx,
        )

        out = {
            "seg_logits": seg_logits,
            "visual_tokens": visual_tokens,
            "field_logits": field_logits,
            "heuristic_strings": heuristic_strings,
            "lm_loss": lm_out["loss"],
            "routing": routing,
            "commit_hidden": lm_out.get("commit_hidden"),   # Rung-3 dx-distill (None if OFF)
            "aux_dx_logits": aux_dx_logits,                 # W-QF-OBJ (None if OFF/all-dummy)
            "aux_dx_slots": aux_dx_slots,                   # batch slots those rows correspond to
        }

        if training:
            total, components = self._compute_total_loss(out, seg, field_targets)
            out["loss"] = total
            out["loss_components"] = components   # individual scalars for logging

        return out

    def _compute_total_loss(
        self,
        model_out: dict[str, Any],
        seg: torch.Tensor,
        field_targets: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (total_loss, {name: scalar_tensor}) for W&B logging."""
        cfg = self.cfg
        losses: dict[str, torch.Tensor] = {}

        # ℒ_seg: auxiliary dice + CE against BraTS GT mask (no grad into frozen backbone)
        losses["seg"] = dice_plus_ce_loss(model_out["seg_logits"], seg, n_classes=cfg.seg_n_classes)

        if field_targets is not None:
            per_field = []
            for name, logits in model_out["field_logits"].items():
                if name in field_targets:
                    per_field.append(F.cross_entropy(logits, field_targets[name]))
            losses["field"] = torch.stack(per_field).mean() if per_field else torch.tensor(0.0, device=seg.device)
        else:
            losses["field"] = torch.tensor(0.0, device=seg.device)

        # ℒ_lm: language modelling loss from LM forward
        losses["gen"] = model_out["lm_loss"]

        # ℒ_dx_distill (Rung-3): the real training `total` is rebuilt in train.py from the
        # loss components (this internal total is discarded during training), so the distill
        # term is computed there via self.dx_distill_loss(out) and added to train.py's total.
        # Kept 0 here so the model's own total is byte-identical when the flag is OFF.
        losses["dx_distill"] = torch.tensor(0.0, device=seg.device)

        # ℒ_ground: Q-Former attention IoU vs. GT masks (stub — needs attn weight extraction)
        losses["ground"] = torch.tensor(0.0, device=seg.device)

        # ℒ_kl: KL(LoRA-adapted || frozen-base) on replay buffer (stub — anti-forgetting)
        losses["kl_base"] = torch.tensor(0.0, device=seg.device)

        total = (
            cfg.w_seg * losses["seg"]
            + cfg.w_field * losses["field"]
            + cfg.w_gen * losses["gen"]
            + cfg.w_ground * losses["ground"]
            + cfg.kl_to_base_weight * losses["kl_base"]
            + getattr(cfg, "dx_distill_weight", 0.0) * losses["dx_distill"]
        )
        return total, losses

    def dx_distill_loss(self, model_out: dict[str, Any]) -> torch.Tensor:
        """Rung-3 head-distillation loss: KL(dx_probe(commit_hidden) ‖ frozen-head softmax).

        The frozen DiagnosisHead reads the dominant lesion's pooled router features
        (reproduces predict_m6._diagnose EXACTLY: mean-pool crop ++ normalized n_lesions,
        standardize, MLP, /T, softmax); the teacher is detached. The student is the trained
        dx_probe over the LM's last-prefix hidden. Returns 0 when the flag is OFF / no probe /
        no commit_hidden, so the caller's total is unchanged. Called from train.py's loop
        (where the real `total` is assembled)."""
        ch = model_out.get("commit_hidden")
        if (not getattr(self.cfg, "use_dx_distill", False) or ch is None
                or getattr(self, "dx_head_module", None) is None
                or getattr(self.lm, "dx_probe", None) is None):
            return torch.tensor(0.0, device=ch.device if ch is not None else "cpu")
        routing = model_out["routing"]
        bidx = routing["batch_idx"]; lidx = routing["lesion_idx"]
        teach, keep = [], []
        for b in range(ch.shape[0]):
            bmask = (bidx == b)
            if bmask.sum() == 0:
                continue
            rows = bmask.nonzero(as_tuple=True)[0]
            r = int(rows[lidx[rows].argmin()])           # dominant = smallest lesion_idx
            feat = routing["lesion_features"][r].mean(dim=(-3, -2, -1)).float()
            nl = float(int(bmask.sum().item()))
            nl_norm = (nl - self.dx_nl_mean) / (self.dx_nl_std + 1e-6)
            x = torch.cat([feat, feat.new_tensor([nl_norm])], dim=0)          # [321]
            x = (x - self.dx_feat_mu.to(x.device)) / self.dx_feat_sd.to(x.device)
            with torch.no_grad():
                logit = self.dx_head_module(x.unsqueeze(0)).squeeze(0) / self.dx_T
                teach.append(torch.softmax(logit, dim=0))                     # [4]
            keep.append(b)
        if not keep:
            return torch.tensor(0.0, device=ch.device)
        teacher = torch.stack(teach).detach()                                # [V, 4]
        student_logp = F.log_softmax(self.lm.dx_probe(ch[keep]), dim=-1)      # [V, 4]
        return F.kl_div(student_logp, teacher, reduction="batchmean")


# ==========================================================================
# 7. HiResCAM hook for post-hoc grounding (inference-only)
# ==========================================================================


class HiResCAM:
    """HiResCAM (Draelos & Carin 2021) for 3D saliency over MedNeXt + Q-Former.

    Two hook targets (selected at construction time):
      - "qformer": last Q-Former decoder layer — 32-token query grid [N, 32, D_lm]
      - "backbone": last Conv3d of MedNeXt encoder — 3D feature map [B, C, d, h, w]

    HiResCAM guarantee: element-wise product (activations × gradients) without
    spatial averaging. This provably highlights only regions the model actually
    attended to (unlike Grad-CAM's gap-averaged weights).

    Usage (inference only)::

        cam = HiResCAM("enhancement_pattern", model, target="backbone")
        saliency_3d = cam(mri, seg, target_class_idx=3)  # [D, H, W] ∈ [0, 1]
    """

    def __init__(
        self,
        field_name: str,
        model: NeuroFusion,
        target: str = "qformer",
    ) -> None:
        self.field_name = field_name
        self.model = model
        self.target = target
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        if target == "qformer":
            hook_module = model.qformer.decoder.layers[-1]
        elif target == "backbone":
            # Last Conv3d in MedNeXt encoder (or placeholder feat_head)
            hook_module = None
            for m in model.vision.backbone.modules():
                if isinstance(m, nn.Conv3d):
                    hook_module = m
            if hook_module is None:
                raise ValueError("No Conv3d found in backbone for HiResCAM hooking")
        else:
            raise ValueError(f"Unknown HiResCAM target: {target!r}. Use 'qformer' or 'backbone'.")

        hook_module.register_forward_hook(self._save_activations)
        hook_module.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out[0] if isinstance(out, tuple) else out

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(
        self,
        mri: torch.Tensor,
        seg: torch.Tensor,
        target_class_idx: int,
        output_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Compute HiResCAM saliency for `target_class_idx` of `self.field_name`.

        Args:
            mri: [B, 4, D, H, W]
            seg: [B, D, H, W]
            target_class_idx: class index within field_name's classifier head
            output_shape: if given, trilinearly upsample saliency to (D, H, W)

        Returns:
            saliency: [N_total, d, h, w] (backbone) or [N_total, 32] (qformer),
                      normalised to [0, 1]. If output_shape given, returns
                      [B, D, H, W] upsampled to original MRI resolution.
        """
        self.model.zero_grad()
        out = self.model(mri, seg, training=False)
        logits = out["field_logits"][self.field_name]   # [N_total, n_classes]
        score = logits[:, target_class_idx].sum()
        score.backward()

        assert self.activations is not None and self.gradients is not None, \
            "HiResCAM hooks did not fire — ensure model ran a full forward+backward pass."

        # HiResCAM: element-wise product (no gap), then sum over channel axis
        act = self.activations
        grad = self.gradients

        if self.target == "backbone":
            # act: [B, C, d, h, w] → cam: [B, d, h, w]
            cam = (act * grad).sum(dim=1)
        else:
            # act: [N, 32, D_lm] → cam: [N, 32]
            cam = (act * grad).sum(dim=-1)

        cam = F.relu(cam)

        # Normalise each item in the batch to [0, 1]
        flat = cam.view(cam.shape[0], -1)
        cam_min = flat.min(dim=-1).values
        cam_max = flat.max(dim=-1).values
        denom = (cam_max - cam_min).clamp(min=1e-8)
        for i in range(cam.shape[0]):
            cam[i] = (cam[i] - cam_min[i]) / denom[i]

        if output_shape is not None and self.target == "backbone":
            # Upsample 3D saliency to full MRI resolution
            cam = F.interpolate(
                cam.unsqueeze(1),               # [B, 1, d, h, w]
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(1)                        # [B, D, H, W]

        return cam.detach()


# ==========================================================================
# 8. Qwen2.5-VL-7B parallel-pilot stub
# ==========================================================================


class NeuroFusionQwenVariant(nn.Module):
    """Parallel pilot: Qwen2.5-VL-7B with native 3-axis M-RoPE.

    Depth-as-temporal adaptation: slice index -> t axis, in-plane patches -> (h, w).
    4 MRI sequences handled via channel-expanded patch embed (Conv3d in_ch=4 at input).
    Zero code changes inside Qwen2.5-VL itself; the M-RoPE machinery is already there.

    This variant runs as an ablation foil to quantify the gain of 4-axis over 3-axis M-RoPE.
    If MedGemma retrofit proves unstable in Phase 2a, this is the insurance policy.
    """

    def __init__(self, cfg: NeuroFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO:
        # from transformers import Qwen2_5_VLForConditionalGeneration
        # self.lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)
        # - expand patch_embed.proj: Conv3d(in_channels=4, ...)
        # - at forward time, pass position_ids with t=slice_idx, h/w=in-plane
        # - rest of pipeline identical (seg backbone, router, Q-Former, classifier)
        raise NotImplementedError("wire up in Week 4 parallel pilot")


# ==========================================================================
# Loss utilities
# ==========================================================================


def dice_plus_ce_loss(logits: torch.Tensor, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * target_onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    dice_loss = 1 - dice[1:].mean()
    return 0.5 * ce + 0.5 * dice_loss


# ==========================================================================
# Smoke test
# ==========================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = NeuroFusionConfig()
    model = NeuroFusion(cfg)
    print(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    mri = torch.randn(2, 4, 64, 64, 64)
    seg = torch.randint(0, 4, (2, 64, 64, 64))
    out = model(mri, seg, training=True)
    print(f"seg_logits shape:    {out['seg_logits'].shape}")
    print(f"visual_tokens shape: {out['visual_tokens'].shape}")
    print(f"heuristic strings:   {out['heuristic_strings']}")
    print(f"total loss:          {out['loss'].item():.4f}")
    print()
    print("[ok] v4 skeleton flows end-to-end. Swap placeholders in Week 2.")
    print("     - Phase 1: train MedNeXt on BraTS 2020+2021+2023 (~45 GPU-h)")
    print("     - Phase 2a: M-RoPE identity-init sanity check on text-only batch")
    print("     - Phase 2b: 5-fold CV multi-task training on NeuroFusion-330")
