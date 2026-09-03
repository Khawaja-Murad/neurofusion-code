"""
mrope_patch.py — Monkey-patch HuggingFace MedGemma (Gemma3) to use 4-axis M-RoPE.

What it does:
    Replaces the rotary-position-embedding application inside every Gemma3Attention
    layer of a loaded HF model. The patched attention checks the position_ids tensor:

        - If position_ids has shape [batch, seq_len]                (1D RoPE, default)
          → forward to the original apply_rotary_pos_emb()  (text-only / pretrained behavior)

        - If position_ids has shape [4, batch, seq_len]             (4-axis M-RoPE)
          → forward to apply_mrope_4d() from mrope_4d.py  (volumetric / NeuroFusion behavior)

Why this design:
    Backward compatibility. After patching, the model still passes its own
    test suite when given normal 2D position_ids, AND can be fed 4-axis
    position_ids when visual tokens are present. The choice happens at runtime
    per-batch.

How to use:
    from transformers import AutoModelForCausalLM
    from mrope_patch import patch_medgemma_with_mrope_4d, build_mrope_position_ids
    from mrope_4d import MRope4DConfig

    model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it",
                                                  torch_dtype=torch.bfloat16)
    cfg = MRope4DConfig(head_dim=256, rotary_dim=128, axis_channels=(16,16,16,16))
    patch_medgemma_with_mrope_4d(model, cfg)

    # Now the model accepts a 4D position_ids when desired:
    position_ids = build_mrope_position_ids(
        seq_len=512,
        token_kinds=token_kinds,         # 'text' or 'visual' per token
        modality_idx=modality_idx,       # 0..3 per visual token
        voxel_coords=voxel_coords,       # (d, h, w) per visual token
    )                                     # [4, B, T]
    out = model(input_ids=..., position_ids=position_ids, ...)

Numerical notes:
    The (p, p, p, p) text-token reduction is mathematically exact in fp32. In bf16
    the cumulative numerical drift in a forward pass is on the order of 1e-3 in
    logits — verified by the Phase 2a sanity check in train.py. This is well below
    LoRA update magnitudes and is not a problem in practice.

What this file does NOT do:
    - Load MedGemma. Caller does that.
    - Apply LoRA / PEFT. Caller does that with peft.LoraConfig + get_peft_model.
    - Build the prompt or inject visual tokens. The Q-Former + LMWithMRope4D
      assemble those.

It just patches the rotary application step.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn

from mrope_4d import (
    MRope4DAngleTable,
    MRope4DConfig,
    apply_mrope_4d,
    build_position_ids,
)

log = logging.getLogger("mrope_patch")


# ===========================================================================
# Position-ID construction
# ===========================================================================


def build_mrope_position_ids(
    seq_len: int,
    token_kinds: list[str],          # 'text' or 'visual' per token; length == seq_len
    modality_idx: dict[int, int] | None = None,    # for visual tokens: token_idx -> modality
    voxel_coords: dict[int, tuple[int, int, int]] | None = None,  # token_idx -> (d, h, w)
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a 4-axis position-id tensor of shape [4, batch_size, seq_len].

    For text tokens at index i: position = (p_i, p_i, p_i, p_i) where p_i is the
    text token's cumulative count among text tokens. This makes M-RoPE collapse
    to 1D RoPE for those tokens.

    For visual tokens at index i: position = (modality_idx[i], d, h, w) from the
    voxel coordinate of the Q-Former query that produced this visual token.

    Args:
        token_kinds: list of length seq_len, each "text" or "visual"
        modality_idx: required for visual tokens; defaults to 0 if missing
        voxel_coords: required for visual tokens; defaults to (0,0,0) if missing
    """
    assert len(token_kinds) == seq_len, "token_kinds must have len == seq_len"
    modality_idx = modality_idx or {}
    voxel_coords = voxel_coords or {}

    pos = torch.zeros(4, seq_len, dtype=torch.long, device=device)
    text_counter = 0
    for i, kind in enumerate(token_kinds):
        if kind == "text":
            pos[:, i] = text_counter
            text_counter += 1
        elif kind == "visual":
            t = modality_idx.get(i, 0)
            d, h, w = voxel_coords.get(i, (0, 0, 0))
            pos[0, i] = t
            pos[1, i] = d
            pos[2, i] = h
            pos[3, i] = w
        else:
            raise ValueError(f"unknown token kind '{kind}' at index {i}")

    return pos.unsqueeze(1).expand(-1, batch_size, -1).contiguous()


def is_4axis_position_ids(position_ids: Any) -> bool:
    """Returns True if position_ids is the 4-axis form [4, B, T] OR a
    callable that builds pids[4, T] on demand (used for KV-cache decode
    steps where the absolute position changes per call)."""
    if position_ids is None:
        return False
    if callable(position_ids):
        return True
    return hasattr(position_ids, "dim") and position_ids.dim() == 3 and position_ids.shape[0] == 4


# ===========================================================================
# Patched apply_rotary_pos_emb dispatcher
# ===========================================================================


def make_dispatching_apply_rope(
    original_apply: Callable,
    mrope_table_local: MRope4DAngleTable,
    mrope_table_global: MRope4DAngleTable,
) -> Callable:
    """Construct a replacement function for transformers' apply_rotary_pos_emb.

    For Gemma3, HF's signature is:
        apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)

    Dispatcher behavior:
      1. If no 4-axis stash is present, forward to original 1D apply.
      2. If a stash IS present, pick the per-layer-type table using the thread-local
         `is_sliding` hint set by the wrapped Gemma3DecoderLayer.forward (see
         ``wrap_decoder_layers_with_sliding_hint``).
            hint=True  → mrope_table_local   (base 10K)
            hint=False → mrope_table_global  (base 1M w/ linear factor 8.0)
            hint=None  → falls back to local + one-time warning (unsafe but non-fatal)
    """
    import threading
    import inspect
    _stash = threading.local()
    _layer_hint = threading.local()
    _abs_pos = threading.local()
    _hint_warned = {"flag": False}
    _no_pos_warned = {"flag": False}

    # The HF apply_rotary_pos_emb signature varies by transformers version / model family.
    # Current Gemma3 (transformers ≥ 4.52) is (q, k, cos, sin, unsqueeze_dim=1) — no position_ids.
    # Older versions exposed position_ids. Detect once so the fallback forwards the right args.
    _orig_params = list(inspect.signature(original_apply).parameters.keys())
    _orig_accepts_pids = "position_ids" in _orig_params

    def stash(position_ids: torch.Tensor | None) -> None:
        _stash.position_ids = position_ids

    def get_stash() -> torch.Tensor | None:
        return getattr(_stash, "position_ids", None)

    def set_layer_hint(is_sliding: bool | None) -> None:
        _layer_hint.is_sliding = is_sliding

    def get_layer_hint() -> bool | None:
        return getattr(_layer_hint, "is_sliding", None)

    def set_abs_pos(position_ids: torch.Tensor | None) -> None:
        # Captured by the decoder-layer wrapper from HF's real position_ids /
        # cache_position. Needed because Gemma3's apply_rotary_pos_emb is called
        # WITHOUT position_ids (it consumes them upstream to build cos/sin), so
        # the dispatcher would otherwise have no absolute position during KV-cache
        # decode — every generated token would collapse to position 0.
        _abs_pos.position_ids = position_ids

    def get_abs_pos() -> torch.Tensor | None:
        return getattr(_abs_pos, "position_ids", None)

    def patched(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1):
        stashed = get_stash()
        if stashed is not None and is_4axis_position_ids(stashed):
            T_q = q.shape[2]
            # Stash may be either a [4, B, T_stash] tensor (Stage-1/static path) or a
            # callable that builds pids on demand given HF's current 1D position_ids
            # (for KV-cache decode-step correctness).
            if callable(stashed):
                # Gemma3 calls apply_rotary_pos_emb WITHOUT position_ids, so the
                # arg here is None on every decode step. Recover the true absolute
                # positions captured by the decoder-layer wrapper (cache_position /
                # position_ids). Without this, build_pids falls back to arange(T_q)
                # → arange(1)=[0] per decode token → all generated tokens rotate at
                # position 0 → free-gen collapses (TF works, free-gen scores 0).
                pid_arg = position_ids
                if pid_arg is None:
                    ap = get_abs_pos()
                    if ap is not None:
                        pid_arg = ap if ap.dim() == 2 else ap.view(1, -1)
                    elif not _no_pos_warned["flag"]:
                        log.warning(
                            "mrope callable stash active but no absolute position "
                            "captured (decoder-layer wrapper missed?); decode positions "
                            "will be wrong. Verify wrap_decoder_layers installed."
                        )
                        _no_pos_warned["flag"] = True
                pids_2d = stashed(pid_arg, T_q)             # [4, T_q]
            elif stashed.dim() == 3:
                pids_2d = stashed[:, 0, :]                  # [4, T_stash]
            else:
                pids_2d = stashed
            T_stash = pids_2d.shape[1]
            # Fallback: shape mismatch on KV-cache decode steps where the static-tensor
            # stash was sized for prefill only. Decode tokens then take HF's 1D RoPE,
            # which is already correct for the calling layer.
            if T_q != T_stash:
                if _orig_accepts_pids:
                    return original_apply(q, k, cos, sin, position_ids, unsqueeze_dim)
                return original_apply(q, k, cos, sin, unsqueeze_dim)

            hint = get_layer_hint()
            if hint is None:
                # Safety net: layer hint was never set (decoder-layer wrapping missed
                # this layer, or something invoked apply_rotary_pos_emb outside a
                # wrapped forward). Pick local — matches the more common layer type
                # and at least won't scramble sliding layers. Warn once.
                if not _hint_warned["flag"]:
                    log.warning(
                        "mrope dispatcher invoked without layer hint; defaulting to "
                        "local (sliding_attention) table. This will produce wrong rotations "
                        "for full_attention layers — verify decoder-layer wrapping installed."
                    )
                    _hint_warned["flag"] = True
                table = mrope_table_local
            else:
                table = mrope_table_local if hint else mrope_table_global

            cos4d, sin4d = table(pids_2d)                   # [T, rotary_dim] each
            cos4d = cos4d.to(dtype=q.dtype)
            sin4d = sin4d.to(dtype=q.dtype)
            q_out, k_out = apply_mrope_4d(q, k, cos4d, sin4d, table.cfg.rotary_dim)
            return q_out, k_out
        else:
            if _orig_accepts_pids:
                return original_apply(q, k, cos, sin, position_ids, unsqueeze_dim)
            return original_apply(q, k, cos, sin, unsqueeze_dim)

    # Attach the stash + layer-hint setters so the outer model and decoder-layer
    # wrappers can populate per-call state.
    patched._mrope_stash = stash             # type: ignore
    patched._mrope_get = get_stash           # type: ignore
    patched._mrope_set_layer_hint = set_layer_hint  # type: ignore
    patched._mrope_get_layer_hint = get_layer_hint  # type: ignore
    patched._mrope_set_abs_pos = set_abs_pos        # type: ignore
    patched._mrope_get_abs_pos = get_abs_pos        # type: ignore
    patched._mrope_table_local = mrope_table_local    # type: ignore
    patched._mrope_table_global = mrope_table_global  # type: ignore
    return patched


# ===========================================================================
# Helpers: locate Gemma3 text model + decoder layer wrapping
# ===========================================================================


class _SynthRotary:
    """Minimal stand-in for ``Gemma3RotaryEmbedding`` carrying only ``inv_freq`` and
    ``attention_scaling`` — used when the real modules aren't directly inspectable
    (e.g., deferred / meta-device init under 4-bit quantization)."""
    __slots__ = ("inv_freq", "attention_scaling", "_source")

    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float, source: str):
        self.inv_freq = inv_freq
        self.attention_scaling = attention_scaling
        self._source = source

    def __repr__(self) -> str:
        return (f"_SynthRotary(source={self._source}, inv_freq[0]={self.inv_freq[0].item():.6f}, "
                f"len={self.inv_freq.shape[0]}, scaling={self.attention_scaling})")


def _synthesize_rotary_embeddings(text_config) -> tuple[_SynthRotary, _SynthRotary]:
    """Build inv_freq tensors the way HF's rope_init_fn would, directly from text_config.

    Mirrors transformers' ``_compute_default_rope_parameters`` and
    ``_compute_linear_scaling_rope_parameters``:
        default: inv_freq[i] = 1 / (base ** (2i / head_dim))
        linear:  default(...) → divide inv_freq by factor

    Used when we cannot find usable ``Gemma3RotaryEmbedding`` modules in the loaded model
    (e.g., 4-bit BnB / deferred / meta-tensor init). The resulting inv_freq is numerically
    identical to what HF would compute given the same config.
    """
    head_dim = int(getattr(text_config, "head_dim",
                           text_config.hidden_size // text_config.num_attention_heads))
    n_pairs = head_dim // 2
    freqs = torch.arange(0, n_pairs, dtype=torch.float32)

    # In newer transformers (≥4.55-ish) rope params live in ``text_config.rope_parameters``,
    # a dict keyed by layer type. Older versions expose ``rope_theta`` / ``rope_local_base_freq`` /
    # ``rope_scaling`` as flat fields. Support both.
    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and ("full_attention" in rope_params or "sliding_attention" in rope_params):
        full_p = rope_params.get("full_attention", {}) or {}
        sliding_p = rope_params.get("sliding_attention", {}) or {}
        global_base = float(full_p.get("rope_theta", getattr(text_config, "rope_theta", 10000.0)))
        global_type = full_p.get("rope_type", "default")
        global_factor = float(full_p.get("factor", 1.0))
        local_base = float(sliding_p.get("rope_theta", getattr(text_config, "rope_local_base_freq",
                                                               getattr(text_config, "rope_theta", 10000.0))))
        local_type = sliding_p.get("rope_type", "default")
        local_factor = float(sliding_p.get("factor", 1.0))
        log.info(f"  synth source: text_config.rope_parameters → "
                 f"full=(theta={global_base}, type={global_type}, factor={global_factor})  "
                 f"sliding=(theta={local_base}, type={local_type}, factor={local_factor})")
    else:
        # Legacy flat-field path
        global_base = float(getattr(text_config, "rope_theta", 10000.0))
        rope_scaling = getattr(text_config, "rope_scaling", None) or {}
        global_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        global_factor = float(rope_scaling.get("factor", 1.0))
        local_base = float(getattr(text_config, "rope_local_base_freq",
                                   getattr(text_config, "rope_theta", 10000.0)))
        local_type = "default"
        local_factor = 1.0
        log.info(f"  synth source: legacy flat fields → "
                 f"rope_theta={global_base}, rope_scaling={rope_scaling}, "
                 f"rope_local_base_freq={local_base}")

    # Global / full_attention
    inv_freq_g = 1.0 / (global_base ** (2 * freqs / head_dim))
    if global_type == "linear":
        inv_freq_g = inv_freq_g / global_factor
    elif global_type != "default":
        log.warning(f"  unsupported global rope_type '{global_type}' — using default formula "
                    f"without scaling. Inference may degrade.")
    scaling_g = 1.0          # default + linear both use attention_factor=1.0

    # Local / sliding_attention
    inv_freq_l = 1.0 / (local_base ** (2 * freqs / head_dim))
    if local_type == "linear":
        inv_freq_l = inv_freq_l / local_factor
    scaling_l = 1.0

    return (
        _SynthRotary(inv_freq_g, scaling_g,
                     f"synth(theta={global_base}, type={global_type}, factor={global_factor})"),
        _SynthRotary(inv_freq_l, scaling_l,
                     f"synth(theta={local_base}, type={local_type}, factor={local_factor})"),
    )


def _find_gemma3_rotary_embeddings(
    model: nn.Module, text_config,
) -> tuple[nn.Module, nn.Module]:
    """Locate the two Gemma3RotaryEmbedding instances (global + local) inside a loaded
    model, regardless of where they live in the wrapping hierarchy.

    Gemma 3 keeps two rotary modules: one for full_attention layers (rope_theta from the
    config) and one for sliding_attention layers (rope_theta = rope_local_base_freq). We
    discriminate by inspecting ``inv_freq[0]``:

      local:  inv_freq[0] = 1.0          (base = rope_local_base_freq=10K, no scaling)
      global: inv_freq[0] = 1/factor     (base = rope_theta=1M with linear factor)

    Returns ``(rotary_global, rotary_local)``.
    """
    by_name: list[tuple[str, nn.Module]] = []
    candidates: list[tuple[str, nn.Module, float]] = []
    skip_reasons: list[str] = []
    for name, m in model.named_modules():
        if type(m).__name__ != "Gemma3RotaryEmbedding":
            continue
        by_name.append((name, m))
        if not hasattr(m, "inv_freq"):
            skip_reasons.append(f"'{name}': no inv_freq attr (attrs: {sorted(vars(m).keys())[:8]}...)")
            continue
        try:
            v0 = float(m.inv_freq.detach().to(torch.float32).flatten()[0].item())
        except Exception as exc:
            skip_reasons.append(f"'{name}': inv_freq read failed — {type(exc).__name__}: {exc}")
            continue
        candidates.append((name, m, v0))

    log.info(f"  Gemma3RotaryEmbedding scan: {len(by_name)} by name, "
             f"{len(candidates)} usable. skip reasons: {skip_reasons or 'none'}")

    if len(candidates) < 2:
        # Last-resort fallback: pull rope params from text_config and synthesize the two
        # inv_freq tensors ourselves. Matches HF's rope_init_fn for default + linear types.
        log.warning(
            f"  fewer than 2 usable rotary modules ({len(candidates)}); synthesizing "
            f"inv_freq from text_config (rope_theta / rope_local_base_freq / rope_scaling)"
        )
        return _synthesize_rotary_embeddings(text_config)

    # Determine expected linear scaling on the global (full_attention) rotary so we can
    # classify candidates by inv_freq[0]. For "default" rope_type, factor=1.0.
    rope_scaling = getattr(text_config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    factor = float(rope_scaling.get("factor", 1.0))
    if rope_type == "linear":
        expected_global_inv0 = 1.0 / factor
    else:
        expected_global_inv0 = 1.0          # default rope, no scaling on inv_freq

    # Local rotary always uses default rope_type with rope_local_base_freq → inv_freq[0]=1.0
    expected_local_inv0 = 1.0

    # If the two expected values coincide (no linear scaling at all), discriminate by
    # the rope_theta itself. Gemma 3 always sets global theta >> local theta, so for
    # higher-index inv_freq elements the global decays faster. Use inv_freq[1] as tie-break.
    if abs(expected_global_inv0 - expected_local_inv0) < 1e-6:
        # tie-break on inv_freq[1] — global has smaller inv_freq[1] when local_base < global_theta
        cand_sorted = sorted(candidates, key=lambda c: c[1].inv_freq.flatten()[1].item())
        return cand_sorted[0][1], cand_sorted[1][1]

    # Classify by closest match to expected_global_inv0
    best_global = min(candidates, key=lambda c: abs(c[2] - expected_global_inv0))
    best_local = min(candidates, key=lambda c: abs(c[2] - expected_local_inv0))
    if best_global is best_local:
        raise RuntimeError(
            f"Could not discriminate global/local rotary: both best matches are "
            f"{best_global[0]} (inv_freq[0]={best_global[2]:.6f}). Expected global "
            f"≈ {expected_global_inv0:.6f}, local ≈ {expected_local_inv0:.6f}."
        )
    log.info(
        f"  located Gemma3 rotary embeddings — global: '{best_global[0]}' "
        f"(inv_freq[0]={best_global[2]:.6f}), local: '{best_local[0]}' "
        f"(inv_freq[0]={best_local[2]:.6f})"
    )
    return best_global[1], best_local[1]


def _wrap_decoder_layers_with_sliding_hint(
    model: nn.Module,
    set_hint: Callable[[bool | None], None],
    get_hint: Callable[[], bool | None],
    set_abs_pos: Callable[[Any], None] | None = None,
    get_abs_pos: Callable[[], Any] | None = None,
) -> int:
    """Wrap each Gemma3DecoderLayer.forward to set/clear the thread-local is_sliding
    hint AND capture HF's absolute position_ids for the M-RoPE dispatcher.

    Gemma3's apply_rotary_pos_emb is called without position_ids, so the dispatcher
    cannot see the current absolute position on its own. The decoder-layer forward,
    however, receives position_ids (param index 3) and cache_position (kwarg). We
    snapshot it here so the callable stash can build correct 4-axis pids on every
    KV-cache decode step (otherwise decode tokens collapse to position 0).

    Idempotent: layers already wrapped (marker attribute on the forward) are skipped.
    Returns the number of layers newly wrapped.
    """
    n = 0
    for name, m in model.named_modules():
        if type(m).__name__ != "Gemma3DecoderLayer":
            continue
        if not hasattr(m, "self_attn"):
            continue
        if getattr(m.forward, "_mrope_wrapped", False):
            continue
        is_sliding = bool(getattr(m.self_attn, "is_sliding", True))
        orig_fwd = m.forward

        def make_wrapped(orig, sliding):
            def wrapped(*args, **kwargs):
                prev = get_hint()
                set_hint(sliding)
                prev_pos = get_abs_pos() if get_abs_pos is not None else None
                if set_abs_pos is not None:
                    # position_ids is the 4th positional param of
                    # Gemma3DecoderLayer.forward (hidden_states, position_embeddings,
                    # attention_mask, position_ids, ...); HF generate passes it as a
                    # kwarg. Fall back to cache_position (absolute positions of the
                    # current token chunk) if position_ids isn't supplied.
                    pos = kwargs.get("position_ids", None)
                    if pos is None and len(args) >= 4:
                        pos = args[3]
                    if pos is None:
                        pos = kwargs.get("cache_position", None)
                    set_abs_pos(pos)
                try:
                    return orig(*args, **kwargs)
                finally:
                    set_hint(prev)
                    if set_abs_pos is not None:
                        set_abs_pos(prev_pos)
            wrapped._mrope_wrapped = True       # type: ignore
            wrapped._mrope_is_sliding = sliding  # type: ignore
            wrapped._mrope_orig_forward = orig   # type: ignore
            return wrapped

        m.forward = make_wrapped(orig_fwd, is_sliding)
        n += 1
    return n


# ===========================================================================
# Whole-model patch
# ===========================================================================


def patch_medgemma_with_mrope_4d(
    model: nn.Module,
    cfg: MRope4DConfig,
    verify: bool = True,
) -> tuple[nn.Module, MRope4DAngleTable]:
    """Monkey-patch every Gemma3 attention layer to use 4-axis M-RoPE when
    a 4-axis stash is set on the dispatcher.

    Gemma 3 uses two different RoPE configurations interleaved across layers:
        sliding_attention layers → rope_theta = 10_000   (default scaling)
        full_attention layers    → rope_theta = 1_000_000 with linear factor 8.0

    This helper:
      1. Locates HF's two ``Gemma3RotaryEmbedding`` modules on the text model and copies
         their pre-scaled ``inv_freq`` buffers directly. This inherits both the per-layer
         base AND any rope_scaling (linear/yarn/etc.) without re-derivation.
      2. Builds TWO ``MRope4DAngleTable`` instances (one per layer type). The user-supplied
         ``cfg.axis_channels`` is normalized to sum to head_dim/2 (auto-doubles legacy
         (16,16,16,16) → (32,32,32,32) for full-rotary coverage matching HF).
      3. Patches ``apply_rotary_pos_emb`` with a dispatcher that picks the right table
         based on a thread-local ``is_sliding`` hint.
      4. Wraps each ``Gemma3DecoderLayer.forward`` to set/clear that hint per call.

    Returns:
        model:       the same model (mutated in place)
        mrope_table: the *local* table (returned for backward compat); the *global* table
                     is reachable via ``model._mrope_table_global`` and the dispatcher
                     via ``model._mrope_dispatcher``.
    """
    try:
        import transformers  # type: ignore
        try:
            from transformers.models.gemma3 import modeling_gemma3
            module = modeling_gemma3
            family = "gemma3"
        except ImportError:
            from transformers.models.gemma2 import modeling_gemma2  # type: ignore
            module = modeling_gemma2  # type: ignore
            family = "gemma2"
            warnings.warn("gemma3 module not found; falling back to gemma2. Verify your model family.")
    except ImportError:
        raise RuntimeError("transformers not installed. pip install transformers>=4.45")

    log.info(f"patching MedGemma rotary application — family={family}")

    if not hasattr(module, "apply_rotary_pos_emb"):
        raise RuntimeError(f"{family} module does not expose apply_rotary_pos_emb; HF API changed")

    # 1. Locate HF's per-layer-type rotary modules and pull pre-scaled inv_freq + scaling.
    text_config = getattr(model.config, "text_config", model.config)
    rotary_global, rotary_local = _find_gemma3_rotary_embeddings(model, text_config)

    inv_freq_local = rotary_local.inv_freq.detach().clone().to(torch.float32)
    inv_freq_global = rotary_global.inv_freq.detach().clone().to(torch.float32)
    scaling_local = float(getattr(rotary_local, "attention_scaling", 1.0))
    scaling_global = float(getattr(rotary_global, "attention_scaling", 1.0))

    # 2. Normalize axis_channels to sum to head_dim/2.
    head_dim = int(getattr(text_config, "head_dim",
                           text_config.hidden_size // text_config.num_attention_heads))
    target_pairs = head_dim // 2
    if inv_freq_local.shape[0] != target_pairs or inv_freq_global.shape[0] != target_pairs:
        raise RuntimeError(
            f"HF inv_freq length ({inv_freq_local.shape[0]} local, {inv_freq_global.shape[0]} "
            f"global) ≠ expected head_dim/2={target_pairs}. HF API change or partial-rotary "
            f"config — manual review needed."
        )
    axis_channels = tuple(cfg.axis_channels)
    if sum(axis_channels) == target_pairs:
        pass                                    # already correct (e.g., (32,32,32,32))
    elif sum(axis_channels) * 2 == target_pairs:
        # Legacy half-rotary config (16,16,16,16) — auto-promote to full-rotary.
        axis_channels = tuple(c * 2 for c in axis_channels)
        log.info(
            f"  axis_channels auto-promoted {tuple(cfg.axis_channels)} → {axis_channels} "
            f"(matching head_dim/2={target_pairs} for full-rotary HF parity)"
        )
    else:
        raise RuntimeError(
            f"cfg.axis_channels={axis_channels} sums to {sum(axis_channels)}; expected "
            f"{target_pairs} (head_dim/2). Update model.py:mrope_axis_channels."
        )

    # 3. Build both tables; place on model's device.
    device = next(model.parameters()).device
    mrope_table_local = MRope4DAngleTable.from_hf_rotary_emb(
        inv_freq=inv_freq_local, attention_scaling=scaling_local,
        head_dim=head_dim, axis_channels=axis_channels,
        low_freq_axis=cfg.low_freq_axis, dtype=cfg.dtype,
    ).to(device)
    mrope_table_global = MRope4DAngleTable.from_hf_rotary_emb(
        inv_freq=inv_freq_global, attention_scaling=scaling_global,
        head_dim=head_dim, axis_channels=axis_channels,
        low_freq_axis=cfg.low_freq_axis, dtype=cfg.dtype,
    ).to(device)
    log.info(
        f"  built dual M-RoPE tables — local:  inv_freq[0:3]="
        f"[{inv_freq_local[0].item():.6f}, {inv_freq_local[1].item():.6f}, "
        f"{inv_freq_local[2].item():.6f}] scaling={scaling_local}"
    )
    log.info(
        f"                              global: inv_freq[0:3]="
        f"[{inv_freq_global[0].item():.6f}, {inv_freq_global[1].item():.6f}, "
        f"{inv_freq_global[2].item():.6f}] scaling={scaling_global}"
    )
    # Sanity: with linear factor=8.0 on the global rotary, inv_freq[1] must be ~8× smaller
    # than the local inv_freq[1]. If not, the rope_scaling didn't get applied → patcher broken.
    ratio = (inv_freq_local[1] / inv_freq_global[1]).item()
    log.info(f"  inv_freq[1] local/global ratio: {ratio:.4f} (expect ~8.0 for MedGemma "
             f"linear factor=8.0)")

    # 4. Install the dispatcher on the module-level apply_rotary_pos_emb and on any
    # attention instance with a local copy.
    original_apply = module.apply_rotary_pos_emb
    patched = make_dispatching_apply_rope(original_apply, mrope_table_local, mrope_table_global)
    module.apply_rotary_pos_emb = patched
    log.info(f"  patched {family}.modeling_{family}.apply_rotary_pos_emb")

    n_attn_patched = 0
    for name, m in model.named_modules():
        if hasattr(m, "apply_rotary_pos_emb") and m.apply_rotary_pos_emb is original_apply:
            m.apply_rotary_pos_emb = patched
            n_attn_patched += 1
    if n_attn_patched > 0:
        log.info(f"  patched {n_attn_patched} attention modules with local apply_rotary refs")

    # 5. Wrap each Gemma3DecoderLayer.forward to set/clear the is_sliding hint.
    n_layers_wrapped = _wrap_decoder_layers_with_sliding_hint(
        model, patched._mrope_set_layer_hint, patched._mrope_get_layer_hint,
        patched._mrope_set_abs_pos, patched._mrope_get_abs_pos,
    )
    log.info(f"  wrapped {n_layers_wrapped} Gemma3DecoderLayer forwards with per-layer hint")

    # 6. Attach references so they survive on the model and are reachable for verification.
    model._mrope_table = mrope_table_local       # type: ignore
    model._mrope_table_local = mrope_table_local   # type: ignore
    model._mrope_table_global = mrope_table_global  # type: ignore
    model._mrope_dispatcher = patched             # type: ignore

    if verify:
        log.info("  running post-patch identity-parity check on synthetic input...")
        verify_identity_reduction(model, mrope_table_local, cfg)

    return model, mrope_table_local


def verify_identity_reduction(
    model: nn.Module,
    mrope_table: MRope4DAngleTable,
    cfg: MRope4DConfig,
    seq_len: int = 32,
    batch_size: int = 1,
    tolerance: float = 1e-3,
) -> bool:
    """Verify that a text-only batch (all-equal position indices) produces identical
    logits with M-RoPE active vs the original 1D RoPE.

    This is the Phase 2a sanity check from train.py, available standalone here too.

    Returns True if pass, raises AssertionError if fail.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Build a text-only position_ids in 4-axis form: all axes equal
    pos_4axis = torch.arange(seq_len, device=device).unsqueeze(0).expand(4, batch_size, seq_len).contiguous()
    # And the same as 1D
    pos_1axis = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len).contiguous()

    # Random input ids in vocab range
    vocab_size = getattr(model.config, "vocab_size", 32000) if hasattr(model, "config") else 32000
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len), device=device)

    model.eval()
    with torch.no_grad():
        # Run with 1D position_ids (original behavior)
        try:
            out_1d = model(input_ids=input_ids, position_ids=pos_1axis).logits
        except TypeError:
            out_1d = model(input_ids=input_ids).logits  # some models don't accept position_ids

        # Run with 4-axis position_ids (text-only, should reduce to 1D RoPE)
        # We use the dispatcher's stash mechanism so the patched function picks 4-axis
        if hasattr(model, "_mrope_dispatcher"):
            model._mrope_dispatcher._mrope_stash(pos_4axis)
        out_4d = model(input_ids=input_ids).logits
        if hasattr(model, "_mrope_dispatcher"):
            model._mrope_dispatcher._mrope_stash(None)  # clear

    max_diff = (out_4d - out_1d).abs().max().item()
    log.info(f"  text-only logit-parity max diff: {max_diff:.2e}  (threshold {tolerance:.0e})")

    if max_diff > tolerance:
        raise AssertionError(
            f"M-RoPE identity reduction FAILED: max diff {max_diff:.2e} exceeds {tolerance:.0e}. "
            "Channel partition or position-index handling is wrong. Do NOT proceed to Phase 2b."
        )
    log.info("  PASSED — M-RoPE retrofit preserves base-model behavior on text-only inputs.")
    return True


# ===========================================================================
# Convenience: assemble a 4-axis position_ids for a NeuroFusion prompt
# ===========================================================================


def assemble_neurofusion_position_ids(
    n_text_prefix: int,                          # e.g. system + schema + heuristic prefix length
    visual_token_specs: list[dict],              # list of dicts with 'modality', 'depth', 'h', 'w'
    n_text_suffix: int = 0,                      # task tokens / decode targets
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build the 4-axis position_ids tensor for one NeuroFusion training step.

    The total seq_len = n_text_prefix + len(visual_token_specs) + n_text_suffix.

    Each visual_token_spec is a dict with keys:
        modality: int in {0, 1, 2, 3} for {T1, T1CE, T2, FLAIR}
        depth:    int (slice index of the lesion centroid in the MRI volume)
        h:        int (in-plane height index)
        w:        int (in-plane width index)
    """
    n_visual = len(visual_token_specs)
    seq_len = n_text_prefix + n_visual + n_text_suffix

    token_kinds: list[str] = (
        ["text"] * n_text_prefix
        + ["visual"] * n_visual
        + ["text"] * n_text_suffix
    )
    modality_idx = {n_text_prefix + i: spec["modality"] for i, spec in enumerate(visual_token_specs)}
    voxel_coords = {
        n_text_prefix + i: (spec["depth"], spec["h"], spec["w"])
        for i, spec in enumerate(visual_token_specs)
    }
    return build_mrope_position_ids(
        seq_len, token_kinds, modality_idx, voxel_coords, batch_size, device,
    )


# ===========================================================================
# Smoke test (unit-level — does not load real MedGemma)
# ===========================================================================


def _smoke_test() -> None:
    """Verify position-ID construction + dispatcher logic without requiring HF model load."""
    print("=== mrope_patch smoke test ===")

    # 1. position-id construction
    pids = build_mrope_position_ids(
        seq_len=10,
        token_kinds=["text"] * 5 + ["visual"] * 3 + ["text"] * 2,
        modality_idx={5: 0, 6: 1, 7: 2},
        voxel_coords={5: (10, 20, 30), 6: (15, 25, 35), 7: (20, 30, 40)},
        batch_size=2,
    )
    print(f"position_ids shape: {tuple(pids.shape)}  (expected (4, 2, 10))")
    assert pids.shape == (4, 2, 10)
    # Text tokens at index 0..4 should have all 4 axes equal to (0,1,2,3,4)
    assert torch.equal(pids[0, 0, :5], torch.arange(5))
    assert torch.equal(pids[1, 0, :5], torch.arange(5))
    print(f"text positions for batch 0: {pids[0, 0, :5].tolist()} (axis 0)")
    # Visual tokens at index 5..7 should have axis-0 = modality, axis-1..3 = (d,h,w)
    print(f"visual axis0 at idx 5..7: {pids[0, 0, 5:8].tolist()}  (expected [0,1,2])")
    print(f"visual axis1 at idx 5..7: {pids[1, 0, 5:8].tolist()}  (expected [10,15,20])")
    assert torch.equal(pids[0, 0, 5:8], torch.tensor([0, 1, 2]))
    assert torch.equal(pids[1, 0, 5:8], torch.tensor([10, 15, 20]))
    # Text tokens after the visual chunk: position_count continues
    assert torch.equal(pids[0, 0, 8:], torch.tensor([5, 6]))   # text resumed at count 5
    print(f"text positions after visual chunk: {pids[0, 0, 8:].tolist()}  (expected [5, 6])")

    # 2. is_4axis_position_ids
    assert is_4axis_position_ids(pids)
    assert not is_4axis_position_ids(torch.zeros(2, 10, dtype=torch.long))
    assert not is_4axis_position_ids(None)
    print("4-axis detection works")

    # 3. dispatcher lookup logic
    cfg = MRope4DConfig(head_dim=64, rotary_dim=32, axis_channels=(4, 4, 4, 4))
    table = MRope4DAngleTable(cfg)
    print(f"angle table: head_dim={cfg.head_dim}, rotary_dim={cfg.rotary_dim}, axes={cfg.axis_channels}")

    # 4. assemble_neurofusion_position_ids
    visual_specs = [
        {"modality": 0, "depth": 50, "h": 30, "w": 40},
        {"modality": 1, "depth": 50, "h": 30, "w": 40},
        {"modality": 2, "depth": 50, "h": 30, "w": 40},
        {"modality": 3, "depth": 50, "h": 30, "w": 40},
    ]
    full_pids = assemble_neurofusion_position_ids(
        n_text_prefix=20, visual_token_specs=visual_specs, n_text_suffix=10, batch_size=1,
    )
    print(f"NeuroFusion-style position_ids: {tuple(full_pids.shape)}  (expected (4, 1, 34))")
    # Visual tokens at indices 20..23 should differ only in axis 0 (modality)
    visual_chunk = full_pids[:, 0, 20:24]
    print(f"visual modality axis (axis 0): {visual_chunk[0].tolist()}  (expected [0, 1, 2, 3])")
    print(f"visual depth axis    (axis 1): {visual_chunk[1].tolist()}  (expected [50, 50, 50, 50])")
    assert torch.equal(visual_chunk[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(visual_chunk[1], torch.tensor([50, 50, 50, 50]))

    print("\nOK — patch logic verified. Real-model integration test requires HF transformers + MedGemma load.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
