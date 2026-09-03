"""
xgrammar_decoder.py — JSON-constrained decoding for NeuroFusion's 9-field schema.

Uses XGrammar (https://github.com/mlc-ai/xgrammar) to enforce the BrainTumorReport
Pydantic schema at every decode step, guaranteeing 100% schema adherence by
construction. From LoV3D (2026): generalist 3D medical VLMs (RadFM, M3D-LaMed)
hit 0% valid JSON zero-shot; constrained decoding fixes this.

What this module provides:

    1. build_grammar_from_schema(BrainTumorReport)
         Convert the Pydantic schema to a JSON Schema, then to an xgrammar
         CompiledGrammar.

    2. XGrammarLogitsProcessor
         A transformers.LogitsProcessor subclass that masks invalid next-tokens
         per step. Drop into HuggingFace generate() via a LogitsProcessorList.

    3. generate_with_xgrammar(model, tokenizer, ...)
         Convenience wrapper. Handles the per-call grammar-matcher reset,
         supports K-sample generation for the semantic-entropy UQ stack.

    4. parse_constrained_output(text)
         Robust parser: even constrained output can have whitespace/escape quirks.

How constrained decoding interacts with the v4 stack:

    Visual tokens + classifier-heuristic prefix + JSON_OPEN -> [LM generates] -> JSON_CLOSE

    The grammar runs from the moment the LM enters the JSON output region. Anything
    before that (system prompt, schema description, classifier heuristics) is
    unconstrained.

Install:
    pip install xgrammar
    pip install transformers

Smoke test:
    python xgrammar_decoder.py
"""
from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Iterable, Protocol

import torch

from schema import BrainTumorReport

log = logging.getLogger("xgrammar_decoder")


# ===========================================================================
# Late-fusion of an auxiliary field-classifier's softmax into constrained decoding
#
# This is the ACL paper's proposed fix for generative collapse: when the LM is
# about to emit a token inside a typed-field value span (e.g. the value of
# "composition"), bias its allowed-token logits by `lambda_f * log p_head(c)`
# where p_head is the field-classification head's softmax over enum classes c.
#
# We expose this as a thin Protocol so the in-model wiring (which needs to know
# the current matcher's grammar-state -> field-name mapping) lives outside the
# LP. A concrete implementation is sketched at the bottom of this file
# (PerFieldLogitFuser) and is the integration point the G1 retrain will use.
# The decoder code path falls through unchanged when no fuser is installed.
# ===========================================================================


class FieldLogitFuser(Protocol):
    """Per-step late-fusion of an auxiliary classifier head's logits.

    The LP calls `observe_token(batch_idx, tok)` for every newly-sampled token
    so the fuser can track grammar state (e.g. "am I currently emitting the
    value of `composition`?"). Then it calls `compute_bias(...)` once per
    active batch slot to get the bias to add to the still-finite logits.

    `compute_bias` may return None to signal "no fusion at this step" (matcher
    not inside a typed-field value span the fuser knows about, or no head
    distribution registered for this case). Otherwise it returns a
    [vocab_size]-shaped tensor of additive log-prior contributions; only
    positions where allowed_mask is True are read by the LP.
    """

    def observe_token(self, batch_idx: int, tok: int) -> None: ...

    def compute_bias(
        self,
        batch_idx: int,
        allowed_mask: torch.Tensor,   # [vocab_size] bool — currently-allowed tokens
        scores_device: torch.device,
        scores_dtype: torch.dtype,
        vocab_size: int,
    ) -> torch.Tensor | None: ...


# ===========================================================================
# Schema -> JSON Schema -> XGrammar compiled grammar
# ===========================================================================


def export_pydantic_to_json_schema(model_class: type = BrainTumorReport) -> dict[str, Any]:
    """Convert the Pydantic BrainTumorReport to a JSON Schema dict.

    Pydantic 2's model_json_schema() does most of the work. We post-process to
    relax a few constraints that XGrammar can't enforce purely structurally:
      - min_length on free-text fields (XGrammar enforces structure, not lengths)
      - cross-field validators (must be enforced at parse-time by Pydantic)

    case_id is intentionally KEPT in the grammar so the matcher can advance
    through a prefilled `{"case_id": "<id>", ` prompt prefix at inference time
    (this matches the training distribution where case_id was the first key).
    The prefill prevents hallucination: the model never has the opportunity to
    pick a wrong ID, because the value is supplied as context.

    Returns a dict suitable for xgrammar.Grammar.from_json_schema().
    """
    schema = model_class.model_json_schema()
    # Strip impossible-for-grammar constraints (lengths, content-validators)
    _strip_grammar_unfriendly(schema)
    # Add additionalProperties: false on every object (tighter validation)
    _set_additional_properties_false(schema)
    return schema


def _strip_grammar_unfriendly(node: Any) -> None:
    """Recursively remove keys XGrammar can't enforce structurally.

    ☠️☠️ `maxLength` IS NO LONGER STRIPPED (Khawaja's ruling, 2026-08-15).

    The docstring above this function said XGrammar "enforces structure, not lengths". That
    WAS true of the version it was written against. It is NOT true of xgrammar 0.1.33, which
    compiles a bounded repetition directly:

        root_prop_6 ::= (("\\"" root_prop_6_1{10, 500} "\\""))

    Measured on the shipped schema: findings at 2000 chars is ACCEPTED, at 2001 REJECTED.

    Why it matters. Stripping `maxLength` left EVERY string field unbounded in the compiled
    grammar, so a decode that wandered inside `findings` had nothing to stop it but
    `max_new_tokens` -- which is exactly how job 920004 produced 23,942 characters of
    prompt echo and ran to its cap. The same signature is on record for the VERBALIZER
    stage, which "produced exactly 2048 tokens" and was repaired with an external 256-token
    cap. Two stages, one cause, and the cause was a schema key being removed before
    compilation.

    ⇒ ★★★ A CONSTRAINT REMOVED BECAUSE A TOOL COULD NOT ENFORCE IT MUST BE RE-EXAMINED WHEN
      THE TOOL CHANGES. The comment stayed true-sounding for as long as nobody re-measured
      it, and it silently licensed an unbounded generation path in the shipped decoder.

    ☠️☠️ THIS PARAGRAPH USED TO READ "a runaway inside a string is STRUCTURALLY IMPOSSIBLE",
    FULL STOP. THAT WAS FALSE AS WRITTEN, and it is the claim I reported upward.
    `maxLength` is declared on exactly TWO fields -- `findings` (2000) and `impression`
    (500). Dumped from the shipped exporter, `case_id` and every item of
    `differential_diagnosis` carry maxLength=None, so they still compile to the unbounded
    right-recursive `basic_string`/`basic_string_sub`. And `differential_diagnosis` is
    emitted BEFORE `findings` in the required key order (hemisphere,
    differential_diagnosis, overall_mass_effect, lesions, findings, impression), so the
    unbounded production is reached FIRST on every single generation.

    What is actually true, field by field:
      - findings, impression        -- structurally bounded; the grammar forces the closing
                                       quote at the bound.
      - case_id, differential_diagnosis items
                                    -- NOT structurally bounded. Defended behaviourally
                                       only: RepetitionGuard's C/D detectors (which act on
                                       any generated span), the 2048 max_new_tokens cap,
                                       and max_whitespace_cnt=24.
    The OBSERVED failure (job 846085, reproduced 1/30) was in `findings`, which is now
    structurally closed. The remaining exposure is a different, unobserved one.
    ⇒ ★★★ "STRUCTURALLY IMPOSSIBLE" IS A CLAIM ABOUT A DOMAIN, AND A CLAIM WITHOUT ITS
      DOMAIN IS NOT A WEAKER CLAIM, IT IS A DIFFERENT AND FALSE ONE. I verified the bound
      on the field I had been debugging and wrote the conclusion over the whole schema.

    ☠️ AND KEEPING maxLength WAS NOT FREE: it switches those two fields onto a bounded
    negated character class that ADMITS raw C0 control bytes and FORBIDS the backslash.
    See escape_raw_control_chars for the measurement and the parse-boundary repair.

    Still stripped, deliberately:
      - `minLength` -- a MINIMUM cannot prevent a runaway and would force the model to pad
        to reach it, which is a content harm to fix a non-problem.
      - `pattern`   -- xgrammar's regex converter silently drops mid-alternation ^/$ anchors
        and falls back to an unconstrained string on constructs it cannot parse. A
        constraint that degrades SILENTLY to permissive is worse than an absent one.
      - `minimum`/`maximum` -- numeric bounds, no bearing on generation length.

    ⚠️ COST, STATED: this changes the constrained-decoding distribution the model was tuned
    under. The 30-run probe judges it; it is not free and is not assumed safe.
    """
    if isinstance(node, dict):
        for k in ("minLength", "minimum", "maximum", "pattern"):
            node.pop(k, None)
        for v in node.values():
            _strip_grammar_unfriendly(v)
    elif isinstance(node, list):
        for v in node:
            _strip_grammar_unfriendly(v)


def _set_additional_properties_false(node: Any) -> None:
    """For every object schema in the tree, set additionalProperties: false."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            _set_additional_properties_false(v)
    elif isinstance(node, list):
        for v in node:
            _set_additional_properties_false(v)


def build_grammar_from_schema(model_class: type = BrainTumorReport):
    """Build a compiled XGrammar from the BrainTumorReport schema.

    Returns the compiled grammar object. Raises if XGrammar isn't installed.
    """
    try:
        import xgrammar as xgr  # type: ignore
    except ImportError:
        raise RuntimeError("pip install xgrammar required for constrained decoding")

    json_schema = export_pydantic_to_json_schema(model_class)
    log.info(f"compiling grammar from {model_class.__name__} JSON schema")
    # ☠️ WHITESPACE MUST BE BOUNDED. By default `from_json_schema` emits `[ \n\t]*` at 105
    # sites in this schema, so every structural position admits an UNBOUNDED whitespace run.
    # Each space is a legal JSON prefix, so a decode that wanders into one never terminates
    # and stops only when max_new_tokens cuts it off. Measured with the shipped schema: the
    # default grammar ACCEPTS a report padded with 19,450 spaces.
    #
    # 24 is the smallest bound that kills it while leaving real output untouched. Swept:
    #     max_whitespace_cnt:  default    8     16     24     32     64
    #     19450-space runaway     True  False  False  False  False  False
    #     compact json.dumps      True   True   True   True   True   True
    #     indent=2                True  False   True   True   True   True
    #     indent=4                True  False  False   True   True   True
    #     schema-maximal report   True   True   True   True   True   True
    # (8 and 16 reject ordinary pretty-printing, which the model may legitimately emit.)
    #
    # ☠️ THIS IS NOT THE CAUSE OF THE JOB-920004 RUNAWAY and must not be cited as its fix.
    # That failure was a prompt-echo loop inside `findings`, which the CONTENT bounds already
    # cover and which `repetition_guard`'s detectors C/D address. This closes a second,
    # independent unbounded-generation path that nothing was watching.
    grammar = xgr.Grammar.from_json_schema(json.dumps(json_schema), max_whitespace_cnt=24)
    log.info(f"  compiled grammar: {grammar}")
    return grammar


# ===========================================================================
# LogitsProcessor — drop into HF generate()
# ===========================================================================


class XGrammarLogitsProcessor:
    """A transformers-compatible LogitsProcessor that enforces an XGrammar.

    Why this is a custom class instead of subclassing LogitsProcessor: keeping it
    transformers-version-agnostic. Duck-typing on `__call__(input_ids, scores)` is
    enough for every transformers version since 4.30.

    Per-batch state is held in the matcher; users must call `reset()` before each
    new sequence. The `generate_with_xgrammar()` helper handles this.
    """

    def __init__(
        self,
        grammar: Any,                                # xgr.CompiledGrammar
        tokenizer: Any,
        vocab_size: int | None = None,
        terminate_when_complete: bool = True,
        field_logit_fuser: "FieldLogitFuser | None" = None,
        # ☠️ DEFAULT FLIPPED True -> False 2026-07-26. This flag is a MEASURED NO-OP and was
        # built on a false diagnosis. `is_terminated()` only flips True AFTER accept_token(EOS),
        # so "matcher_terminated=False at generation end" is a TAUTOLOGY -- it reads False for
        # every generation by construction and could not have come out any other way. Probing
        # the real grammar: EOS is NEVER legal mid-object (allowed sets of 3/6/31681 tokens at
        # 25/50/75% through a valid report, EOS excluded in all) and is the SOLE allowed token
        # at completion. So masking stop tokens only ever removes what the grammar already
        # forbade. The real cause of the 37% schema-invalidity is RUNAWAY GENERATION, handled by
        # repetition_guard.py. Kept as an opt-in flag rather than deleted so the negative stays
        # discoverable. See [[feedback_xgrammar_is_terminated_tautology]].
        forbid_eos_until_terminated: bool = False,
    ) -> None:
        try:
            import xgrammar as xgr  # type: ignore
        except ImportError:
            raise RuntimeError("pip install xgrammar")

        self._xgr = xgr
        self.grammar = grammar
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size or len(tokenizer)
        self.terminate_when_complete = terminate_when_complete
        self._forbid_eos_until_terminated = forbid_eos_until_terminated
        # Every id that would END generation. `eos_token_id` can be a list on some configs,
        # and Mistral's pad often aliases eos — collect them all rather than assume one int,
        # because a single un-masked stop id reopens the whole failure mode.
        _stop: set[int] = set()
        for attr in ("eos_token_id", "pad_token_id"):
            v = getattr(tokenizer, attr, None)
            if isinstance(v, int):
                _stop.add(v)
            elif isinstance(v, (list, tuple)):
                _stop.update(int(x) for x in v if isinstance(x, int))
        # pad is only a stop token when it aliases eos; masking a distinct pad is harmless
        # (the model should never emit it mid-report) but keep the set explicit and logged.
        self._stop_token_ids = sorted(_stop)
        self._n_forced_stop_allowed = 0
        self._stop_mask = None
        log.info(f"XGrammarLP: forbid_eos_until_terminated={forbid_eos_until_terminated} "
                 f"stop_token_ids={self._stop_token_ids}")

        # Build the tokenizer info (XGrammar needs vocab structure to mask correctly)
        # xgr 0.1.x expects either tokenizer or tokenizer_info
        try:
            self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=self.vocab_size,
            )
        except AttributeError:
            self.tokenizer_info = xgr.TokenizerInfo(tokenizer, vocab_size=self.vocab_size)

        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self.compiled = self.compiler.compile_grammar(grammar) if not hasattr(grammar, "_compiled") else grammar
        # Per-sequence matcher; lazy-initialize per batch
        self._matchers: list[Any] = []

        # Optional late-fusion hook: adds a per-step log-prior bias derived from
        # an auxiliary field-classification head's softmax. See FieldLogitFuser
        # below. When None, behaviour is identical to the pre-fusion path.
        self.field_logit_fuser = field_logit_fuser

    def reset(self, batch_size: int = 1) -> None:
        """Call this before each new generation sequence."""
        import xgrammar as xgr  # type: ignore
        self._matchers = [
            xgr.GrammarMatcher(self.compiled) for _ in range(batch_size)
        ]
        # Track how many tokens of input_ids we've fed each matcher so far.
        # First call to __call__ establishes baseline; only tokens APPENDED after
        # that count as "generated" and must be accepted. This handles both
        # input_ids=prompt and inputs_embeds (where input_ids starts empty).
        self._prev_T_per_batch: list[int | None] = [None] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, T]   scores: [B, vocab]   returns scores with invalid tokens -> -inf.

        Uses the post-0.1.x xgrammar API:
          - allocate_token_bitmask(B, V)        : pre-alloc shared bitmask tensor
          - matcher.fill_next_token_bitmask(bm, index=b) : fill batch slot b in place
          - apply_token_bitmask_inplace(scores, bm, indices=active) : mask in place
        """
        B, T = input_ids.shape
        if not self._matchers or len(self._matchers) != B:
            # CRITICAL: this branch DESTROYS any prior matcher state (including
            # the manually-advanced prefill state). Log loudly so callers can
            # detect when their advance has been blown away by an unexpected
            # batch-size mismatch.
            log.warning(
                f"XGrammarLP: rebuilding matchers (was B={len(self._matchers)}, "
                f"got B={B}); prior matcher state is LOST"
            )
            self.reset(B)
        # Debug instrumentation: count invocations + log first-call mask sizes.
        # Tracks how many times the LP is being entered, and what state the
        # matcher is in at each call. Cheap; turn off by setting cfg.debug=False.
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        # DEBUG: on first call of each generate(), introspect matcher state BEFORE snapshot
        if self._call_count == 1 or (self._prev_T_per_batch and self._prev_T_per_batch[0] is None):
            try:
                _bm_pre = self._xgr.allocate_token_bitmask(B, self.vocab_size)
                self._matchers[0].fill_next_token_bitmask(_bm_pre, index=0)
                _s_pre = torch.zeros(1, self.vocab_size)
                self._xgr.apply_token_bitmask_inplace(_s_pre, _bm_pre, indices=[0])
                _n_pre = int((_s_pre[0] != float("-inf")).sum().item())
                log.warning(
                    f"XGrammarLP PRE-SNAPSHOT call #{self._call_count}: B={B} T={T} "
                    f"prev_T={self._prev_T_per_batch[0] if self._prev_T_per_batch else 'NA'} "
                    f"matcher_terminated={self._matchers[0].is_terminated()} "
                    f"n_allowed_before_any_snapshot_or_advance={_n_pre}"
                )
            except Exception as e:
                log.warning(f"PRE-SNAPSHOT introspection failed: {e}")

        # Advance each matcher with tokens newly appended since the previous call.
        # On the first call, prev_T is None → snapshot baseline T (prompt tokens, if
        # any, are NOT fed to the grammar). On subsequent calls, accept every token
        # in positions [prev_T, T). With inputs_embeds, prev_T starts at 0 and one
        # new token arrives each step; with input_ids=prompt, prev_T starts at
        # len(prompt) so the prompt is ignored.
        for b in range(B):
            if self._prev_T_per_batch[b] is None:
                self._prev_T_per_batch[b] = T
                continue
            for pos in range(self._prev_T_per_batch[b], T):
                if self._matchers[b].is_terminated():
                    break
                tok = int(input_ids[b, pos].item())
                # DEBUG: log first 10 sampled tokens, see what model is producing
                if self._call_count <= 11:
                    try:
                        decoded = self.tokenizer.decode([tok])
                    except Exception:
                        decoded = "<?>"
                    log.warning(
                        f"  SAMPLED-TOK call #{self._call_count} pos={pos} "
                        f"tok={tok} decoded={decoded!r}"
                    )
                accepted = self._matchers[b].accept_token(tok)
                if not accepted:
                    try:
                        decoded = self.tokenizer.decode([tok])
                    except Exception:
                        decoded = "<?>"
                    log.warning(
                        f"!! SAMPLED-TOKEN-NOT-IN-ALLOWED-SET !! call #{self._call_count} "
                        f"pos={pos} tok={tok} decoded={decoded!r} — mask was NOT respected"
                    )
                # Forward the token to the late-fusion fuser so its field-state
                # tracker can update before compute_bias is called this step.
                if self.field_logit_fuser is not None:
                    try:
                        self.field_logit_fuser.observe_token(b, tok)
                    except Exception as e:  # don't let a fuser bug kill decode
                        log.warning(f"field_logit_fuser.observe_token failed (b={b}, tok={tok}): {e}")
            self._prev_T_per_batch[b] = T

        # XGrammar API constraint: fill_next_token_bitmask must run on CPU,
        # apply_token_bitmask_inplace must match the logits device. So fill on CPU,
        # transfer once, then apply.
        bitmask_cpu = self._xgr.allocate_token_bitmask(B, self.vocab_size)
        active: list[int] = []
        for b in range(B):
            if self._matchers[b].is_terminated():
                if self.terminate_when_complete:
                    eos = self.tokenizer.eos_token_id or 0
                    mask = torch.full_like(scores[b], float("-inf"))
                    mask[eos] = 0.0
                    scores[b] = mask
                continue
            self._matchers[b].fill_next_token_bitmask(bitmask_cpu, index=b)
            active.append(b)

        if active:
            # CRITICAL FIX 2026-05-17: apply_token_bitmask_inplace produces wrong
            # masks on CUDA in this transformers/xgrammar/bitsandbytes stack
            # (PRE-SNAPSHOT shows matcher allows 99 tokens, POST-MASK shows 131
            # non-inf in scores — apply isn't fully masking outside positions).
            # Bypass via pure PyTorch: unpack the bitmask to bool, manually mask.
            #
            # Note: scores.shape[-1] (model lm_head vocab, e.g. 262208) may
            # EXCEED the tokenizer's vocab_size (e.g. 262145) — the model has
            # padded the lm_head for alignment. Mask those extras as DISALLOWED
            # (they're never legitimate tokens for the constrained grammar).
            S = scores.shape[-1]
            V_tok = self.vocab_size  # tokenizer-vocab size used for bitmask
            bm = bitmask_cpu  # [B, n_words] int32
            shifts = torch.arange(32, dtype=torch.int32)
            bits = (bm.unsqueeze(-1) >> shifts) & 1  # [B, n_words, 32]
            unpacked = bits.reshape(B, -1)[:, :V_tok].bool()  # [B, V_tok]
            if S > V_tok:
                pad = torch.zeros(B, S - V_tok, dtype=torch.bool)
                bool_mask = torch.cat([unpacked, pad], dim=1)  # [B, S]
            else:
                bool_mask = unpacked[:, :S]
            bool_mask = bool_mask.to(scores.device)
            neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
            for b in active:
                scores[b] = torch.where(bool_mask[b], scores[b], neg_inf)

            # ---------------------------------------------------------------------------
            # FORBID EOS WHILE THE GRAMMAR IS UNSATISFIED  (fix 2026-07-26)
            #
            # `active` is exactly the set of batch rows whose matcher is NOT terminated, i.e.
            # the grammar is not in an accepting state, i.e. stopping here produces an
            # INCOMPLETE report. Measured on the NF_knight 1.0 dumps
            # (~/scratch/predictions_STRIPID_best/*/train.log): **348/348 generations ended
            # with matcher_terminated=False**, 248 of them (71 %) because the model sampled
            # EOS and generation stopped anyway:
            #     [gen-stop] n_new=329/2048 hit_cap=False last_tok=2 is_eos=True
            #                matcher_terminated=False
            # `close_json_prefix` then silently closed the open containers, so the output
            # PARSED and looked fine while missing `findings`/`impression` entirely — 20/54
            # GLI and 11/60 MEN reports scored zero on every field for this reason alone.
            #
            # The bitmask from fill_next_token_bitmask should already exclude the stop token
            # here; empirically it does not (some xgrammar builds always leave the stop token
            # set and expect the caller to own termination). Masking it explicitly is
            # defensive and cannot be wrong: when the matcher is not terminated the grammar
            # genuinely does not admit end-of-string, so EOS is an invalid continuation by
            # definition. This is the exact mirror of the `is_terminated()` branch above,
            # which FORCES EOS once the grammar IS satisfied.
            #
            # Runaway generations (the other 100/348, which hit the 2048 cap) are NOT fixed by
            # this — they need repetition control, tracked separately.
            # ☠️ NEVER EMPTY THE DISTRIBUTION. First attempt at this fix masked the stop tokens
            # unconditionally and crashed job 66456570 (test_men) with
            #     RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            # from torch.multinomial: at some positions the grammar's surviving allowed set is the
            # stop token ALONE, so removing it left every logit at -inf -> softmax -> nan. Those
            # are exactly the positions that produced the premature-EOS stops this fix targets, so
            # the crash hit the common case, not a rare one.
            #
            # Correct behaviour there is to LET THE MODEL STOP and let the caller record an
            # incomplete generation -- not to hand an invalid distribution to the sampler. Masking
            # is therefore conditional on at least one non-stop token remaining.
            # PERF: this runs on EVERY decoded token, so it must not synchronise. The first
            # version called .item() twice per batch row per step (~1500 GPU->CPU syncs per
            # case) and slowed predict_m6 from ~50 s/case to ~197 s/case, which would have
            # timed out job 66458767 at the 8 h wall. The branch is therefore evaluated
            # BRANCHLESSLY on-device with torch.where, so no value is ever read back to Python
            # in the hot path.
            if self._forbid_eos_until_terminated:
                stop_ids = [s for s in (getattr(self, "_stop_token_ids", None) or [])
                            if 0 <= s < scores.shape[-1]]
                if stop_ids:
                    if (self._stop_mask is None
                            or self._stop_mask.shape[-1] != scores.shape[-1]
                            or self._stop_mask.device != scores.device):
                        m = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
                        m[torch.tensor(stop_ids, device=scores.device)] = True
                        self._stop_mask = m
                    sm = self._stop_mask
                    for b in active:
                        finite = torch.isfinite(scores[b])
                        # has_nonstop stays a 0-dim CUDA tensor; broadcasting it through
                        # torch.where keeps the entire decision on-device (no sync).
                        has_nonstop = (finite & ~sm).any()
                        masked = torch.where(sm, neg_inf, scores[b])
                        scores[b] = torch.where(has_nonstop, masked, scores[b])
                        # DIAGNOSTIC, deliberately OFF the hot path. Reading has_nonstop back
                        # costs a sync, so only sample it. ☠️ This comment previously claimed it showed the grammar
                        # reaches a DEAD STATE at object close. That was WRONG -- see the
                        # forbid_eos_until_terminated docstring above. The counter it read is a
                        # tautology and the sampling made "0 occurrences" meaningless too.
                        if self._call_count % 128 == 0 and not bool(has_nonstop.item()):
                            self._n_forced_stop_allowed += 1
                            if self._n_forced_stop_allowed <= 5:
                                log.warning(
                                    f"XGrammarLP: stop token is the ONLY allowed continuation at "
                                    f"T={T} while matcher_terminated=False -> permitting EOS; this "
                                    f"generation will be an INCOMPLETE prefix (sampled "
                                    f"occurrence #{self._n_forced_stop_allowed})")
            # Debug instrumentation (kept after switch): verify 1-bit count vs
            # post-mask finite count.
            if self._call_count <= 3:
                n_bits = int(bool_mask[0].sum().item())
                n_finite = int(scores[0].isfinite().sum().item())
                log.warning(
                    f"  POST-MANUAL-MASK: bitmask_1bits={n_bits} S={S} V_tok={V_tok} "
                    f"scores_finite_after_manual={n_finite}"
                )

            # Late-fusion bias (ACL plan §Methodology). If a fuser is registered,
            # ask it for a per-token log-prior bias derived from an auxiliary
            # classification head's softmax and add it to the still-finite tokens.
            # The fuser sees the current allowed-token bool mask and the batch
            # index; it returns either None (no bias this step) or a [S] tensor
            # of additive log-prior contributions. Tokens outside the allowed set
            # are left at -inf (the bias only re-weights among allowed tokens).
            if self.field_logit_fuser is not None:
                for b in active:
                    bias = self.field_logit_fuser.compute_bias(
                        batch_idx=b,
                        allowed_mask=bool_mask[b],
                        scores_device=scores.device,
                        scores_dtype=scores.dtype,
                        vocab_size=S,
                    )
                    if bias is not None:
                        # Only re-weight tokens that survived the mask
                        scores[b] = torch.where(bool_mask[b], scores[b] + bias, scores[b])

        if self._call_count <= 3 or self._call_count % 100 == 0:
            try:
                n_allowed_post = int((scores[0] != float("-inf")).sum().item())
                n_finite_post = int(scores[0].isfinite().sum().item())
            except Exception:
                n_allowed_post = -1
                n_finite_post = -1
            log.warning(
                f"XGrammarLP call #{self._call_count}: B={B} T={T} "
                f"prev_T_now={self._prev_T_per_batch[0]} "
                f"matcher_terminated={self._matchers[0].is_terminated()} "
                f"active_count={len(active)} n_allowed_post_mask={n_allowed_post} "
                f"n_finite_post={n_finite_post}"
            )

        return scores

    def _mask_scores(self, scores: torch.Tensor, allowed_bitmask: Any) -> torch.Tensor:
        """Apply allowed-token bitmask to score vector."""
        # XGrammar bitmask is a torch.Tensor of int32 packed bits, length ceil(vocab/32)
        try:
            self._xgr.apply_token_bitmask_inplace(scores.unsqueeze(0), allowed_bitmask.unsqueeze(0))
        except (AttributeError, RuntimeError):
            # Fallback: manual mask
            allowed_set = self._extract_allowed_set(allowed_bitmask)
            mask = torch.full_like(scores, float("-inf"))
            mask[list(allowed_set)] = 0.0
            scores = scores + mask
        return scores

    def _extract_allowed_set(self, bitmask: Any) -> set[int]:
        """Convert a packed-int bitmask to a Python set of allowed token IDs."""
        if isinstance(bitmask, torch.Tensor):
            arr = bitmask.cpu().numpy()
        else:
            arr = bitmask
        allowed: set[int] = set()
        for word_idx, word in enumerate(arr):
            for bit in range(32):
                if int(word) & (1 << bit):
                    tok_id = word_idx * 32 + bit
                    if tok_id < self.vocab_size:
                        allowed.add(tok_id)
        return allowed


# ===========================================================================
# generate_with_xgrammar — convenience wrapper
# ===========================================================================


@torch.no_grad()
def generate_with_xgrammar(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,                # [B, T_prefix] already-tokenized prompt
    grammar: Any | None = None,
    schema: type = BrainTumorReport,
    max_new_tokens: int = 512,
    n_samples: int = 1,                       # K-sample for semantic-entropy UQ
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool | None = None,
    **generate_kwargs: Any,
) -> list[list[str]]:
    """Generate K JSON samples per input, each constrained to BrainTumorReport schema.

    Returns:
        list of length B, each containing n_samples generated strings.
    """
    try:
        from transformers import LogitsProcessorList  # type: ignore
    except ImportError:
        raise RuntimeError("transformers required for generate_with_xgrammar")

    if grammar is None:
        grammar = build_grammar_from_schema(schema)

    processor = XGrammarLogitsProcessor(grammar, tokenizer)
    logits_processors = LogitsProcessorList([processor])

    if do_sample is None:
        do_sample = n_samples > 1 or temperature > 0

    B = prompt_ids.shape[0]
    results: list[list[str]] = [[] for _ in range(B)]

    for k in range(n_samples):
        processor.reset(batch_size=B)

        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            logits_processor=logits_processors,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # Strip the prompt; decode only the generated suffix
        gen_ids = out[:, prompt_ids.shape[1]:]
        for b in range(B):
            text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
            results[b].append(text)
        log.debug(f"sample {k+1}/{n_samples} generated; lengths={[len(r[-1]) for r in results]}")

    return results


# ===========================================================================
# Robust parse — even constrained output benefits from a guard
# ===========================================================================


def normalize_lesion_indices(obj: dict) -> dict:
    """Sort lesions by lesion_index, then renumber 1..N.

    XGrammar can constrain the *type* of lesion_index (integer enum 1..4) but
    cannot enforce ordering between lesion entries — schema.validate_lesion_indices
    requires `1..N strictly increasing`, which is sometimes violated by free
    generation (e.g. [3, 4, 1, 2]).

    This is legitimate post-processing: it does NOT change clinical content, only
    the contiguous-index invariant. Reported as ablation A10 in the paper.
    Production deployment would do the same normalization.
    """
    if not isinstance(obj, dict):
        return obj
    lesions = obj.get("lesions")
    if not isinstance(lesions, list) or not lesions:
        return obj
    # Sort by lesion_index (missing -> push to end), then renumber 1..N.
    def _idx(L):
        v = L.get("lesion_index") if isinstance(L, dict) else None
        return v if isinstance(v, int) else 999
    lesions = sorted(lesions, key=_idx)
    for i, L in enumerate(lesions, start=1):
        if isinstance(L, dict):
            L["lesion_index"] = i
    obj["lesions"] = lesions
    return obj


def escape_raw_control_chars(text: str) -> str:
    """Escape raw C0 control bytes that appear INSIDE JSON string values.

    ☠️☠️ THIS EXISTS BECAUSE KEEPING `maxLength` WAS NOT FREE, AND I SHIPPED IT BELIEVING
    IT WAS. Retaining maxLength switches xgrammar from the unbounded `basic_string` rule
    (quote / escape / non-control) onto a length-bounded repetition of the negated class
    `[^"\\\r\n]`. That class excludes exactly four characters, so it ADMITS every other C0
    control byte -- NUL, BEL, VT, ESC and TAB. Measured through the shipped exporter:

        character          OLD (maxLength stripped)   NEW (maxLength kept)
        TAB, NUL, BEL, VT, ESC        masked                  LEGAL
        backslash                     legal                   masked

    A raw C0 byte inside a JSON string is invalid JSON (RFC 8259 s7), so the length fix
    opened a NEW route to the very ReportParseError it was introduced to close. My own
    verification could not see it: I tested LENGTH (2000 ok / 2001 rejected / 23000
    rejected) and concluded the change was length-only.
    ⇒ ★★★ A CONSTRAINT ADDED TO A GRAMMAR CAN CHANGE THE GRAMMAR ELSEWHERE. `maxLength`
      reads like a scalar bound; it actually selects a different production. Test the
      dimension you changed AND the dimension you did not.

    The grammar's character class is generated inside xgrammar from the schema and is not
    ours to edit, so the repair belongs at the parse boundary. It is content-PRESERVING:
    the bytes become their legal escape (\\t, \\n, \\r) or a \\uXXXX escape, so nothing the
    model wrote is discarded -- unlike stripping, which would silently delete text from a
    clinical report.

    ☠️ IN-STRING TRACKING IS NOT OPTIONAL. Raw \\n and \\t BETWEEN tokens are legal JSON
    whitespace; the same bytes inside a string are not. And `differential_diagnosis` items
    and `case_id` carry no maxLength, so they still compile to `basic_string` and CAN
    contain backslash escapes -- the scanner must honour an escape or it will mis-track the
    string state on exactly those fields.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch); escaped = False; continue
            if ch == "\\":
                out.append(ch); escaped = True; continue
            if ch == '"':
                out.append(ch); in_string = False; continue
            if ch < "\x20":
                out.append({"\t": "\\t", "\n": "\\n", "\r": "\\r"}.get(
                    ch, "\\u%04x" % ord(ch)))
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def close_json_prefix(text: str) -> str:
    """Close an unterminated JSON prefix by popping its open-container stack LIFO.

    Grammar-constrained decoding guarantees a syntactically valid PREFIX, so a
    truncated generation lacks only its closers — but they must be emitted
    innermost-first. The older count-based repair appended every "}" and then
    every "]", which yields "}}]" where a {"lesions": [{...  prefix needs "}]}".
    Counting was also string-unaware: a brace inside a findings string inflated
    the tally. This scans with an explicit stack, skipping string interiors and
    honouring backslash escapes, then closes what is actually open.

    Also repairs the three shapes a mid-structure stop leaves behind: an open
    string, a dangling comma, and a key whose value never arrived.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()

    out = text
    if in_str:                      # stopped mid-string -> terminate it first
        out += '"'
    out = out.rstrip()
    if out.endswith(","):           # dangling separator before the stop
        out = out[:-1].rstrip()
    if out.endswith(":"):           # key emitted, value never arrived
        out += " null"
    for opener in reversed(stack):  # innermost first
        out += "}" if opener == "{" else "]"
    return out


def parse_constrained_output(text: str, schema: type = BrainTumorReport) -> Any | None:
    """Parse a constrained-decoded output back into a schema instance.

    Even with grammar-constrained decoding, edge cases (premature EOS, escaped
    quotes near the boundary) can produce partial JSON. Try strict parse first;
    fall back to bracket-matching + unterminated-string repair, then apply
    lesion-index normalization (legitimate post-processing — see ablation A10).
    """
    text = text.strip()

    # Strip any leading non-{ junk (in case the prompt boundary leaked)
    if "{" in text:
        text = text[text.index("{"):]

    # Close whatever containers are still open, innermost-first, and repair an
    # open string / dangling comma / valueless key. Replaces the older
    # count-based append, which emitted closers in the wrong order and counted
    # braces inside string values (see close_json_prefix).
    text = close_json_prefix(text)

    # ☠️ BEFORE close_json_prefix's repairs are handed to json.loads: a raw C0 byte inside a
    # string is invalid JSON and became emittable when maxLength was kept. See
    # escape_raw_control_chars for the measured character-class change.
    text = escape_raw_control_chars(text)

    # Strategy: pre-normalize lesion indices via json.loads if possible (so the
    # strict @model_validator passes), then validate. If json.loads fails (rare
    # — pydantic-core's parser is more permissive), fall back to direct
    # model_validate_json which can handle some edge cases json.loads can't.
    try:
        d = json.loads(text)
        d = normalize_lesion_indices(d)
        return schema.model_validate(d)
    except json.JSONDecodeError as e:
        log.debug(f"json.loads failed ({e}); falling back to model_validate_json")
    except Exception as e:
        log.debug(f"parse-then-validate failed: {e}")
        # If json.loads succeeded but model_validate failed (e.g. content
        # validators), no point retrying with model_validate_json on the same
        # text — it would also fail. Only retry if json.loads itself failed.
        return None
    # Fallback for inputs json.loads can't handle.
    try:
        return schema.model_validate_json(text)
    except Exception as e2:
        log.warning(f"parse failed even after bracket repair + index norm: {e2}")
        return None


# ===========================================================================
# Smoke test (no transformers / xgrammar required)
# ===========================================================================


def _smoke_test() -> None:
    """Verify schema export + grammar build (mock when xgrammar absent)."""
    print("=== xgrammar_decoder smoke test ===")

    # 1. JSON schema export
    print("[1/4] export Pydantic schema -> JSON schema")
    js = export_pydantic_to_json_schema(BrainTumorReport)
    print(f"  top-level type: {js.get('type')}")
    print(f"  required fields: {js.get('required', [])}")
    print(f"  has $defs (subschemas): {'$defs' in js}")
    if "$defs" in js:
        print(f"  $defs keys: {list(js['$defs'].keys())[:5]}{'...' if len(js['$defs']) > 5 else ''}")
    assert js.get("type") == "object"
    assert "lesions" in js.get("required", []) or "lesions" in js.get("properties", {})

    # 2. Grammar build (skip if xgrammar not installed)
    print("[2/4] compile XGrammar from JSON schema")
    try:
        import xgrammar  # noqa: F401
        grammar = build_grammar_from_schema(BrainTumorReport)
        print(f"  grammar compiled: {grammar}")
    except (RuntimeError, ImportError) as e:
        print(f"  SKIPPED — {e}")

    # 3. Robust parse on known-good and known-bad inputs
    print("[3/4] robust parse_constrained_output")
    good = json.dumps({
        "case_id": "TR01", "hemisphere": "right",
        "differential_diagnosis": ["GBM"], "overall_mass_effect": "moderate",
        "lesions": [{
            "lesion_index": 1, "location": "frontal", "composition": "necrotic",
            "enhancement_pattern": "heterogeneous",
            "surrounding_effects": {"edema_severity": "mild", "mass_effect": "present", "midline_shift": "none"},
            "involvement": {"ventricular_compression": "none", "brainstem_compression": "none", "midbrain_compression": "none"},
            "axis_shift": "none",
        }],
        "findings": "There is a 3 cm enhancing lesion with heterogeneous enhancement in the right frontal lobe with surrounding edema and mass effect.",
        "impression": "Right frontal lobe mass with edema and mass effect.",
    })
    parsed = parse_constrained_output(good)
    print(f"  good input parsed: {parsed is not None}  case_id={parsed.case_id if parsed else 'None'}")
    assert parsed is not None

    # Truncated input — bracket-balance repair should fix
    truncated = good[:-2]  # chop the last "}}"
    parsed2 = parse_constrained_output(truncated)
    print(f"  truncated input parsed (after bracket repair): {parsed2 is not None}")

    # 4. LogitsProcessor instantiation requires xgrammar + transformers
    print("[4/4] LogitsProcessor instantiation (requires xgrammar + transformers)")
    try:
        import transformers, xgrammar  # noqa: F401
        # Without an actual tokenizer this can't go further than the import
        print("  imports OK; full instantiation requires a HF tokenizer")
    except ImportError:
        print("  SKIPPED — xgrammar or transformers not installed")

    print("\nOK")


# ===========================================================================
# PerFieldLogitFuser — concrete late-fusion reference implementation
#
# The ACL paper's headline mechanism. Tracks the input_ids token stream emitted
# during constrained decoding, detects when the matcher has entered a typed-
# field value span (by matching the recently-emitted tokens against precomputed
# field-key token sequences like `"composition":`), and for the first token of
# the value position returns a [vocab]-shaped bias whose nonzero entries
# correspond to the first-token of each enum-class string.
#
# The bias at token t is  lambda_f * log(p_head(class(t)) + eps)  for each
# token t that is the FIRST token of some known enum class string under the
# tokenizer's BPE. Adding this to the LM's logits implements:
#   log p_fused(t) ∝ log p_LM(t) + lambda_f * log p_head(class(t))
# which is exactly the convex log-domain late fusion described in §Methodology.
#
# Assumptions / limits:
#   - Enum class strings are short. We bias ONLY the first-token; the rest of
#     the enum string is forced by the grammar anyway because XGrammar already
#     restricts the allowed set to the prefix-completable enum members.
#   - We track `current_field` via a small ring buffer of the last N tokens
#     decoded back to text. This is approximate but adequate: the LP only
#     sees one new token per step and the field-key sequences are short
#     (e.g. `"composition":` is ~4 tokens after BPE).
#   - The state tracker resets `current_field` when it sees `,` `}` `]` —
#     these terminate a value span in JSON. New value spans start when the
#     suffix matches a known field-key sequence.
#
# Usage (per case, per batch slot):
#   fuser = PerFieldLogitFuser(tokenizer, lambda_per_field={"composition": 1.5})
#   fuser.register_case(batch_idx=0, field_softmax={
#       "composition": {"solid": 0.05, "cystic": 0.20, ..., "necrotic-cystic": 0.40},
#       ...
#   })
#   processor = XGrammarLogitsProcessor(grammar, tok, field_logit_fuser=fuser)
#   # generate as usual; per-step LP calls observe_token + compute_bias
#   fuser.reset_case(batch_idx=0)  # before the next case
#
# Calibration of lambda_per_field: the G1 retrain pass calibrates lambda_f on
# the 50-case conformal-calibration split by minimizing a cross-entropy
# objective. For L3 offline gate (scripts/run_late_fusion_offline.py) the
# equivalent is the alpha sweep — fused = alpha * LM + (1-alpha) * head.
# ===========================================================================


# Enum classes the fuser will bias. Matches code/model.py FieldClassificationHead.ENUM_STRINGS.
# Kept here so the fuser can be used without importing model.py (which pulls
# heavy training-side deps).
_DEFAULT_FIELD_ENUM_STRINGS: dict[str, list[str]] = {
    "location": [
        "frontal", "temporal", "parietal", "occipital", "insular", "cerebellar",
        "brainstem", "parieto-temporal", "temporo-parietal", "fronto-parietal",
        "fronto-temporal", "temporo-occipital", "parieto-occipital", "thalamic",
        "basal-ganglia", "corpus-callosum",
    ],
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

# Tokens whose decoded form indicates "end of value span" — when one of these
# is observed, the fuser resets current_field to None. The actual decoding is
# tokenizer-dependent, so we match by decoded string content.
_VALUE_END_DECODED = {",", "}", "]", '"', " ,", " }", " ]"}


# ---------------------------------------------------------------------------
# differential_diagnosis (list[str]) array-slot fusion — EVAL-TIME ONLY (L2).
#
# Unlike the scalar enum fields above, differential_diagnosis is `list[str]`
# (schema.py), so at decode the value opens with an array + string: after the
# `"differential_diagnosis":` key come the structural tokens `[` then `"`
# (often merged, e.g. Mistral emits ` ["`=7367 or `["`=2221) and only THEN the
# first VALUE token (the top-1 dx string's first content token, e.g. `M`/`G`/
# `Met`). PerFieldLogitFuser handles this with an array phase machine
# (arr_open/str_open/content_started) so the classifier-free-guidance bias lands
# on the first CONTENT token of differential_diagnosis[0], leaving the prose free.
#
# The "classes" are the canonical cohort dx strings. Keep IN SYNC with
# scripts/probe_anchor.py::CANON so a decode-time bias targets the SAME string
# the post-hoc override (probe_anchor.write_override) would write — that keeps
# the two anchoring paths interchangeable (see the RED-TEAM note on the fuser).
#
# GATED: dx is added to the fuser maps ONLY when enable_dx_fusion=True. The
# DEFAULT fuser (enable_dx_fusion=False, used by generate()) never touches dx —
# byte-identical to the pre-dx path.
_DX_FIELD = "differential_diagnosis"
_DEFAULT_DX_COHORT_STRINGS: dict[str, str] = {
    "GLI": "Glioblastoma",
    "MEN": "Meningioma",
    "MET": "Metastasis",
}


# Fields the field-head is empirically untrustworthy on (out/field_head_test.json,
# 2026-05-28). Fusing the LM's logits with these fields' head distributions injects
# noise or trivial-degenerate signal, so DEFAULT_LAMBDA_PER_FIELD turns fusion off
# unless the caller explicitly overrides:
#
#   location: head macro-F1 0.030 (≈ noise floor across 16 classes)
#   midbrain_compression: head macro-F1 1.0 (almost certainly degenerate on the
#       n=39 test set — head likely learned a single-class shortcut; trusting its
#       distribution would force one class regardless of LM evidence)
#
# Callers can override by passing lambda_per_field={"location": 0.5, ...}.
_UNTRUSTWORTHY_HEAD_FIELDS = ("location", "midbrain_compression")

DEFAULT_LAMBDA_PER_FIELD: dict[str, float] = {
    fname: (0.0 if fname in _UNTRUSTWORTHY_HEAD_FIELDS else 1.0)
    for fname in _DEFAULT_FIELD_ENUM_STRINGS
}


class PerFieldLogitFuser:
    """Concrete FieldLogitFuser. See module-top comment for the math + protocol.

    Per-field lambda controls the fusion strength:
      lambda_f = 0.0  ->  no fusion (identical to vanilla constrained decoding)
      lambda_f = 1.0  ->  full log-prior bias from the head's softmax
      lambda_f > 1.0  ->  amplified (sharper) head bias (useful when the head
                          is high-precision but the LM's distribution is too flat)

    Default policy (2026-05-28 field-head gate, see DEFAULT_LAMBDA_PER_FIELD):
      lambda=0 for `location` (head ≈ noise) and `midbrain_compression` (head
      almost certainly degenerate at 1.0 on n=39). lambda=1 elsewhere.
      Pass lambda_per_field=... to override per field.

    Calibration: in the G1 retrain, train lambda per field on the conformal-
    calibration split. For the L3 offline gate, the alpha-sweep equivalent is
    in scripts/run_late_fusion_offline.py.

    differential_diagnosis array-slot fusion (L2, eval-time only) — enable_dx_fusion:
      When enable_dx_fusion=True the fuser ALSO biases the free-text
      differential_diagnosis[0] slot (a `list[str]`) toward a cohort's canonical
      dx string, classifier-free-guidance style — lambda>1 amplifies. This is the
      lever used by model.generate_draft_commit(dx_softmax_per_batch=...). The
      array open (`[` then `"`, often merged as ` ["`/`["`) is walked by an
      internal phase machine so the bias lands on the first CONTENT token of the
      top-1 element, NOT the structural bracket/quote. Register the target with
      register_case(b, {canonical_dx_string: prob}). DEFAULT (enable_dx_fusion=
      False, used by generate()) is byte-identical to the pre-dx path: dx is never
      added to the field maps and the array code paths are never entered.

      RED-TEAM (JSON-break safety): a too-strong lambda can push a first token the
      XGrammar matcher then cannot complete into valid JSON. For a free-text string
      slot any letter is grammar-valid so the risk is low, but callers MUST still
      validate the committed JSON (json.loads + schema) and, on ANY invalid/empty
      output, fall back to the post-hoc override scripts/probe_anchor.py::
      write_override (which sets differential_diagnosis[0] to the same CANON string
      post-generation). Never let a decode-time bias silently corrupt the JSON.
    """

    def __init__(
        self,
        tokenizer: Any,
        lambda_per_field: dict[str, float] | None = None,
        field_enum_strings: dict[str, list[str]] | None = None,
        recent_token_window: int = 16,
        eps: float = 1e-6,
        enable_dx_fusion: bool = False,
        dx_cohort_strings: dict[str, str] | None = None,
    ) -> None:
        import numpy as np  # local import keeps the module-top deps minimal

        self.tokenizer = tokenizer
        # Start from the data-anchored default policy, then layer caller overrides.
        # This makes the "untrustworthy field is auto-disabled" behaviour the
        # default while still letting callers re-enable fusion on those fields
        # for an explicit ablation (e.g. lambda_per_field={"location": 0.0,
        # "midbrain_compression": 0.5}).
        merged_lambda = dict(DEFAULT_LAMBDA_PER_FIELD)
        if lambda_per_field:
            merged_lambda.update(lambda_per_field)
        self.lambda_per_field = merged_lambda
        # None => data-anchored default enum set (generate()'s path, unchanged).
        # An explicit dict (incl. an EMPTY {} for the dx-only fuser) is used as-is,
        # so callers can build a fuser that touches ONLY differential_diagnosis.
        self.field_enum_strings = (
            _DEFAULT_FIELD_ENUM_STRINGS if field_enum_strings is None else field_enum_strings
        )
        self.recent_token_window = recent_token_window
        self.eps = eps
        self._np = np

        # Precompute: per field, map from first-token-id to enum-class string.
        # Two encodings: with and without a leading double-quote, since the LM
        # may have emitted the opening quote separately.
        self.first_tok_to_class_per_field: dict[str, dict[int, str]] = {}
        for fname, classes in self.field_enum_strings.items():
            d: dict[int, str] = {}
            for c in classes:
                for prefix in ('"', ""):
                    try:
                        toks = tokenizer.encode(prefix + c, add_special_tokens=False)
                    except Exception:
                        continue
                    if toks:
                        d.setdefault(int(toks[0]), c)
            self.first_tok_to_class_per_field[fname] = d

        # Precompute: per field-name, the token sequence for `"<field>":` and
        # the variant for nested fields where the key may end with `": "` or `":`
        self.field_key_token_seqs: dict[str, list[list[int]]] = {}
        for fname in self.field_enum_strings:
            seqs: list[list[int]] = []
            for variant in (f'"{fname}":', f'"{fname}": '):
                try:
                    s = tokenizer.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                if s:
                    seqs.append(s)
            self.field_key_token_seqs[fname] = seqs

        # differential_diagnosis (list[str]) array-slot support — L2, eval-time.
        # OFF by default => the maps below stay EXACTLY as the enum-only build
        # above, _array_fields is empty, and the array phase machine in
        # observe_token/compute_bias is never reached => byte-identical decode.
        self._array_fields: frozenset[str] = frozenset()
        if enable_dx_fusion:
            dx_strings = dx_cohort_strings or _DEFAULT_DX_COHORT_STRINGS
            self._array_fields = frozenset({_DX_FIELD})
            # first-token map: BARE canonical dx string -> the string itself. The
            # first token is the string's first CONTENT token (e.g. "Meningioma"
            # -> `M`), reached AFTER the `[ "` opener is walked by the phase
            # machine. We deliberately do NOT bake the structural `["`/`[ "`
            # prefixes in here: those merge to a class-AGNOSTIC token (Mistral
            # `["`=2221 is identical for every cohort) so biasing them cannot
            # disambiguate, and mapping the lone opening quote could nudge an
            # empty-string close. The array open is handled by state, not vocab.
            dxmap: dict[int, str] = {}
            for canon in dx_strings.values():
                try:
                    toks = tokenizer.encode(canon, add_special_tokens=False)
                except Exception:
                    toks = []
                if toks:
                    dxmap.setdefault(int(toks[0]), canon)
            self.first_tok_to_class_per_field[_DX_FIELD] = dxmap
            # Base key seqs only (`"differential_diagnosis":` / `...": `). The
            # `[`/`"` that follow are consumed by the array phase machine, so the
            # key match sets current_field the moment the key colon lands.
            dx_seqs: list[list[int]] = []
            for variant in (f'"{_DX_FIELD}":', f'"{_DX_FIELD}": '):
                try:
                    s = tokenizer.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                if s:
                    dx_seqs.append(s)
            self.field_key_token_seqs[_DX_FIELD] = dx_seqs

        # Per-batch parsing state
        self._state: dict[int, dict] = {}
        # Per-batch field-head softmax (registered per case)
        self._head_softmax: dict[int, dict[str, dict[str, float]]] = {}

    # -----------------------------------------------------------------------
    # Per-case lifecycle
    # -----------------------------------------------------------------------

    def register_case(
        self,
        batch_idx: int,
        field_softmax: dict[str, dict[str, float]],
    ) -> None:
        """Bind the head's per-field softmax dict to this batch slot for the
        current generation. Call before generate(). The dict is a mapping
        {field_name: {class_label: probability}} — usually obtained by running
        FieldClassificationHead and taking softmax over each field's logits."""
        self._head_softmax[batch_idx] = field_softmax
        self._state[batch_idx] = {"current_field": None, "recent_toks": [], "value_pos": 0}

    def reset_case(self, batch_idx: int | None = None) -> None:
        if batch_idx is None:
            self._head_softmax.clear()
            self._state.clear()
        else:
            self._head_softmax.pop(batch_idx, None)
            self._state.pop(batch_idx, None)

    # -----------------------------------------------------------------------
    # LP-facing API (Protocol)
    # -----------------------------------------------------------------------

    def observe_token(self, batch_idx: int, tok: int) -> None:
        state = self._state.setdefault(
            batch_idx, {"current_field": None, "recent_toks": [], "value_pos": 0}
        )
        recent: list[int] = state["recent_toks"]
        recent.append(int(tok))
        if len(recent) > self.recent_token_window:
            del recent[0 : len(recent) - self.recent_token_window]

        # If we're currently inside a value span, advance the per-field value-start
        # tracker so compute_bias knows whether the NEXT token is the first value
        # token. Two regimes:
        #   * scalar enum fields -> value_pos counter (bias at value_pos == 0).
        #     UNCHANGED / byte-identical to the pre-dx path.
        #   * differential_diagnosis (list[str], only when enable_dx_fusion) ->
        #     array phase machine that walks the `[ "` opener and fires on the
        #     first content token of the top-1 element.
        if state["current_field"] is not None:
            if state["current_field"] in self._array_fields:
                self._advance_array_phase(state, tok)
            else:
                state["value_pos"] += 1
                # Detect end-of-value
                try:
                    dec = self.tokenizer.decode([tok])
                except Exception:
                    dec = ""
                if any(ev in dec for ev in (",", "}", "]")):
                    state["current_field"] = None
                    state["value_pos"] = 0

        # If we're NOT in a value span, check whether the recent suffix matches
        # a known field-key sequence.
        if state["current_field"] is None:
            for fname, seqs in self.field_key_token_seqs.items():
                for seq in seqs:
                    if len(recent) >= len(seq) and recent[-len(seq):] == seq:
                        state["current_field"] = fname
                        state["value_pos"] = 0
                        if fname in self._array_fields:
                            # (Re)start the array opener phase for this element span.
                            state["arr_open"] = False
                            state["str_open"] = False
                            state["content_started"] = False
                        break
                if state["current_field"] is not None:
                    break

    @staticmethod
    def _has_content_char(dec: str) -> bool:
        """True if the decoded token carries a value character (letter/digit) —
        i.e. it is string CONTENT, not a structural `[ " ,` array opener."""
        return any(ch.isalnum() for ch in dec)

    def _advance_array_phase(self, state: dict, tok: int) -> None:
        """Walk the `[ " <content>` opener of a list[str] value (differential_
        diagnosis). Marks arr_open once `[` is seen, str_open once the opening `"`
        is seen (often the same merged ` ["`/`["` token), content_started once the
        first content token of the top-1 element is seen. Resets current_field at
        the element/array terminator (`,`/`]`/`}`) so ONLY differential_diagnosis[0]
        (the top-1 ranked dx) is biased — later ranks and prose stay free.

        Assumes the tokenizer never merges string CONTENT into the `["`/`[ "`
        opener token (verified for Mistral-7B: `["`=2221, ` ["`=7367, `[`=28792,
        `"`=28739, ` "`=345 all carry no letters). If a future tokenizer merged
        content into the opener, the bias would fire one token late (never a crash;
        the JSON stays valid and the post-hoc override remains the safety net)."""
        try:
            dec = self.tokenizer.decode([tok])
        except Exception:
            dec = ""
        if not state.get("arr_open"):
            if "[" in dec:
                state["arr_open"] = True
                if '"' in dec:            # merged `["` / ` ["` opens the string too
                    state["str_open"] = True
        elif not state.get("str_open"):
            if '"' in dec:
                state["str_open"] = True
        elif not state.get("content_started"):
            if self._has_content_char(dec):
                state["content_started"] = True
        # Element / array terminator ends the top-1 span.
        if any(ev in dec for ev in (",", "}", "]")):
            state["current_field"] = None

    def compute_bias(
        self,
        batch_idx: int,
        allowed_mask: torch.Tensor,
        scores_device: torch.device,
        scores_dtype: torch.dtype,
        vocab_size: int,
    ) -> torch.Tensor | None:
        state = self._state.get(batch_idx)
        if state is None:
            return None
        fname = state.get("current_field")
        if fname is None:
            return None
        # Fire only on the FIRST value token — later tokens of a multi-token value
        # are forced by the grammar's prefix tree.
        #   * scalar enum fields: value_pos == 0 (UNCHANGED).
        #   * differential_diagnosis array: the first CONTENT token, i.e. the `[ "`
        #     opener is complete (arr_open & str_open) but no content emitted yet.
        #     This lands the bias on the top-1 dx string token, not the bracket/quote.
        if fname in self._array_fields:
            if not (state.get("arr_open") and state.get("str_open")
                    and not state.get("content_started")):
                return None
        elif state["value_pos"] != 0:
            return None
        head_dist = self._head_softmax.get(batch_idx, {}).get(fname)
        if not head_dist:
            return None
        lambda_f = float(self.lambda_per_field.get(fname, 1.0))
        if lambda_f == 0.0:
            return None
        first_tok_map = self.first_tok_to_class_per_field.get(fname)
        if not first_tok_map:
            return None

        np = self._np
        bias = torch.zeros(vocab_size, device=scores_device, dtype=scores_dtype)
        for tok_id, c in first_tok_map.items():
            if tok_id >= vocab_size:
                continue
            p = head_dist.get(c, self.eps)
            bias_val = lambda_f * float(np.log(max(p, self.eps)))
            bias[tok_id] = bias_val
        return bias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
