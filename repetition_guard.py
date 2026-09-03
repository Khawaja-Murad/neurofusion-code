"""Repetition control for constrained decoding — the ACTUAL fix for schema-invalid reports.

WHAT WENT WRONG BEFORE THIS FILE EXISTED
----------------------------------------
The 37 % schema-invalid GLI reports were blamed on "a grammar dead state at object close",
inferred from `matcher_terminated=False` on 348/348 generations. That counter is a TAUTOLOGY:
xgrammar's `is_terminated()` only flips True after `accept_token(EOS)`, so it reads False for
every generation by construction. The fix built on it (`forbid_eos_until_terminated`) is a
measured NO-OP -- probing the real grammar shows EOS is never legal mid-object and is the SOLE
allowed token at completion, so masking stop tokens only ever removes what was already
forbidden. See [[feedback_xgrammar_is_terminated_tautology]].

THE REAL CAUSE, measured
------------------------
18/57 (32 %) of generations hit the 2048-token cap with the same token repeating
(`last_tok=13`), inside a free-text string where the grammar permits 31 681 tokens:

    25 % through a valid report : allowed=    3
    50 %                        : allowed=    6
    75 %  (inside `findings`)   : allowed=31681   <- nothing constrains content here
    100 %                       : allowed=    1   (EOS)

XGrammar enforces STRUCTURE, never CONTENT ([[feedback_three_tier_validity]]). Inside a JSON
string the model is unconstrained, it loops, the cap hits, and `findings`/`impression` are
never emitted -> the report is structurally incomplete. 32 % ~= the measured 37 % invalidity.

WHAT THIS DOES
--------------
Two detectors, both acting ONLY on tokens the grammar already allows:
  A. CONSECUTIVE run  -- the same token id emitted `max_run` times in a row.
  B. PERIODIC loop    -- the last W tokens are identical to the W before them, for any W in
                         `periods`. Catches phrase loops ("of the of the of the") that a
                         consecutive-run check cannot see.
In both cases the single token that would CONTINUE the loop is masked, forcing the model to
pick any other grammar-legal token -- typically the closing quote, which lets the object close.

THREE INVARIANTS, each earned by a specific past failure
--------------------------------------------------------
1. NEVER EMPTY THE DISTRIBUTION. Masking the last finite logit yields all -inf -> softmax ->
   nan -> `torch.multinomial` raises. That crashed job 66456570 mid-run. Every mask here is
   applied through `torch.where` guarded on "at least one finite token survives".
2. NO SYNC IN THE HOT PATH. This runs on every decoded token. An earlier guard called
   `.item()` twice per row per step (~1500 syncs/case) and slowed generation from ~50 s to
   ~197 s per case. Everything below stays on-device; nothing is read back to Python.
3. NEVER PUNISH LEGITIMATE JSON REPETITION. `", "`, `": "` and `"` recur constantly in valid
   output. Both detectors are therefore LOCAL (consecutive / immediately-periodic) and never
   count global frequency, which is what `no_repeat_ngram_size` and `repetition_penalty` do --
   both would corrupt valid JSON and are deliberately NOT used.

The guard is a no-op unless it fires, so a generation that never loops is byte-identical.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("neurofusion.repetition_guard")

# Defaults chosen against the observed failure (a single token repeating to the 2048 cap) with
# headroom for legal JSON. Valid reports do contain short runs -- indentation, repeated spaces
# inside prose -- so `max_run` must sit above those, not at 2.
DEFAULT_MAX_RUN = 8
DEFAULT_PERIODS = (2, 3, 4, 6, 8)
DEFAULT_MIN_REPEATS = 3          # a period-W block must recur this many times before firing

# ---------------------------------------------------------------------------------------
# ☠️☠️ ADDED 2026-08-14. THE DETECTORS ABOVE COULD NOT SEE THE FAILURE THEY WERE BUILT FOR.
#
# Job 920004 run 1: 23,942 chars, both JSON stages at their cap. The raw text shows the model
# reaching `"findings": "` and then ECHOING ITS OWN PROMPT inside the free-text string:
#
#   ... [END CLASSIFIER HEURISTICS] You are a neuroradiologist reading a brain MRI for a
#   tumor report. Heuristic findings from segmentation: lesion_2.composition=necrotic ...
#
# repeating. That IS the failure mode this module documents -- but `DEFAULT_PERIODS` tops out
# at 8 tokens and the echoed block is a prompt-length span, on the order of a hundred tokens.
# Detector B needs the loop to be periodic within an 8-token window; this one is periodic on a
# ~100-token window. The guard was present, imported, and enabled, and it was structurally
# incapable of firing. ⇒ ★★ A GUARD'S THRESHOLD IS PART OF ITS COVERAGE CLAIM: "the loop guard
# is running" says nothing until you state the loop LENGTHS it can see.
#
# ☠️ AND THE OBVIOUS FIX IS A TRAP. Scanning the whole of `input_ids` for a repeated long span
# would fire on LEGITIMATE prompt-grounded copying -- valid reports quote the heuristic block
# verbatim ("Heuristic findings from segmentation: lesion_2.composition=necrotic" appears in
# the PASSING generations too). So C and D below are restricted to the GENERATED span. They
# detect the model repeating ITSELF, never the model quoting its prompt.
DEFAULT_LONG_PERIODS = (16, 32, 64, 128)
DEFAULT_LONG_MIN_REPEATS = 2     # a 16+ token block recurring even TWICE is already pathological
# ☠️ 64 WAS WRONG AND REAL DATA SAID SO. Swept against the 9 tokenized generations from job
# 920004 (3 BraTS TRAINING cases -- fit basis, no looks): at w=64 the guard caught the failure
# AND fired on 2 of 8 HEALTHY reports. The false positive is not prompt quoting -- detectors C
# and D already exclude the prompt -- it is the model legitimately repeating ITSELF: a report
# with several lesions emits near-identical 64-token sub-objects, and where the differing
# values are short the spans match exactly.
#     w:   64   96  128  192  256
#     TP:   1    1    1    1    0
#     FP:   2    1    0    0    0
# 128 is the smallest window that keeps the true positive with zero false positives.
# ☠️ DENOMINATORS, BECAUSE THIS IS A THRESHOLD CHOICE AND NOT A MEASURED RATE: TP is 1 of 1
# and FP is 0 of 8. A 0/8 false-positive count has a 95% upper bound near 31%. This setting
# SEPARATES the 9 runs it was chosen on; it does not establish a rate on anything.
# ☠️ AND THE min_gen COLUMN IS UNEVALUABLE HERE: the probe stores only the first 2000 chars
# (693 tokens), so every min_gen>=1024 row read TP=0 because the DATA ends, not because the
# guard fails. Do not read those rows as evidence against a late-arming threshold.
DEFAULT_ECHO_WINDOW = 128


class RepetitionGuard:
    """Masks the token that would continue a degenerate loop. Grammar-safe, sync-free.

    Intended to run AFTER the XGrammar processor in the LogitsProcessorList, so `scores`
    already has every grammar-illegal token at -inf and this can only ever remove a token
    from the legal set -- never add one back.
    """

    def __init__(self, max_run: int = DEFAULT_MAX_RUN,
                 periods: tuple[int, ...] = DEFAULT_PERIODS,
                 min_repeats: int = DEFAULT_MIN_REPEATS,
                 enabled: bool = True,
                 long_periods: tuple[int, ...] = DEFAULT_LONG_PERIODS,
                 long_min_repeats: int = DEFAULT_LONG_MIN_REPEATS,
                 echo_window: int = DEFAULT_ECHO_WINDOW) -> None:
        self.max_run = int(max_run)
        self.periods = tuple(int(p) for p in periods)
        self.min_repeats = int(min_repeats)
        self.enabled = bool(enabled)
        self.long_periods = tuple(int(p) for p in long_periods)
        self.long_min_repeats = int(long_min_repeats)
        self.echo_window = int(echo_window)
        self.n_fired = 0            # incremented only by the sampled diagnostic, see below
        self._call_count = 0
        self._neg_inf: torch.Tensor | None = None    # cached per (device, dtype)
        self._min_val: float = 0.0
        # Where the PROMPT ends. Detectors C and D look only past this point, so quoting the
        # prompt -- which valid reports do -- can never trip them.
        self._prompt_len: int | None = None
        self._last_T: int | None = None

    # -- generation boundary -----------------------------------------------------------
    def reset(self, prompt_len: int | None = None) -> None:
        """Start a new generation. `prompt_len` defaults to auto-detection on the next call."""
        self._prompt_len = int(prompt_len) if prompt_len is not None else None
        self._last_T = None

    def _note_step(self, T: int) -> None:
        """Track the prompt boundary without needing the caller to announce it.

        ☠️ AUTO-DETECTION, NOT A GUESS DRESSED AS ONE. HF calls a processor with the full
        sequence, so the FIRST call of a generation carries exactly the prompt. A sequence
        length that fails to grow means a new generation has begun -- the only way T can stop
        increasing is a fresh call. `reset()` remains available for callers that know better,
        and is what `XGrammarLogitsProcessor.reset` should drive if these are ever paired.
        """
        # ☠️ THIS USED TO SILENTLY DISCARD AN EXPLICIT reset(prompt_len=...). The condition was
        # `if self._last_T is None or T < self._last_T`, and reset() sets _last_T = None, so
        # the FIRST call after an explicit reset always overwrote the caller's value with T.
        # Measured: reset(prompt_len=160) then a first call at T=300 left _prompt_len = 300.
        # ⇒ ★★ A PARAMETER THAT NEVER TAKES EFFECT IS WORSE THAN AN ABSENT ONE — it advertises
        #   a control the caller does not have, and the failure is invisible because the
        #   auto-detected value is usually right anyway. Same family as a path-existence check
        #   standing in for a safety property.
        if self._last_T is None:
            # First sight of a generation: honour an explicit reset(prompt_len), else detect.
            if self._prompt_len is None:
                self._prompt_len = T
        elif T < self._last_T:
            # A NEW generation began without a reset(); any explicit value belonged to the
            # previous one, so fall back to auto-detection rather than carrying it over.
            self._prompt_len = T
        self._last_T = T

    # -- detectors C and D: long-range, generated-span only -----------------------------
    @staticmethod
    def long_loop_mask(gen_ids: torch.Tensor, long_periods: tuple[int, ...],
                       long_min_repeats: int, echo_window: int) -> torch.Tensor:
        """[B] bool: True where the GENERATED tail is repeating something it already generated.

        C. LONG-PERIOD immediate repeat -- the last W generated tokens equal the W before them,
           for W in `long_periods`, `long_min_repeats` blocks over. Catches a paragraph-scale
           loop that `periods=(2..8)` cannot see.
        D. ECHO -- the last `echo_window` generated tokens occur EARLIER in the generated span,
           at a NON-OVERLAPPING position. Catches the first restatement, before it has had to
           repeat periodically, and catches loops whose period is not in `long_periods`.

        Sync-free: every result stays a device tensor.
        """
        B, G = gen_ids.shape
        fire = torch.zeros(B, dtype=torch.bool, device=gen_ids.device)
        if G == 0:
            return fire

        # C
        for w in long_periods:
            need = w * long_min_repeats
            if G < need or w < 2:
                continue
            blocks = gen_ids[:, -need:].reshape(B, long_min_repeats, w)
            periodic = (blocks == blocks[:, :1, :]).all(dim=2).all(dim=1)
            # Same in-block variety guard as detector B: a run of identical tokens satisfies
            # every periodicity test, and single-token runs belong to detector A.
            varied = (blocks[:, 0, :] != blocks[:, 0, :1]).any(dim=1)
            fire |= (periodic & varied)

        # D
        w = echo_window
        if w >= 2 and G >= 2 * w:
            tail = gen_ids[:, -w:]                                    # [B, w]
            # Windows ending at or before index G-w-1 -> strictly non-overlapping with `tail`.
            earlier = gen_ids[:, : G - w].unfold(1, w, 1)             # [B, G-2w+1, w]
            same = (earlier == tail.unsqueeze(1)).all(dim=2).any(dim=1)
            varied = (tail != tail[:, :1]).any(dim=1)                 # not a single-token run
            fire |= (same & varied)
        return fire

    # -- pure core, testable without a model -------------------------------------------
    @staticmethod
    def loop_token_mask(input_ids: torch.Tensor, max_run: int,
                        periods: tuple[int, ...], min_repeats: int) -> torch.Tensor:
        """[B] bool: True where the LAST token is continuing a degenerate loop.

        Sync-free: every operation is a tensor op and nothing is read back to Python.
        """
        B, T = input_ids.shape
        if T == 0:
            return torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        last = input_ids[:, -1]
        fire = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        # A. consecutive run: the final `max_run` tokens are all equal to `last`
        if max_run >= 1 and T >= max_run:
            tail = input_ids[:, -max_run:]
            fire |= (tail == last.unsqueeze(1)).all(dim=1)

        # B. periodic loop: the last W tokens repeat the preceding W, `min_repeats` times over.
        #
        # ☠️ The block must contain at least TWO DISTINCT tokens. Without that guard a run of
        # identical tokens also satisfies every periodicity test -- [13]*6 is a period-2 block
        # repeated 3x -- so the periodic detector silently subsumed the consecutive detector and
        # the EFFECTIVE run threshold became w*min_repeats = 6, not the documented max_run = 8.
        # Two detectors with independent, honest thresholds; single-token loops belong to A.
        for w in periods:
            need = w * min_repeats
            if T < need or w < 2:
                continue
            blocks = input_ids[:, -need:].reshape(B, min_repeats, w)
            periodic = (blocks == blocks[:, :1, :]).all(dim=2).all(dim=1)
            varied = (blocks[:, 0, :] != blocks[:, 0, :1]).any(dim=1)   # >=2 distinct in-block
            fire |= (periodic & varied)
        return fire

    # -- LogitsProcessor interface -----------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self.enabled or input_ids.numel() == 0:
            return scores
        self._call_count += 1
        self._note_step(int(input_ids.shape[1]))

        fire = self.loop_token_mask(input_ids, self.max_run, self.periods, self.min_repeats)
        # C and D see only what THIS generation produced, so quoting the prompt is never a loop.
        p = self._prompt_len
        if p is not None and input_ids.shape[1] > p:
            fire |= self.long_loop_mask(input_ids[:, p:], self.long_periods,
                                        self.long_min_repeats, self.echo_window)
        last = input_ids[:, -1].unsqueeze(1)                       # [B, 1]
        # Mask with -inf, matching xgrammar_decoder's own convention (it builds
        # `torch.full_like(scores[b], float("-inf"))`). Masking with `finfo.min` instead would
        # be FINITE, so any downstream processor -- or this guard on the next step -- would
        # read the token as still available and the two layers would disagree about what is
        # legal. Cached per (device, dtype) rather than rebuilt every step: this runs once per
        # decoded token and a fresh host-side scalar each time is pure waste.
        if (self._neg_inf is None or self._neg_inf.device != scores.device
                or self._neg_inf.dtype != scores.dtype):
            self._neg_inf = torch.tensor(-torch.inf, device=scores.device, dtype=scores.dtype)
            self._min_val = torch.finfo(scores.dtype).min
        neg_inf = self._neg_inf

        # Candidate: the looping token removed.
        cur = scores.gather(1, last)                               # [B, 1]
        proposed = scores.scatter(1, last, torch.full_like(cur, neg_inf))

        # INVARIANT 1: only adopt the mask where a legal token still survives it.
        # "available" is `> finfo.min`, NOT `isfinite`: that is False for -inf AND for
        # finfo.min, so the check holds whichever sentinel an upstream processor used. A bare
        # isfinite() would call a finfo.min-masked token available and let this guard remove
        # the last real option -> all-masked distribution -> multinomial nan (crashed 66456570).
        # Both tests are [B]-shaped tensors, so the decision stays on-device (INVARIANT 2).
        survives = (proposed > self._min_val).any(dim=1, keepdim=True)                 # [B, 1]
        adopt = (fire.unsqueeze(1) & survives)                         # [B, 1]
        out = torch.where(adopt, proposed, scores)

        # Diagnostic OFF the hot path: reading `fire` back costs a sync, so sample it. A
        # sampled counter is NOT a count -- label it as such wherever it is reported, which is
        # the mistake that made "escape hatch fired 0 times" look meaningful.
        if self._call_count % 256 == 0 and bool(fire.any().item()):
            self.n_fired += 1
            if self.n_fired <= 5:
                log.warning(
                    f"RepetitionGuard: degenerate loop detected at T={input_ids.shape[1]} -> "
                    f"masking the continuing token (SAMPLED occurrence #{self.n_fired}; "
                    f"sampled 1-in-256 calls, so this is not a total)")
        return out


# ---------------------------------------------------------------------------
# SELF-TEST (added 2026-08-08 at RECOVERY, not part of the original module).
# ☠️ This file was missing from the repo entirely -- present only in a stale
# worktree -- so `from repetition_guard import RepetitionGuard` raised ImportError on
# EVERY generation this project has ever run, including all of B'9's. A guard nobody
# has watched go RED is not evidence, and this one has never run at all. Prove both
# detectors fire, prove INVARIANT 1 holds, and prove a non-looping sequence is
# untouched, before trusting it in a decode path.
# ---------------------------------------------------------------------------
def self_test() -> None:
    import torch as _t

    g = RepetitionGuard()
    V = 32

    # A. CONSECUTIVE run: token 7 repeated max_run times -> the continuing token is masked.
    ids = _t.full((1, DEFAULT_MAX_RUN), 7, dtype=_t.long)
    scores = _t.zeros(1, V)
    out = g(ids, scores)
    assert _t.isneginf(out[0, 7]), "consecutive-run detector did NOT fire"
    assert (out[0, :7] == 0).all() and (out[0, 8:] == 0).all(), "masked more than the loop token"

    # B. PERIODIC loop: block [1,2] repeated min_repeats times -> continuing token masked.
    per = [1, 2] * DEFAULT_MIN_REPEATS
    ids = _t.tensor([per], dtype=_t.long)
    out = g(ids, _t.zeros(1, V))
    assert _t.isneginf(out[0, per[-1]]), "periodic-loop detector did NOT fire"

    # C. NO-OP on a non-looping sequence (byte-identical passthrough).
    ids = _t.tensor([[1, 2, 3, 4, 5, 6, 9, 11]], dtype=_t.long)
    s = _t.randn(1, V)
    assert _t.equal(g(ids, s), s), "guard mutated a non-looping generation"

    # D. INVARIANT 1: when the looping token is the ONLY finite logit, do NOT mask it
    # (masking it empties the distribution -> multinomial nan; that crashed job 66456570).
    ids = _t.full((1, DEFAULT_MAX_RUN), 7, dtype=_t.long)
    s = _t.full((1, V), float("-inf"))
    s[0, 7] = 1.0
    out = g(ids, s)
    assert _t.isfinite(out[0, 7]), "INVARIANT 1 VIOLATED: distribution emptied"

    # E. disabled -> passthrough
    g_off = RepetitionGuard(enabled=False)
    s = _t.randn(1, V)
    assert _t.equal(g_off(_t.full((1, 16), 7, dtype=_t.long), s), s), "disabled guard mutated scores"

    # ------------------------------------------------------------------------------
    # F-J. DETECTORS C AND D (added 2026-08-14). Each is driven RED on the shape it
    # exists to catch and GREEN on the shape it must not touch. The false-positive
    # case (I) is the one that matters most: firing on legitimate prompt-grounded
    # copying would degrade every valid report to fix a 1-in-30 failure.
    # ------------------------------------------------------------------------------
    PROMPT = list(range(100, 260))          # 160 "prompt" tokens
    P = len(PROMPT)

    def _run(gen_tokens, guard=None):
        g2 = guard or RepetitionGuard()
        ids = _t.tensor([PROMPT + list(gen_tokens)], dtype=_t.long)
        g2.reset(P)
        g2._last_T = ids.shape[1]           # pin the boundary; no auto-detect in the drill
        g2._prompt_len = P
        return g2(ids, _t.zeros(1, 1024)), ids

    # F. LONG-PERIOD loop (detector C): a 32-token block repeated twice -> fires.
    blk = list(range(300, 332))
    out, ids = _run(blk + blk)
    assert _t.isneginf(out[0, blk[-1]]), "detector C did NOT fire on a 32-token repeated block"

    # G. ECHO (detector D): the last 64 generated tokens reappear from earlier, with
    # unrelated tokens in between -> fires even though nothing is periodic.
    # ☠️ SIZED FROM THE CONSTANT, NOT HARDCODED. This fixture was written as a literal 64 and
    # silently stopped testing detector D the moment DEFAULT_ECHO_WINDOW moved to 128 -- it
    # failed loudly here, but a fixture pinned to an old threshold is how a drill quietly
    # stops covering the thing it names.
    span = list(range(400, 400 + DEFAULT_ECHO_WINDOW))
    filler = list(range(700, 740))
    out, _ = _run(span + filler + span)
    assert _t.isneginf(out[0, span[-1]]), "detector D did NOT fire on a non-adjacent echo"

    # H. THE ACTUAL OBSERVED SHAPE: a ~100-token prompt-style block restated inside the
    # generated text, which detectors A and B (periods<=8) cannot see.
    echoed = list(range(500, 500 + DEFAULT_ECHO_WINDOW))   # the observed shape, at window size
    out, _ = _run(echoed + echoed)
    assert _t.isneginf(out[0, echoed[-1]]), "the observed failure shape is STILL not caught"
    a_b_only = RepetitionGuard.loop_token_mask(
        _t.tensor([PROMPT + echoed + echoed], dtype=_t.long),
        DEFAULT_MAX_RUN, DEFAULT_PERIODS, DEFAULT_MIN_REPEATS)
    assert not bool(a_b_only.any()), (
        "detectors A/B were supposed to be BLIND to this shape -- if they see it, the "
        "premise for adding C and D is wrong and the diagnosis needs redoing")

    # I. ☠️ NO FALSE POSITIVE ON PROMPT QUOTING. The generation copies 120 tokens of the
    # PROMPT verbatim -- which valid reports do -- and must be left alone.
    out, _ = _run(PROMPT[:120] + [900, 901, 902])
    assert _t.equal(out, _t.zeros(1, 1024)), (
        "FIRED ON LEGITIMATE PROMPT COPYING -- this would degrade every valid report")

    # J. INVARIANT 1 still holds for the new detectors.
    g3 = RepetitionGuard()
    ids = _t.tensor([PROMPT + blk + blk], dtype=_t.long)
    g3.reset(P); g3._prompt_len = P; g3._last_T = ids.shape[1]
    s = _t.full((1, 1024), float("-inf")); s[0, blk[-1]] = 1.0
    assert _t.isfinite(g3(ids, s)[0, blk[-1]]), "INVARIANT 1 VIOLATED by detector C/D"

    # K. ☠️ THE BOUNDARY MUST SURVIVE REUSE. model.py appends ONE RepetitionGuard per
    # generate() call, but that call runs a K-SAMPLE LOOP inside it, so the same instance
    # sees several generations. If `_prompt_len` stuck at the first sample's value, sample 2
    # would have detectors C/D scanning part of its PROMPT -- reintroducing exactly the false
    # positive the generated-span restriction exists to prevent. Nothing in the code path
    # calls reset(), so the AUTO-DETECT is load-bearing and is tested here rather than assumed.
    g4 = RepetitionGuard()
    quoted = PROMPT[:130]                       # a legal long quote from the prompt
    # ☠️ THE CALLING CONVENTION IS PART OF THE TEST. HF calls a processor with the sequence
    # SO FAR and asks for the NEXT token, so the first call of a generation carries the bare
    # prompt and nothing else. Writing this loop as `quoted[:step + 1]` handed the guard a
    # prompt + 1 token on its first sight and it pinned the boundary one token late -- the
    # assertion below caught it. A fixture that calls the unit differently from production
    # tests a different unit.
    for sample in range(3):
        for step in range(len(quoted) + 1):
            ids = _t.tensor([PROMPT + quoted[:step]], dtype=_t.long)
            out = g4(ids, _t.zeros(1, 1024))
            assert g4._prompt_len == P, (
                f"prompt boundary drifted to {g4._prompt_len} on sample {sample} step {step}")
            assert _t.equal(out, _t.zeros(1, 1024)), (
                f"FIRED on legal prompt quoting, sample {sample} step {step} -- the boundary "
                f"did not survive K-sample reuse")
    # ...and a genuine loop in a LATER sample is still caught -- stepped through properly,
    # because handing the guard a whole finished sequence in one call is a NEW generation from
    # its point of view (T drops), so it correctly reads the lot as prompt and finds no
    # generated span. That is the auto-detect working, not failing; the earlier version of this
    # assertion tested the wrong thing and read as "the guard went deaf".
    gen = blk + blk
    for step in range(len(gen) + 1):
        out = g4(_t.tensor([PROMPT + gen[:step]], dtype=_t.long), _t.zeros(1, 1024))
    assert g4._prompt_len == P, f"boundary wrong on the later sample: {g4._prompt_len}"
    assert _t.isneginf(out[0, gen[-1]]), "guard went deaf after K-sample reuse"

    print("repetition_guard self-test: 11/11 PASS "
          "(A consecutive, B periodic, no-op clean, INVARIANT 1, disable, K-sample boundary, "
          "C long-period, D echo, the OBSERVED shape, A/B proven blind to it, "
          "no false positive on prompt quoting)")


if __name__ == "__main__":
    self_test()
