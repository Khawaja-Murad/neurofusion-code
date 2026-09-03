# Ethics, governance, and clinical scope

This document states precisely what has and has not been done, so that the paper's claims are not read
more broadly than they were made.

## The published work

**Data.** All imaging is public, under the original dataset licences (BraTS-2020, RadGenome, BraTS-MET —
see [DATA.md](DATA.md)). **No new patient data were collected** for this paper and **no ethics approval
beyond the source datasets' own was required.** The 121 structured reports were authored by the study
team from those public volumes.

**Reader study.** Two board-certified neurologists independently rated **de-identified reports** — text
only, no patient-identifying content — for nine held-out cases (3 per cohort, drawn at random *before*
scoring), under opaque system labels A–F in fixed order. The two readers agreed on every rated item
(all 255 axis scores, the critical-error flags, and the nine sign-offs). Because agreement was complete,
chance-corrected coefficients are degenerate and we report raw item-level agreement.

This is a **pilot**. We report descriptive means and critical-error counts, **not *p*-values**, and the
paper says so.

## Intended use and standing limits

NeuroFusion is an **assistive draft tool**: a first-pass structured record and narrative for a
radiologist to review and correct.

- It is **not** an autonomous diagnostic system, and not a triage system.
- It is **not** a cleared or approved medical device, and has not been evaluated prospectively.
- Geometric completeness is bounded by the segmentation; **empty masks route to human review**.
- Diagnosis recall as reported is a **descriptive lexical proxy** (does the top differential name the
  cohort type), deliberately not a primary endpoint.

The system has **no abstention mechanism at present**. Where planning documents describe abstention, they
describe design intent, not measured behaviour.

## Clinical evaluation — planned, not run

A clinical evaluation protocol exists in **design / pre-registration draft** form. As of this release:

- **Nothing in it has been run.** No patient has been enrolled and no clinical data have been accessed.
- **IRB submission is deferred**, to be raised when a deployable frozen checkpoint exists.
- The design is a **DECIDE-AI** early-stage evaluation of AI decision support: retrospective,
  de-identified, paired, multi-reader multi-case (MRMC), pre-registered, with model parameters
  **frozen and hash-recorded before the first read**.
- The intended reader panel is **radiologists** — stratified into generalist/trainee and subspecialist
  neuroradiologist tiers — because the task is image interpretation. Neurologists and neurosurgeons are
  consumers of the report and would be a separate sub-study. The nine-case reader study in the paper
  used neurologists and is labelled a pilot for exactly this reason.
- Any prospective work would require local ethics/IRB review, a data use agreement, and verified
  no-PHI-egress. An "IRB-light" pathway is still IRB review.

## Deployment posture

No model weights are distributed. Any checkpoint trained on a non-commercially-licensed corpus is
treated as a **research lineage** that cannot become a served model without a relicence, and lineage is
tracked explicitly rather than assumed.

## Reporting discipline

The project holds itself to a few rules that shape what appears in the paper:

- A failing gate is a result; tolerances are never retuned to make one pass.
- Negative results are kept and reported — the diagnosis-pin ablation is in the paper because it failed.
- Every reported number carries its basis, denominator, and marginal null.
- A claim without its domain is a different, false claim.
