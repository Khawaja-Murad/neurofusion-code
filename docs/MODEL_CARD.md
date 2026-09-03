# Model card — NeuroFusion

Following the Model Cards framework (Mitchell et al., 2019). **No weights are distributed with this
repository**; this card describes the system evaluated in the MLCN 2026 paper.

## Model details

| | |
|---|---|
| **Developed by** | Khawaja Murad ul Hassan et al. — NUST Islamabad; Ontario Tech University |
| **Type** | Multimodal 3D vision → structured-record + narrative report generator |
| **Segmentation backbone** | MedNeXt-L, multi-cohort pretrained, **frozen** (val mean fg Dice 0.89) |
| **Visual compression** | Per-lesion Q-Former, 32 tokens/lesion, ≤4 lesions → 128 visual tokens |
| **Language model** | LLaVA-Med v1.5 medical Mistral-7B, QLoRA-tuned |
| **Decoding** | XGrammar schema-constrained JSON; single draft-then-review pass (K=1) |
| **Trained components** | Lesion routing, Q-Former, field classifiers, LoRA adapter |
| **Frozen components** | MedNeXt backbone, base LM weights |
| **Licence** | Code Apache-2.0; weights not released |
| **Paper** | [arXiv:2609.02411](https://arxiv.org/abs/2609.02411) |

The LM is the **same** medical Mistral-7B as the LLaVA-Med baseline, so reported gains come from the
architecture rather than a stronger backbone.

## Intended use

**Intended:** assistive first-pass drafting of structured findings and a narrative for brain-tumor MRI,
for a radiologist to review and correct; and research on report faithfulness and head conditioning.

**Out of scope:** autonomous diagnosis; triage or prioritization; any clinical use without radiologist
review; non-brain, non-tumor, or non-MRI imaging; sequences other than T1/T1ce/T2/FLAIR; use as a
medical device. The system is not cleared or approved by any regulator.

## Inputs and outputs

**Input:** a four-channel skull-stripped 3D volume (T1, T1ce, T2, FLAIR) in BraTS format.
**Output:** a schema-valid JSON record — categorical fields (composition, enhancement pattern, mass
effect, edema, involvement, axis shift), a differential diagnosis, and free-text findings and impression.

## Performance

Headline numbers, with their bases — see the README for full tables and the paper for confidence
intervals.

| Metric | Value | Basis |
|---|---|---|
| Schema validity | 92.3% (36/39), Wilson [0.80, 0.97] | in-distribution test, *n*=39 |
| Same-base prose-content wins | 8 of 9 cells, 0 losses, 1 ns | GLI *n*=54, MEN/MET *n*=60, Holm-corrected |
| Diagnosis recall (GLI/MEN/MET) | 0.98 / 0.92 / 0.75 (CoT: 0.98 / 0.44 / 0.07) | descriptive lexical proxy |
| Latency | ≈73–89 s/case (CoT: 457 s) | A100 |
| Sentence contradiction rate | 7.5% (direct generation: 36.8%) | *n*=39, entailment-judged |
| Calibration (15-bin ECE) | 0.128 → 0.095 after temperature scaling | 50 calibration cases, test split |
| Reader study | rated highest in every cohort; 0/9 critical errors | 9 cases, 2 neurologists, pilot |

## Factors and evaluation

Evaluation is stratified by **cohort** (GLI / MEN / MET), which is the axis along which the failure mode
appears. Metrics: schema validity; per-field union-class macro-F1 with an over-prediction penalty on
predicted segmentation; RadGraph-F1, RaTEScore, GREEN scored prose-vs-prose; and a blinded 1–5 clinical
rubric. Analysis was **pre-specified**: primary endpoint RaTEScore-MEN, a nine-cell Holm-corrected
superiority family, TOST equivalence at ±0.03, two-sided paired BCa bootstrap (20,000 resamples).
Directional non-significant cells are never reported as wins.

Judge-free metrics (RaTEScore, RadGraph-F1) carry the claim; GREEN and the clinical rubric are LLM-scored
and only corroborate.

## Known limitations and failure modes

- **Segmentation-bounded.** Geometric fields are only as good as the mask (met-FT Dice 0.67). An empty
  mask produces no report and routes to review.
- **Small in-distribution test set** (*n*=39); same-base claims rest on 174 pooled held-out cases.
- **Only metastasis is OOD for the reporter**; the segmenter has seen metastasis.
- **A mask rule beats NeuroFusion on geometry** in all three cohorts and tops the glioma rubric, because
  a fixed glioma differential is right by construction there — and that same constant costs it both
  other cohorts.
- **Do not pin the diagnosis.** A calibrated diagnosis head that *overrides* the decoder produced 0 of 9
  gains and collapsed OOD metastasis recall 0.75 → 0.03. This is a load-bearing negative result.
- **No abstention mechanism** exists today.
- **Cohort ≡ data source** on BraTS-style archives: scanner, preprocessing, annotation protocol, case mix
  and biology are collinear, so source-robust diagnosis is falsifiable but not confirmable on these data.
- The reader study is a **pilot** with neurologists, not the radiologist panel a decisive study needs.

## Ethical considerations

See [ETHICS.md](ETHICS.md). All training and evaluation data are public; no new patient data were
collected; the reader study used de-identified report text only. The system is assistive and requires
radiologist review. Automation bias in expert readers is an explicit concern for any future study.
