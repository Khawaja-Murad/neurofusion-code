# Reproducibility

What is needed to reproduce the MLCN 2026 numbers, and what this repository does and does not give you.

## What you need to supply

This repository contains **code only**. To reproduce the paper you also need:

1. **Imaging** — BraTS-2020, RadGenome-Brain (GLI, MEN), BraTS-MET, obtained from their sources under
   their own licences. See [DATA.md](DATA.md).
2. **The 121 structured reports** — release pending curation; not in this repository.
3. **Base LM weights** — LLaVA-Med v1.5 medical Mistral-7B, from its own distribution.
4. **A frozen MedNeXt segmentation checkpoint.**
5. **Metric packages** — RaTEScore, RadGraph (RadGraph-XL), and GREEN, installed per their own
   instructions. GREEN and the clinical rubric additionally need an LLM judge.

Without (2) the training and per-field evaluation cannot be rerun. The architecture, decoding, metric,
and statistical code is complete and inspectable regardless.

## Environment

```bash
export NF_MISTRAL_BASE=/path/to/llava-med-v1.5-mistral-7b
export NF_REPO=$PWD
export NF_PRED_DIR=$PWD/predictions
```

Python 3.11; NVIDIA A100 for training and full evaluation; inference fits an L4.

## Pipeline

| Stage | Entry point | Produces |
|---|---|---|
| 1. Report curation | `build_dataset.py`, `preprocess_reports.py` | Free text → validated structured records against `schema.py` |
| 2. Dataset assembly | `dataset.py`, `bratscombined_dataset.py`, `dataset_augment.py` | Multi-cohort BraTS-format loaders |
| 3. Training | `train.py` | Lesion router, Q-Former, field heads, LoRA adapter (backbone and base LM stay frozen) |
| 4. Inference | `predict.py` | Per-case prediction JSONL under `$NF_PRED_DIR` |
| 5. Calibration | `calibrate.py` | Temperature scaling fitted by NLL on the 50 calibration cases |
| 6. Evaluation | `eval.py`, `eval_folds.py` | Schema validity, per-field union-class macro-F1 with over-prediction penalty |
| 7. Narrative metrics | `scripts/eval_narrative_rate.py`, `scripts/eval_narrative_radgraph.py`, `scripts/eval_narrative.py` | RaTEScore, RadGraph-F1, GREEN, prose-vs-prose |
| 8. Significance | `scripts/bootstrap_significance.py` | Paired BCa bootstrap and Holm correction |
| 9. Clinical rubric | `scripts/aggregate_opus_judge.py` | Clin-O, the blinded 1–5 rubric |
| 10. Reader study | `scripts/analyze_reader_study.py` | Reader-study aggregate (paper Table 4) |
| 11. Faithfulness | `scripts/verbalizer_faithfulness.py`, `scripts/faithfulness_v3.py` | Per-sentence entailment / contradiction rate |
| 12. Latency | `scripts/nfdx_struct_latency.py` | Per-case wall-clock |
| — CoT baseline | `scripts/cot_prompts.py` | The multi-chain (K=2) CoT prompts used as the same-base control |

## Statistical protocol

Pre-specified before the comparisons were run:

- **Primary endpoint:** RaTEScore on meningioma.
- **Superiority family:** the nine same-base cells {RaTEScore, RadGraph-F1, GREEN} × {GLI, MEN, MET},
  Holm-corrected.
- **Equivalence:** TOST at ±0.03, below half the metric SD.
- **External comparison:** the 12-test MET family, {RaTEScore, GREEN, Clin-O} against each of four
  external baselines.
- **Intervals:** two-sided BCa bootstrap over paired per-case differences, 20,000 resamples. Reported
  intervals come from one resampling seed; five seeds confirm the sign of every bound.
- A directional non-significant cell is **never** reported as a win.

## Known reproduction caveats

- **The pipeline is not bit-reproducible.** TF32 and non-deterministic kernels move numbers in the
  fourth decimal. The reported effects are far larger than that drift, but exact digit-for-digit
  reproduction should not be expected.
- **GREEN and Clin-O are LLM-scored.** Judge model versions change over time; the paper notes one
  baseline re-scored 2.88 → 2.98 under a newer judge. The judge-free metrics carry the claim.
- **Two trainers on shared Lustre** will contend; stage dataset roots to node-local scratch.
- `scripts/` paths are environment-driven (`NF_REPO`, `NF_PRED_DIR`). The original runs used absolute
  cluster paths; those have been parameterized for this release, which is the only functional change
  made to the code for publication.
