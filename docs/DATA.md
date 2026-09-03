# Data provenance

Every volume used in the MLCN 2026 paper comes from a **public dataset under its original licence**.
No new patient imaging was collected, and no institutional ethics approval beyond the source datasets'
own was required for the work reported in the paper. See [ETHICS.md](ETHICS.md) for the governance
posture and for the separate, not-yet-run clinical evaluation.

## Cohorts

| Cohort | Source | *n* | Role |
|---|---|---|---|
| BraTS-2020 | [Menze et al., BraTS](https://www.med.upenn.edu/cbica/brats2020/) | 121 reported cases | Training (39 in-distribution test, 50 calibration held out) |
| GLI — glioma | RadGenome-Brain | 60 (54 twin-excluded) | In-distribution, patient-disjoint held-out extension |
| MEN — meningioma | RadGenome-Brain | 60 | In-distribution, patient-disjoint held-out extension |
| MET — metastasis | BraTS-MET | 60 | **Held out for the reporter** (see below) |

All inputs are four-sequence volumes: T1, T1ce, T2, FLAIR.

### What "held out for the reporter" means

For the MET cohort the Q-Former, the field heads, and the LoRA decoder **never see a metastasis report**.
The *segmenter* is metastasis-fine-tuned and the references come from AutoRG-Brain's corpus. So the
reporter is out-of-distribution but the segmentation is not. We state this rather than claim a clean
zero-shot result, and it is why AutoRG-Brain's lead on structured metrics for MET is reported as a
train-on-distribution artifact rather than a defeat.

### What "in-distribution" means for GLI/MEN

The RadGenome glioma and meningioma cohorts **extend training patient-disjointly**. They are held-out,
**not zero-shot**. The paper labels them that way throughout.

## Structured reports

121 BraTS-2020 cases carry **human-authored, radiologist-reviewed structured reports**, written by the
study team. The diagnosis is taken from histopathology where available and from the stated differential
otherwise. 98% passed schema validation.

- These reports are **not in this repository.** They will be released publicly once curation is complete.
- `build_dataset.py` contains the curation logic and the row-index → study case-ID map (`TR###`).
  Those identifiers are **study-internal pseudonyms**: no names, no medical record numbers, no dates.
  The source spreadsheet they index is not redistributed.
- The 121 reports **lack formal inter-annotator agreement**. This is a stated limitation of the paper.

## Deduplication and leakage control

- All cases are **same-subject deduplicated**.
- A preflight gate confirms **0 train/test overlap** by patient hash **and** content hash.
- Six GLI subject-twins are dropped for the twin-excluded same-base contrast (*n*=54). The external
  baseline comparison keeps the full *n*=60, and the tables say which is which.

Leakage control is treated as a provenance property, not a similarity threshold: a test set is only
clean if its *source* contributed no training data.

## What this repository does and does not redistribute

| | |
|---|---|
| **Included** | Model, training, inference, evaluation, and analysis code |
| **Not included** | Imaging volumes — obtain them from the sources above under their own licences |
| **Not included** | The 121 structured reports (release pending curation) |
| **Not included** | Model weights — see the README |
| **Not included** | Any per-case prediction or ground-truth JSONL |

Scripts expect prediction files under `$NF_PRED_DIR` and dataset artifacts under `$NF_REPO/out/`;
neither directory is populated by this repository.
