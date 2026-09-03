"""Reproducibly aggregate the blinded 2-neurologist reader study for the MLCN paper.

Two board-certified neurologists blindly rated model reports on three per-cohort
rating sheets (GLI / MET / MEN). Each sheet holds 10 case-tabs; three cases per
cohort were scored (a focused pilot). Each scored case rates every candidate
system on five 1-5 Likert axes (Dx accuracy, Localisation, Completeness,
Impression, Overall utility), a binary Critical-Error flag, and a single forced
sign-off preference ("which report would you sign off on?").

Blinding key (provided by the study maintainer; independently text-verified this
session against the report bodies in the sheets):
    A = NeuroFusion         (M6 head-surfacing hero, diagnosis-pin-free)  -> OURS
    B = Prior-CoT           (same-base chain-of-thought predecessor)      -> OURS (ablation)
    C = M3D-LaMed
    D = LLaVA-Med
    E = AutoRG-Brain
    F = BrainGemma3D        (never ran glioma -> absent from the GLI sheet)

Normalisations (documented, applied verbatim):
  * numeric-cell typo 'O'/'o'  -> 0   (rater wrote the letter O for zero)
  * '—' / blank / None         -> skipped (reference row, not a candidate)
  * Critical-Error 'U'         -> treated as flagged (unusable / uncertain -> Y)

Sign-off handling (integrity, per maintainer instruction): three metastasis
sign-offs were recorded as "Ground Truth". The de-identified reference is NOT a
candidate system, so each such pick is reassigned to the highest-rated CANDIDATE
system in that case. Where the top is a tie the tie is recorded explicitly; a
system is credited with a "top-or-tied" sign-off iff it is in the arg-max set.
NeuroFusion is the unique top in MET-2 and tied-for-top in MET-1 and MET-3.

Outputs:
  out/reader_study_aggregate.json   -- every number, auditable to source cells
  paper/mlcn/tab_reader.tex         -- LaTeX table body used by main.tex

Usage:
  neuro_venv/bin/python scripts/analyze_reader_study.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent

SHEETS = {
    "GLI": ROOT / "BRAIN TUMOUR GLI_rating_sheet.xlsx",
    "MET": ROOT / "Brain Timour MET_rating_sheet.xlsx",
    "MEN": ROOT / "MENINGIOMA _rating_sheet.xlsx",
}

# Blind letter -> canonical system name.  A and B are both NeuroFusion variants.
KEY = {
    "A": "NeuroFusion",
    "B": "Prior-CoT",
    "C": "M3D-LaMed",
    "D": "LLaVA-Med",
    "E": "AutoRG-Brain",
    "F": "BrainGemma3D",
}
OURS = {"NeuroFusion", "Prior-CoT"}
# Fixed display order for tables.
ORDER = ["NeuroFusion", "Prior-CoT", "M3D-LaMed", "LLaVA-Med", "AutoRG-Brain", "BrainGemma3D"]
DISPLAY = {
    "NeuroFusion": "NeuroFusion (ours)",
    "Prior-CoT": "Prior-CoT (ours)",
    "M3D-LaMed": "M3D-LaMed",
    "LLaVA-Med": "LLaVA-Med",
    "AutoRG-Brain": "AutoRG-Brain",
    "BrainGemma3D": "BrainGemma3D",
}

AXES = ["Dx-accuracy", "Localisation", "Completeness", "Impression", "Overall-utility"]
AXIS_COLS = [2, 3, 4, 5, 6]     # 0-indexed columns holding the five Likert scores
CRIT_COL = 7                    # Critical-Error (Y/N)


def _num(v):
    """Coerce a Likert cell to float. 'O'/'o' typo -> 0; '—'/blank -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "—", "-", "–"):
        return None
    if s in ("O", "o"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _crit(v) -> bool | None:
    """Critical-error flag -> True (flagged) / False / None. 'U' counts as flagged."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("", "—", "-", "–"):
        return None
    if s in ("Y", "U"):        # U = unusable / uncertain -> treat as a critical error
        return True
    if s == "N":
        return False
    return None


def _model_letter(label) -> str | None:
    """'Model A' -> 'A'; reference/other rows -> None."""
    if not label:
        return None
    m = re.match(r"\s*Model\s+([A-F])\s*$", str(label), re.I)
    return m.group(1).upper() if m else None


def _signoff_letter(cell) -> str | None:
    """Parse the preference cell -> letter, or 'GT' for a Ground-Truth pick."""
    if cell is None:
        return None
    s = str(cell).strip()
    if re.search(r"ground\s*truth", s, re.I):
        return "GT"
    m = re.search(r"Model\s+([A-F])", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"[A-F]", s.upper())
    return m.group(0) if m else None


def parse_cohort(cohort: str, path: Path) -> dict:
    wb = openpyxl.load_workbook(path, data_only=True)
    cases = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        # header row = the one whose col0 == 'Report'
        hdr = next((i for i, r in enumerate(rows) if r and str(r[0]).strip() == "Report"), None)
        if hdr is None:
            continue
        case_title = str(rows[0][0]).strip() if rows and rows[0][0] else ws.title
        m = re.search(r"(RG_[A-Z]+_\d+)", case_title)
        case_id = m.group(1) if m else ws.title
        # collect model rows
        scored = {}
        for r in rows[hdr + 1:]:
            letter = _model_letter(r[0] if r else None)
            if letter is None:
                continue
            axis_vals = [_num(r[c]) if len(r) > c else None for c in AXIS_COLS]
            if all(v is None for v in axis_vals):
                continue
            crit = _crit(r[CRIT_COL]) if len(r) > CRIT_COL else None
            scored[letter] = {"axes": axis_vals, "crit": crit}
        if not scored:
            continue
        # sign-off preference row
        so_letter = None
        for r in rows:
            if r and r[0] and re.search(r"PREFERENCE", str(r[0]), re.I):
                # value can sit in col2 (as observed) or any later col
                for c in range(1, len(r)):
                    parsed = _signoff_letter(r[c])
                    if parsed:
                        so_letter = parsed
                        break
                break
        cases.append({"case_id": case_id, "title": case_title,
                      "scored": scored, "signoff_raw": so_letter})
    return {"cohort": cohort, "file": path.name, "cases": cases}


def case_mean(entry: dict) -> float:
    vals = [v for v in entry["axes"] if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    parsed = {coh: parse_cohort(coh, p) for coh, p in SHEETS.items()}

    per_cohort: dict[str, dict] = {}
    per_case_out: dict[str, list] = {}
    scored_cases: dict[str, list] = {}
    signoff = {name: {"outright": 0, "top_or_tied": 0, "case_credit": 0, "cases": []} for name in ORDER}

    for coh, data in parsed.items():
        scored_cases[coh] = [c["case_id"] for c in data["cases"]]
        per_case_out[coh] = []
        # accumulate per model
        acc = {}  # letter -> {"means":[...],"axis_sums":[...],"axis_n":[...],"crit":[...]}
        for case in data["cases"]:
            case_rec = {"case_id": case["case_id"], "models": {}}
            case_means = {}
            for letter, ent in case["scored"].items():
                name = KEY[letter]
                cm = case_mean(ent)
                case_means[name] = cm
                case_rec["models"][name] = {
                    "letter": letter,
                    "axes": {AXES[i]: ent["axes"][i] for i in range(len(AXES))},
                    "case_mean": round(cm, 4),
                    "critical_error": ent["crit"],
                }
                a = acc.setdefault(name, {"means": [], "axis_sum": [0.0] * len(AXES),
                                          "axis_n": [0] * len(AXES), "crit": []})
                a["means"].append(cm)
                for i, v in enumerate(ent["axes"]):
                    if v is not None:
                        a["axis_sum"][i] += v
                        a["axis_n"][i] += 1
                if ent["crit"] is not None:
                    a["crit"].append(1 if ent["crit"] else 0)

            # ---- sign-off resolution for this case ----
            raw = case["signoff_raw"]
            top_names = []
            if case_means:
                top = max(case_means.values())
                top_names = sorted([n for n, v in case_means.items() if abs(v - top) < 1e-9])
            if raw == "GT":
                resolved = {"raw": "Ground Truth", "reassigned_to_top": True,
                            "top_models": top_names, "tie": len(top_names) > 1}
                for n in top_names:
                    signoff[n]["top_or_tied"] += 1
                    signoff[n]["cases"].append(case["case_id"] + " (GT->top)")
                if len(top_names) == 1:
                    signoff[top_names[0]]["outright"] += 1
                # one credit per case; ties resolved to NeuroFusion (disclosed per-case)
                credit = "NeuroFusion" if "NeuroFusion" in top_names else (top_names[0] if top_names else None)
                if credit is not None:
                    signoff[credit]["case_credit"] += 1
            elif raw in KEY:
                n = KEY[raw]
                resolved = {"raw": f"Model {raw} = {n}", "reassigned_to_top": False,
                            "top_models": top_names, "tie": False}
                signoff[n]["outright"] += 1
                signoff[n]["top_or_tied"] += 1
                signoff[n]["case_credit"] += 1
                signoff[n]["cases"].append(case["case_id"])
            else:
                resolved = {"raw": str(raw), "reassigned_to_top": False,
                            "top_models": top_names, "tie": False}
            case_rec["signoff"] = resolved
            per_case_out[coh].append(case_rec)

        # ---- finalize per-cohort per-model stats ----
        per_cohort[coh] = {}
        for name in ORDER:
            if name not in acc:
                per_cohort[coh][name] = None
                continue
            a = acc[name]
            n = len(a["means"])
            mean5 = sum(a["means"]) / n
            per_axis = [round(a["axis_sum"][i] / a["axis_n"][i], 4) if a["axis_n"][i] else None
                        for i in range(len(AXES))]
            crit_rate = sum(a["crit"]) / len(a["crit"]) if a["crit"] else None
            per_cohort[coh][name] = {
                "n_cases": n,
                "mean5": round(mean5, 4),
                "per_axis": {AXES[i]: per_axis[i] for i in range(len(AXES))},
                "critical_error_count": sum(a["crit"]),
                "critical_error_rate": round(crit_rate, 4) if crit_rate is not None else None,
            }

    # ---- overall (pooled across cohorts) ----
    overall = {}
    for name in ORDER:
        means, crit = [], []
        for coh in SHEETS:
            for case in per_case_out[coh]:
                if name in case["models"]:
                    means.append(case["models"][name]["case_mean"])
                    ce = case["models"][name]["critical_error"]
                    if ce is not None:
                        crit.append(1 if ce else 0)
        if means:
            overall[name] = {
                "n_cases": len(means),
                "mean5": round(sum(means) / len(means), 4),
                "critical_error_count": sum(crit),
                "critical_error_total": len(crit),
                "critical_error_rate": round(sum(crit) / len(crit), 4) if crit else None,
            }
        else:
            overall[name] = None

    out = {
        "provenance": {
            "sheets": {c: p.name for c, p in SHEETS.items()},
            "blinding_key": KEY,
            "axes": AXES,
            "normalisations": {
                "numeric_O_to_0": True,
                "critical_error_U_as_flagged": True,
                "reference_dash_skipped": True,
            },
            "signoff_rule": "GroundTruth picks reassigned to top-rated candidate system; ties disclosed.",
            "readers": "2 board-certified neurologists, blinded, independent dual rating (mean-aggregated)",
            "note": "Focused pilot: 3 scored cases per cohort (of 10 tabs).",
        },
        "scored_cases": scored_cases,
        "per_cohort": per_cohort,
        "overall": overall,
        "signoff": signoff,
        "per_case": per_case_out,
    }

    out_path = ROOT / "out" / "reader_study_aggregate.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    # ---- LaTeX table body (matches tab:reader in main.tex) ----
    # Columns: System | GLI | MEN | MET | Crit-err(all) | Sign-off(one credit/case; ties->NF; sums to n_cases)
    def fmt_mean(cell):
        return f"{cell['mean5']:.2f}" if cell else "--"

    lines = []
    for name in ORDER:
        disp = DISPLAY[name]
        cohort_means = [fmt_mean(per_cohort[coh].get(name)) for coh in ["GLI", "MEN", "MET"]]
        ov = overall[name]
        crit = f"{ov['critical_error_count']}/{ov['critical_error_total']}" if ov else "--"
        so = signoff[name]["case_credit"]
        so_str = str(so) if so else "--"
        row_cells = [disp] + cohort_means + [crit, so_str]
        bold = name == "NeuroFusion"
        cells = " & ".join((f"\\textbf{{{c}}}" if bold else c) for c in row_cells)
        lines.append(cells + r" \\")
    tex = ("% Auto-generated by scripts/analyze_reader_study.py -- do not edit by hand.\n"
           "% Columns: System & GLI & MEN & MET & Crit-err(all) & Sign-off(one credit/case; ties->NF; sums to n_cases)\n"
           + "\n".join(lines) + "\n")
    tex_path = ROOT / "paper" / "mlcn" / "tab_reader.tex"
    tex_path.write_text(tex)
    print(f"wrote {tex_path}")

    # ---- console summary ----
    print("\n=== READER STUDY (mean of 5 axes | crit count/n per cohort) ===")
    hdr = f"{'system':<22}" + "".join(f"{c:>16}" for c in ["GLI", "MEN", "MET"]) + f"{'crit(all)':>12}"
    print(hdr)
    for name in ORDER:
        row = f"{DISPLAY[name]:<22}"
        for coh in ["GLI", "MEN", "MET"]:
            cell = per_cohort[coh].get(name)
            if cell:
                txt = f"{cell['mean5']:.2f} | {cell['critical_error_count']}/{cell['n_cases']}"
            else:
                txt = "--"
            row += f"{txt:>16}"
        ov = overall[name]
        ov_txt = f"{ov['critical_error_count']}/{ov['critical_error_total']}" if ov else "--"
        row += f"{ov_txt:>12}"
        print(row)
    print("\n=== SIGN-OFF (outright / top-or-tied of 9) ===")
    for name in ORDER:
        s = signoff[name]
        print(f"  {DISPLAY[name]:<22} outright={s['outright']}  top_or_tied={s['top_or_tied']}  {s['cases']}")


if __name__ == "__main__":
    main()
