"""Verbalizer faithfulness: does each narrative sentence FOLLOW from the structured
record it was generated from?  (RadFact-style logical precision, frozen + post-hoc.)

NeuroFusion's structured-record-then-verbalize design gives us something end-to-end
VLMs lack: an explicit, machine-checkable PREMISE (the fused structured record). We
check the frozen base-Mistral narrative against that premise sentence by sentence with
a local Llama-3.1-8B entailment judge -- no retraining, no GT report needed.

For each verbalized case (fields: case_id, fused_json, generated_prose) we:
  1. render the structured record's TYPED fields as the premise,
  2. split the prose into sentences,
  3. label each sentence SUPPORTED / UNSUPPORTED / CONTRADICTED w.r.t. the premise.

Aggregate: per-case faithfulness = supported / total; corpus faithfulness, hallucination
rate (unsupported+contradicted), contradiction rate, and % of cases with >=1 contradiction.
Per-sentence labels are kept so a worked example can be shown in the paper.

Usage:
  python scripts/verbalizer_faithfulness.py \
      --pred out/verbalized_mistral_test_baseLM.jsonl \
      --judge meta-llama/Llama-3.1-8B-Instruct --device cuda \
      --out out/xai_faithfulness_mistral_test.json
"""
from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path

log = logging.getLogger("verbalizer_faithfulness")
REPO = Path(__file__).resolve().parents[1]

ENTAIL_PROMPT = """You are auditing a radiology narrative against the STRUCTURED FINDINGS it was generated from. The structured findings are the ONLY ground truth; treat them as complete.

STRUCTURED FINDINGS:
{record}

Classify the SENTENCE below into exactly one label:
- SUPPORTED: every clinical claim in the sentence follows from the structured findings.
- CONTRADICTED: a clinical claim in the sentence conflicts with the structured findings.
- UNSUPPORTED: the sentence introduces a clinical claim that is neither stated in nor derivable from the structured findings (a possible hallucination). Generic non-clinical phrasing with no new claim counts as SUPPORTED.

SENTENCE: {sentence}

Respond ONLY with a JSON object: {{"label": "SUPPORTED|CONTRADICTED|UNSUPPORTED", "reason": "<one short clause>"}}"""


def render_record(fused_json_str) -> str:
    """Render the TYPED structured fields (not free-text findings) as the premise.

    Accepts either a JSON string or an already-parsed dict (the verbalizer writes
    ``fused_json`` as a dict), so handle both rather than assuming a string.
    """
    if isinstance(fused_json_str, dict):
        r = fused_json_str
    else:
        try:
            r = json.loads(fused_json_str)
        except Exception:
            return str(fused_json_str)[:1200]
    lines = []
    if r.get("hemisphere"):
        lines.append(f"Hemisphere: {r['hemisphere']}")
    if r.get("overall_mass_effect"):
        lines.append(f"Overall mass effect: {r['overall_mass_effect']}")
    ddx = r.get("differential_diagnosis") or []
    if ddx:
        lines.append("Differential diagnosis: " + "; ".join(str(d) for d in ddx))
    for les in (r.get("lesions") or []):
        idx = les.get("lesion_index", "?")
        parts = []
        for k in ("location", "composition", "enhancement_pattern", "edema_severity",
                  "mass_effect", "midline_shift", "ventricular_compression",
                  "brainstem_compression", "midbrain_compression", "axis_shift",
                  "size_mm", "margins"):
            v = les.get(k)
            if v not in (None, "", []):
                parts.append(f"{k}={v}")
        lines.append(f"Lesion {idx}: " + ", ".join(parts))
    return "\n".join(lines) if lines else str(fused_json_str)[:1200]


def split_sentences(prose: str) -> list[str]:
    prose = (prose or "").replace("\n", " ").strip()
    # split on sentence-final punctuation followed by space+capital, keep it simple/robust
    raw = re.split(r"(?<=[.!?])\s+", prose)
    out = []
    for s in raw:
        s = s.strip()
        if len(s) >= 12 and re.search(r"[A-Za-z]", s):   # drop fragments
            out.append(s)
    return out


def parse_label(resp: str) -> dict | None:
    m = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
    if not m:
        # fall back: bare keyword
        for lab in ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED"):
            if lab in resp.upper():
                return {"label": lab, "reason": ""}
        return None
    try:
        obj = json.loads(m.group(0))
        lab = str(obj.get("label", "")).upper()
        if lab not in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"):
            for k in ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED"):
                if k in lab:
                    lab = k; break
            else:
                return None
        return {"label": lab, "reason": str(obj.get("reason", ""))[:160]}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path, help="verbalized jsonl (case_id, fused_json, generated_prose)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--judge", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    log.info(f"loading judge {args.judge}")
    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype="bfloat16", device_map=args.device)
    model.eval()

    records = [json.loads(l) for l in open(args.pred)]
    if args.limit:
        records = records[:args.limit]

    per_case = []
    for i, rec in enumerate(records):
        cid = rec.get("case_id", f"case{i}")
        prose = rec.get("generated_prose") or ""
        premise = render_record(rec.get("fused_json") or (rec.get("predictions") or [""])[0])
        sents = split_sentences(prose)
        labels = []
        for s in sents:
            prompt = ENTAIL_PROMPT.format(record=premise, sentence=s)
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          return_tensors="pt", add_generation_prompt=True, return_dict=True)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            ilen = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][ilen:], skip_special_tokens=True)
            lab = parse_label(resp) or {"label": "UNSUPPORTED", "reason": "unparseable"}
            lab["sentence"] = s
            labels.append(lab)
        n = len(labels)
        nsup = sum(1 for l in labels if l["label"] == "SUPPORTED")
        ncon = sum(1 for l in labels if l["label"] == "CONTRADICTED")
        nuns = sum(1 for l in labels if l["label"] == "UNSUPPORTED")
        per_case.append({
            "case_id": cid, "n_sent": n, "n_supported": nsup,
            "n_unsupported": nuns, "n_contradicted": ncon,
            "faithfulness": (nsup / n) if n else 1.0,
            "hallucination_rate": ((nuns + ncon) / n) if n else 0.0,
            "labels": labels,
        })
        if (i + 1) % 5 == 0:
            log.info(f"  [{i+1}/{len(records)}] {cid}: faith={per_case[-1]['faithfulness']:.2f}")

    import statistics as st
    tot_sent = sum(c["n_sent"] for c in per_case)
    tot_sup = sum(c["n_supported"] for c in per_case)
    tot_con = sum(c["n_contradicted"] for c in per_case)
    tot_uns = sum(c["n_unsupported"] for c in per_case)
    agg = {
        "pred_path": str(args.pred),
        "n_cases": len(per_case),
        "n_sentences": tot_sent,
        "corpus_faithfulness": (tot_sup / tot_sent) if tot_sent else 0.0,
        "corpus_hallucination_rate": ((tot_uns + tot_con) / tot_sent) if tot_sent else 0.0,
        "corpus_contradiction_rate": (tot_con / tot_sent) if tot_sent else 0.0,
        "mean_case_faithfulness": st.mean([c["faithfulness"] for c in per_case]) if per_case else 0.0,
        "pct_cases_with_contradiction": (sum(1 for c in per_case if c["n_contradicted"] > 0) / len(per_case)) if per_case else 0.0,
        "pct_cases_fully_faithful": (sum(1 for c in per_case if c["hallucination_rate"] == 0) / len(per_case)) if per_case else 0.0,
        "per_case": per_case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(agg, open(args.out, "w"), indent=2)
    log.info(f"wrote {args.out}: corpus_faithfulness={agg['corpus_faithfulness']:.3f} "
             f"halluc={agg['corpus_hallucination_rate']:.3f} contra={agg['corpus_contradiction_rate']:.3f} "
             f"(n_cases={agg['n_cases']}, n_sent={agg['n_sentences']})")


if __name__ == "__main__":
    main()
