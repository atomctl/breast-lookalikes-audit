#!/usr/bin/env python3
"""Ledger entry 55: the HISTAI-breast reports that enter labelling, and the comparison of two labellers.

Usage (from the project folder):
  python -I tools/histai_reports.py select            # -> data/histai_meta/reports_selected.jsonl
  python -I tools/histai_reports.py agree A.jsonl B.jsonl   # agreement of two label files; disagreements -> data/histai_meta/label_disagreements.jsonl
"""
import hashlib
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "data/histai_meta"
FIELDS = ("conclusion", "diagnosis", "diff_diagnostic", "micro_protocol", "additional_info")

# A case is left out only when a sentence of its conclusion AFFIRMS a malignant tumour. A cancer word that is denied,
# only suspected, or to be ruled out keeps the case in: those are the cases where a stain is most probable.
BAD = re.compile(r"invasive|infiltrat|metasta|lymphoma|sarcoma|melanoma|phyllodes|carcinoma(?! in situ)|\bcancer\b|adenocarcinoma", re.I)
HARMLESS = re.compile(r"not otherwise specified|no special type|non-?specific type|unspecified|\bNOS\b|\b(HER2|ER|PR|PgR)[^.;,]{0,25}(negative|not determined|undetermined)"
                      r"|(negative|undetermined)[^.;,]{0,12}(HER2|ER|PR)|triple[- ]negative|non-basal|without (lymph node )?metasta\w*"
                      r"|no (lymph node |evidence of (lymph node )?)?metasta\w*|not detected in[^.;]*lymph|FISH[^.;]*recommend\w*|recommend\w*[^.;]*(FISH|HER2)"
                      r"|lymphovascular invasion[^.;]{0,30}(not|absent)|no (lympho)?vascular invasion|R0|not free|non-luminal|non-keratinizing|no residual", re.I)
NEG = re.compile(r"\bno\b|\bnot\b|without|absent|absence|exclud|rule[sd]? out|ruling out|negative for|free of|differentiat|recommend|suspicious|cannot|possible|to clarify", re.I)


def affirms_malignant(conclusion):
    for s in re.split(r"(?<=[.;])\s+|\n", conclusion):
        if BAD.search(s) and not NEG.search(HARMLESS.sub(" ", s)):
            return True
    return False


def select():
    d = [r for r in json.loads((META / "metadata.json").read_text()) if r["case_mapping"].split("/")[1] == "HISTAI-breast"]
    keep = [{"case": r["case_mapping"].split("/")[-1], **{k: str(r.get(k) or "") for k in FIELDS}} for r in d if not affirms_malignant(r["conclusion"])]
    random.Random(0).shuffle(keep)   # batches then mix the kinds of case
    out = META / "reports_selected.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print("HISTAI-breast cases", len(d), "| selected", len(keep), "| sha256", hashlib.sha256(out.read_bytes()).hexdigest()[:16])


KEYS = ("group", "stain", "stain_status", "reason", "udh_in_conclusion", "said_no_stain_needed", "ck56")


def read(path):
    return {r["case"]: r for r in (json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip())}


def agree(a_path, b_path):
    a, b = read(a_path), read(b_path)
    both = sorted(set(a) & set(b))
    print("cases", len(a), len(b), "in both", len(both))
    diff = []
    for k in KEYS:
        same = sum(a[c].get(k) == b[c].get(k) for c in both)
        labels = sorted({str(a[c].get(k)) for c in both} | {str(b[c].get(k)) for c in both})
        pe = sum((sum(str(a[c].get(k)) == l for c in both) / len(both)) * (sum(str(b[c].get(k)) == l for c in both) / len(both)) for l in labels)
        po = same / len(both)
        print(f"  {k:22s} same {same}/{len(both)} = {po:.3f} | kappa {(po - pe) / (1 - pe) if pe < 1 else 1:.3f}")
    for c in both:
        ks = [k for k in ("group", "stain", "stain_status", "reason", "ck56") if a[c].get(k) != b[c].get(k)]
        if ks:
            diff.append({"case": c, "differs": ks, "a": {k: a[c].get(k) for k in ks}, "b": {k: b[c].get(k) for k in ks}})
    (META / "label_disagreements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in diff))
    print("cases with a difference in group, stain, status, reason or ck56:", len(diff))


if __name__ == "__main__":
    {"select": select, "agree": agree}[sys.argv[1]](*sys.argv[2:])
