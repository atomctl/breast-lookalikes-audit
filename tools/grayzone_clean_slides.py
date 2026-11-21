#!/usr/bin/env python3
"""Ledger entry 54: the frozen systems v4 and v5 seen by their decision, on slides without carcinoma.

For the regions labelled ADH, UDH or FEA on slides whose BRACS label is not DCIS or IC: how many are answered alone,
what the lone answers are, and how many lone answers fall on the wrong side of "cut out or not" (FEA counted both ways).
A description of saved answers; nothing is trained.

Usage (from the project folder):
  python -I tools/grayzone_clean_slides.py   # -> outputs/grayzone/clean_slides.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl
from grayzone_slide_origin import slide_labels
from train_classifier import rows

NAMES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
LINES = {"cut out = ADH, DCIS, IC": {4, 5, 6}, "cut out = FEA, ADH, DCIS, IC": {3, 4, 5, 6}}


def main():
    prob, y, patient, _, vprob, vy, vpatient = sl.load()
    r = rows("train") + rows("val")
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    lab = slide_labels()
    clean = np.array([lab[x["slide"]] not in ("DCIS", "IC") for x in r])
    top = prob.argmax(1)
    score = pp.signals(prob, prob, patient)["patient context"]
    system = {v: json.loads((sl.ROOT / f"outputs/promise/system_{v}.json").read_text()) for v in ("v4", "v5")}
    alone = {"v4": score >= system["v4"]["cut_off"],
             "v5": (score >= system["v5"]["cut_off"]) & (prob[:, 4] < system["v5"]["adh_guard"])}
    result = {"regions_on_clean_slides": int(clean.sum()), "systems": {}, "adh_per_patient": []}
    for v, a in alone.items():
        out = {}
        for t in (4, 2, 3):
            m = clean & (y == t)
            la = m & a
            out[NAMES[t]] = {"regions": int(m.sum()), "patients": len(set(patient[m])), "answered_alone": int(la.sum()),
                             "lone_answers_by_type": {NAMES[k]: int((top[la] == k).sum()) for k in range(7) if (top[la] == k).any()},
                             "lone_answers_of_another_type": int((la & (top != t)).sum())}
        for name, cut in LINES.items():
            truth, said = np.isin(y, list(cut)), np.isin(top, list(cut))
            m = clean & np.isin(y, [2, 3, 4]) & a
            out[name] = {"missed (truth cut out, lone answer not)": int((m & truth & ~said).sum()), "of regions on the cut-out side": int((clean & np.isin(y, [2, 3, 4]) & truth).sum()),
                         "not needed (truth not, lone answer cut out)": int((m & ~truth & said).sum()), "of regions on the other side": int((clean & np.isin(y, [2, 3, 4]) & ~truth).sum())}
        result["systems"][v] = out
    val = set(vpatient.tolist())
    for p in sorted(set(patient[clean & (y == 4)])):
        m = clean & (y == 4) & (patient == p)
        result["adh_per_patient"].append({"patient": int(p), "set": "validation" if p in val else "training", "adh_regions": int(m.sum()),
                                          **{f"{v}_{k}": int(n) for v, a in alone.items() for k, n in (("alone", (m & a).sum()), ("alone_wrong", (m & a & (top != 4)).sum()))}})
    (sl.ROOT / "outputs/grayzone/clean_slides.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
