#!/usr/bin/env python3
"""Reader study kit: the same gray-zone cases read by pathologists without and with the system.

Usage (from the project folder):
  python -I tools/build_reader_study.py   # -> outputs/reader_study/index.html, cases.js, img/, key.json

Cases come from the 30 test patients (no classifier was trained or tuned on them). The page holds no labels:
key.json (the agreed types) stays with the organiser. Real tissue images: local use only (BRACS, CC BY-NC 4.0).
"""
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl

OUT = sl.ROOT / "outputs/reader_study"
SRC = sl.ROOT / "data/bracs/BRACS_RoI/latest_version"
PER_GRAY, PER_ANCHOR, PER_PATIENT = 20, 4, 3
GRAY, ANCHOR = ["UDH", "FEA", "ADH", "DCIS"], ["N", "PB", "IC"]


def main():
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with open(sl.ROOT / "outputs/final/regions_test.csv") as f:
        rows = list(csv.DictReader(f))
    prob = np.array([[float(r[f"p_{t}"]) for t in sl.TYPES] for r in rows])
    patient = np.array([int(r["patient"]) for r in rows])
    act, lists = pp.apply_system(prob, patient)  # the current system (v5: cut-off of entry 17 plus the ADH guard of entry 36)
    rng = np.random.default_rng(0)
    chosen = []
    for t in GRAY + ANCHOR:
        per_patient = {}
        for i in rng.permutation([i for i, r in enumerate(rows) if r["label"] == t]):
            p = rows[i]["patient"]
            if per_patient.get(p, 0) < PER_PATIENT and sum(rows[j]["label"] == t for j in chosen) < (PER_GRAY if t in GRAY else PER_ANCHOR):
                chosen.append(int(i))
                per_patient[p] = per_patient.get(p, 0) + 1
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    cases, key = [], {}
    for n, i in enumerate(rng.permutation(chosen)):
        cid = f"c{n + 1:03d}"
        im = Image.open(SRC / rows[i]["file"]).convert("RGB").reduce(2)  # 20x, 0.5 micrometres per pixel
        im.save(OUT / "img" / f"{cid}.jpg", quality=90)
        order = np.argsort(-prob[i])
        on_list = [sl.TYPES[k] for k in order if lists[i, k]] or [sl.TYPES[order[0]]]
        cases.append({"id": cid, "w": im.width, "h": im.height, "width_mm": round(im.width * 0.0005, 2),
                      "ai": {"alone": bool(act[i]), "answer": sl.TYPES[order[0]], "list": on_list,
                             "prob": {sl.TYPES[k]: round(float(prob[i, k]), 2) for k in order[:3]}}})
        key[cid] = {"label": rows[i]["label"], "file": rows[i]["file"], "patient": rows[i]["patient"]}
    (OUT / "cases.js").write_text("const CASES = " + json.dumps(cases) + ";\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    (OUT / "index.html").write_text((pathlib.Path(__file__).parent / "reader_study_template.html").read_text())
    labels = [key[c["id"]]["label"] for c in cases]
    alone = [c["ai"]["alone"] for c in cases]
    right = [c["ai"]["answer"] == key[c["id"]]["label"] for c in cases]
    hit = [key[c["id"]]["label"] in c["ai"]["list"] for c in cases]
    print(len(cases), "cases:", {t: labels.count(t) for t in GRAY + ANCHOR}, "| patients:", len({k["patient"] for k in key.values()}))
    print("system on these cases: answers alone", sum(alone), "(wrong:", sum(a and not r for a, r in zip(alone, right)), ") | asks", len(cases) - sum(alone),
          "(list holds the agreed type:", sum(h and not a for h, a in zip(hit, alone)), ") | top answer right overall:", sum(right))


if __name__ == "__main__":
    main()
