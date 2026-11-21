#!/usr/bin/env python3
"""Ledger entry 23: what kinds of error does the classifier make on ADH? A description, nothing is tuned.

Usage (from the project folder):
  python -I tools/grayzone_errors.py   # -> outputs/grayzone/adh_errors.json and contact sheets
"""
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import FEATS, pooled, rows

OUT = sl.ROOT / "outputs/grayzone"
ADH, T = 4, sl.TYPES


def area_mm2(file):
    w, h = np.load(FEATS / "uni" / (file[:-4] + ".npz"))["size"]  # pixels at 0.5 um
    return float(w * h * 0.25 / 1e6)


def describe(held, prob, y, patient, name):
    pred = prob.argmax(1)
    slide = np.array([r["slide"] for r in held])
    area = np.array([area_mm2(r["file"]) for r in held])
    is_adh = y == ADH
    wrong = is_adh & (pred != ADH)
    out = {"adh_regions": int(is_adh.sum()), "adh_patients": int(len(np.unique(patient[is_adh]))), "adh_right": int((is_adh & (pred == ADH)).sum()),
           "adh_called": {T[k]: int((is_adh & (pred == k)).sum()) for k in range(7)},
           "called_adh_but_is": {T[k]: int(((pred == ADH) & (y == k)).sum()) for k in range(7)}}
    # (2) patients
    per = sorted(((int((wrong & (patient == p)).sum()), int((is_adh & (patient == p)).sum()), int(p)) for p in np.unique(patient[is_adh])), reverse=True)
    out["errors_by_patient_top5 (wrong, ADH regions, patient)"] = per[:5]
    out["share_of_errors_from_top_3_patients"] = round(sum(w for w, _, _ in per[:3]) / max(wrong.sum(), 1), 3)
    out["patients_with_most_of_their_ADH_wrong"] = sum(w > n / 2 for w, n, _ in per)
    # (3) confident or hesitant
    top = prob.max(1)
    order = np.argsort(-prob, axis=1)
    out["wrong_and_confident (top probability at least 0.7)"] = round(float((top[wrong] >= 0.7).mean()), 3)
    out["wrong_but_ADH_is_runner_up"] = round(float((order[wrong, 1] == ADH).mean()), 3)
    out["median_probability_given_to_ADH_when_wrong"] = round(float(np.median(prob[wrong, ADH])), 3)
    out["median_probability_given_to_ADH_when_right"] = round(float(np.median(prob[is_adh & ~wrong, ADH])), 3)
    # (4) size
    med = lambda m: round(float(np.median(area[m])), 3) if m.any() else None
    out["median_area_mm2"] = {"ADH called ADH": med(is_adh & ~wrong), "ADH called UDH": med(is_adh & (pred == 2)), "ADH called DCIS": med(is_adh & (pred == 5)),
                              "ADH called FEA": med(is_adh & (pred == 3)), "true UDH": med(y == 2), "true FEA": med(y == 3), "true ADH": med(is_adh), "true DCIS": med(y == 5)}
    # (5) the slide also holds regions of the type ADH was mistaken for
    types_on_slide = collections.defaultdict(set)
    for s, k in zip(slide, y):
        types_on_slide[s].add(int(k))
    has = lambda m, k: round(float(np.mean([k in types_on_slide[s] for s in slide[m]])), 3) if m.any() else None
    out["slide_also_has_DCIS"] = {"ADH called DCIS": has(is_adh & (pred == 5), 5), "ADH called ADH": has(is_adh & ~wrong, 5), "all ADH": has(is_adh, 5)}
    out["slide_also_has_UDH"] = {"ADH called UDH": has(is_adh & (pred == 2), 2), "ADH called ADH": has(is_adh & ~wrong, 2), "all ADH": has(is_adh, 2)}
    print(name, json.dumps(out, indent=1), flush=True)
    return out, wrong, pred


def main():
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    held = rows("train") + rows("val")
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    result = {}
    result["121 training and validation patients"], wrong, pred = describe(held, prob, y, patient, "121 patients")

    # (6) label consistency among look-alikes, in UNI's own features, neighbours from other patients
    x = np.vstack([np.hstack([pooled("uni", s, 0.5), np.load(sl.ROOT / "outputs/m1/thunder" / f"uni_{s}.npy")]) for s in ("train", "val")])
    x = (x - x.mean(0)) / x.std(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    sim[patient[:, None] == patient[None, :]] = -np.inf
    nn = np.argsort(-sim, axis=1)[:, :5]
    same = (y[nn] == y[:, None]).mean(1)
    result["neighbours_with_the_same_label_of_5"] = {T[k]: round(float(same[y == k].mean()), 3) for k in range(7)}
    result["share_of_library_per_type"] = {T[k]: round(float((y == k).mean()), 3) for k in range(7)}
    adh = y == ADH
    result["ADH_neighbours_are"] = {T[k]: round(float((y[nn[adh]] == k).mean()), 3) for k in range(7)}
    result["ADH_same_label_neighbours"] = {"when the classifier is right": round(float(same[adh & ~wrong].mean()), 3), "when it is wrong": round(float(same[adh & wrong].mean()), 3)}
    print(json.dumps({k: result[k] for k in ("neighbours_with_the_same_label_of_5", "ADH_neighbours_are", "ADH_same_label_neighbours")}, indent=1))

    # the test patients, described only
    with open(sl.ROOT / "data/regions.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    with open(sl.ROOT / "outputs/final/regions_test.csv") as f:
        saved = list(csv.DictReader(f))
    tprob = np.array([[float(r[f"p_{t}"]) for t in T] for r in saved])
    result["30 test patients (described only)"], _, _ = describe(test, tprob, np.array([int(r["label"]) for r in test]), np.array([int(r["patient"]) for r in test]), "test patients")

    # contact sheets for a human: ADH read right, read as UDH, read as DCIS (first 12 of each, in file order)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 200_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    for label, mask in (("ADH_called_ADH", adh & ~wrong), ("ADH_called_UDH", adh & (pred == 2)), ("ADH_called_DCIS", adh & (pred == 5))):
        idx = np.flatnonzero(mask)[::max(1, mask.sum() // 12)][:12]
        sheet = Image.new("RGB", (4 * 330, 3 * 330), "white")
        for n, i in enumerate(idx):
            im = Image.open(sl.ROOT / "data/bracs/BRACS_RoI/latest_version" / held[i]["file"]).convert("RGB").reduce(2)
            im.thumbnail((320, 320))
            sheet.paste(im, (n % 4 * 330 + 5, n // 4 * 330 + 5))
        sheet.save(OUT / f"{label}.jpg", quality=85)
    (OUT / "adh_errors.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
