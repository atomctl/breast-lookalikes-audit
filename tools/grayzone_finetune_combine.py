#!/usr/bin/env python3
"""Ledger entry 57: put the five held-out folds of the region-loss fine-tune together and score them as one, against the adopted classifier.

Every region is scored by the model that never saw its patient. Same yardsticks and by-patient ranges as `grayzone_finetune.py`.

Usage (from the project folder):
  python -I tools/grayzone_finetune_combine.py    # -> outputs/grayzone/finetune_region_all.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_features as gf
import short_lists as sl
from grayzone_slide_origin import SEVERITY, slide_labels
from train_classifier import patient_draws, range95, rows, weighted_f1

NAMES = ("plain_UNI", "tuned_UNI_project_classifier", "tuned_UNI_own_answer", "adopted_classifier")


def main():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    probs = {n: np.full((len(train), 7), np.nan) for n in NAMES}
    per_fold = []
    for k in range(5):
        z = np.load(sl.ROOT / f"outputs/grayzone/finetune_region_fold{k}.npz")
        for n in NAMES:
            probs[n][z["held"]] = z[n]
        r = json.loads((sl.ROOT / f"outputs/grayzone/finetune_region_fold{k}.json").read_text())
        own, bar = r["tuned UNI, own answer"], r["adopted classifier"]
        per_fold.append({"fold": k, "patients": r["held_out_patients"], "regions": r["held_out_regions"], **r["held_out_adh"],
                         "own": [own[m] for m in ("gray_macro_f1", "adh_f1", "group_A_named_ADH", "group_C_named_ADH")],
                         "adopted": [bar[m] for m in ("gray_macro_f1", "adh_f1", "group_A_named_ADH", "group_C_named_ADH")],
                         "own_minus_adopted_adh_f1": own["minus adopted classifier"]["adh_f1"], "adh_range95": own["minus adopted classifier"]["adh_range95"]})
    assert all(np.isfinite(p).all() for p in probs.values()), "a region has no held-out answer"
    lab = slide_labels()
    adh = y == 4
    grp_a = np.array([SEVERITY[lab[r["slide"]]] == 4 for r in train]) & adh   # ADH on a slide without carcinoma
    named = lambda p, m: int((np.array(gf.GRAY)[p[m][:, gf.GRAY].argmax(1)] == 4).sum())
    result = {"regions": len(train), "patients": int(len(np.unique(patient))), "group_A_regions": int(grp_a.sum()), "group_C_regions": int((adh & ~grp_a).sum()), "per_fold": per_fold}
    draws = patient_draws(patient, np.random.default_rng(0))
    ref = probs["adopted_classifier"]
    for n, p in probs.items():
        macro, f_adh = gf.gray_f1(p, y)
        result[n] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(f_adh, 4), "weighted_f1_7_types": round(weighted_f1(y, p.argmax(1)), 4),
                     "group_A_named_ADH": named(p, grp_a), "group_C_named_ADH": named(p, adh & ~grp_a)}
        if n != "adopted_classifier":
            dm = [gf.gray_f1(p, y, i)[0] - gf.gray_f1(ref, y, i)[0] for i in draws]
            da = [gf.gray_f1(p, y, i)[1] - gf.gray_f1(ref, y, i)[1] for i in draws]
            ga = [named(p[i], grp_a[i]) - named(ref[i], grp_a[i]) for i in draws]
            result[n]["minus adopted classifier"] = {"gray_macro_f1": round(macro - gf.gray_f1(ref, y)[0], 4), "gray_range95": range95(dm),
                                                     "adh_f1": round(f_adh - gf.gray_f1(ref, y)[1], 4), "adh_range95": range95(da), "group_A_regions_range95": range95(ga)}
    (sl.ROOT / "outputs/grayzone/finetune_region_all.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
