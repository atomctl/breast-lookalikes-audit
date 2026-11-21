#!/usr/bin/env python3
"""Ledger entry 42: the transformer with tile positions, trained on the training patients, checked on the validation patients.

Usage (from the project folder):
  python -I tools/grayzone_arrangement_val.py   # -> outputs/grayzone/arrangement_val.json, val_prob_transformer.npy
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_arrangement as ga
import grayzone_features as gf
import short_lists as sl
from train_classifier import FEATS, patient_draws, range95, rows, weighted_f1

STARTS = 5


def regions_of(split):
    out = []
    for r in rows(split):
        z = np.load(FEATS / "uni" / (r["file"][:-4] + ".npz"))
        keep = z["tissue"] >= 0.5
        keep = keep if keep.any() else np.ones(len(keep), bool)
        out.append((z["feats"][keep].astype(np.float32), (z["coords"][keep] / 224).astype(np.float32)))
    return out


def main():
    train, val = rows("train"), rows("val")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, _, vprob, vy, vpatient = sl.load()
    assert len(val) == len(vy) and not set(patient) & set(vpatient)
    reg, vreg = regions_of("train"), regions_of("val")
    t = np.zeros((len(vy), 7))
    for seed in range(STARTS):
        model, mean, std = ga.fit(ga.Transformer, reg, y, patient, np.arange(len(y)), seed)
        t += ga.predict(model, vreg, np.arange(len(vy)), mean, std) / STARTS
        print("start", seed, "done", flush=True)
    np.save(sl.ROOT / "outputs/grayzone/val_prob_transformer.npy", t)
    draws = patient_draws(vpatient, np.random.default_rng(0))
    result = {"validation_patients": int(len(np.unique(vpatient))), "regions": len(vy), "gray_regions": int(np.isin(vy, gf.GRAY).sum()), "adh_regions": int((vy == 4).sum())}
    probs = {"E adopted classifier": vprob, "T transformer": t, "average of E and T": (vprob + t) / 2}
    for name, p in probs.items():
        macro, adh = gf.gray_f1(p, vy)
        result[name] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "weighted_f1_7_types": round(weighted_f1(vy, p.argmax(1)), 4),
                        "gray_macro_f1_range95": range95([gf.gray_f1(p, vy, i)[0] for i in draws])}
        if name != "E adopted classifier":
            d = [gf.gray_f1(p, vy, i)[0] - gf.gray_f1(vprob, vy, i)[0] for i in draws]
            result[name].update({"gray_macro_f1_minus_E": round(macro - result["E adopted classifier"]["gray_macro_f1"], 4), "range95_of_difference": range95(d),
                                 "confirmed": bool(np.percentile(d, 2.5) > 0)})
    (sl.ROOT / "outputs/grayzone/arrangement_val.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
