#!/usr/bin/env python3
"""Ledger entry 47: is ADH two populations in our regions, by the label of the slide a region comes from?

Usage (from the project folder):
  python -I tools/grayzone_slide_origin.py   # -> outputs/grayzone/slide_origin.json

Uses the saved out-of-fold probabilities of the adopted classifier; nothing is trained except the two centres of test 3.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import ROOT, TYPES, patient_draws, pooled, range95, rows

OUT = ROOT / "outputs/grayzone"
N, PB, UDH, FEA, ADH, DCIS, IC = range(7)
SEVERITY = {t: k for k, t in enumerate(TYPES)}


def slide_labels():
    import pandas as pd
    d = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx")
    return dict(zip(d["WSI Filename"], d["WSI label"]))


def auc(y, s, patient, rng):
    draws = patient_draws(patient, rng, 1000)
    d = [roc_auc_score(y[i], s[i]) for i in draws if 0 < y[i].sum() < len(i)]
    return {"auc": round(float(roc_auc_score(y, s)), 3), "range95": range95(d)}


def share_diff(flag, group, patient, rng):
    """Share of regions with the flag, group True minus group False, with its by-patient range."""
    draws = patient_draws(patient, rng, 1000)
    d = [flag[i][group[i]].mean() - flag[i][~group[i]].mean() for i in draws if group[i].any() and (~group[i]).any()]
    return {"C": round(float(flag[group].mean()), 3), "A": round(float(flag[~group].mean()), 3),
            "difference": round(float(flag[group].mean() - flag[~group].mean()), 3), "range95": range95(d)}


def report(prob, y, patient, slide, lab, rng, axis=None):
    slab = np.array([SEVERITY[lab[s]] for s in slide])
    adh = y == ADH
    c = slab[adh] >= DCIS  # group C: the slide is labelled DCIS or IC
    assert (slab[adh] >= ADH).all()
    p, pat = prob[adh], patient[adh]
    top = p.argmax(1)
    out = {"adh_regions": {"A": int((~c).sum()), "C": int(c.sum())},
           "patients": {"A": int(len(set(pat[~c]))), "C": int(len(set(pat[c])))},
           "test1_dcis_over_adh_score": auc(c, np.log(p[:, DCIS] / p[:, ADH]), pat, rng),
           "test2_called_dcis_or_ic": share_diff(top >= DCIS, c, pat, rng),
           "top_type": {g: {TYPES[k]: round(float((top[m] == k).mean()), 3) for k in range(7)} for g, m in (("A", ~c), ("C", c))},
           "adh_recall_among_gray_four": share_diff(p[:, UDH:DCIS + 1].argmax(1) == 2, c, pat, rng)}
    if axis is not None:
        out["test3_position_on_benign_malignant_line"] = dict(auc(c, axis[adh], pat, rng),
                                                             medians={"A": round(float(np.median(axis[adh][~c])), 3), "C": round(float(np.median(axis[adh][c])), 3)})
    # pair AUCs with only one group as the ADH side
    out["pair_auc"] = {}
    for name, other, score in (("ADH-DCIS", DCIS, np.log(prob[:, DCIS] / prob[:, ADH])), ("UDH-ADH", UDH, np.log(prob[:, ADH] / prob[:, UDH]))):
        for g, keep in (("all ADH", adh), ("ADH of group A", adh & (slab == ADH)), ("ADH of group C", adh & (slab >= DCIS))):
            m = keep | (y == other)
            t = (y[m] == DCIS) if other == DCIS else (y[m] == ADH)
            out["pair_auc"][f"{name}, {g}"] = auc(t, score[m], patient[m], rng)
    # the same split for UDH regions: from UDH slides against slides with a more severe label
    u = y == UDH
    worse = slab[u] > UDH
    out["udh_regions"] = {"from UDH slides": int((~worse).sum()), "from more severe slides": int(worse.sum()),
                          "adh_over_udh_score, more severe against UDH slides": auc(worse, np.log(prob[u][:, ADH] / prob[u][:, UDH]), patient[u], rng)}
    return out, c


def main():
    rng = np.random.default_rng(0)
    lab = slide_labels()
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    train, held = rows("train"), rows("val")
    slide, vslide = np.array([r["slide"] for r in train]), np.array([r["slide"] for r in held])
    # test 3: position on the line from the benign centre to the malignant centre, centres from the other folds
    x = pooled("uni", "train", 0.5)
    axis = np.zeros(len(y))
    for a, b in folds:
        ben, mal = x[a][y[a] <= UDH].mean(0), x[a][y[a] >= DCIS].mean(0)
        axis[b] = (x[b] - ben) @ (mal - ben) / ((mal - ben) @ (mal - ben))
    result = {"slide_label_of_adh_regions": {}}
    result["train"], c = report(prob, y, patient, slide, lab, rng, axis)
    tiles = np.array([len(np.load(ROOT / "data/features/uni" / (r["file"][:-4] + ".npz"))["tissue"]) for r in train if int(r["label"]) == ADH])
    result["train"]["tiles_per_adh_region_median"] = {"A": float(np.median(tiles[~c])), "C": float(np.median(tiles[c]))}
    result["val"], _ = report(vprob, vy, vpatient, vslide, lab, rng)
    for name, s, yy in (("train", slide, y), ("val", vslide, vy)):
        labs = [lab[v] for v in s[yy == ADH]]
        result["slide_label_of_adh_regions"][name] = {t: labs.count(t) for t in sorted(set(labs))}
    (OUT / "slide_origin.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
