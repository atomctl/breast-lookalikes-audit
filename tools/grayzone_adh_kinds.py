#!/usr/bin/env python3
"""Ledger entry 50: does the classifier find the ADH of ADH-only slides more often when training treats the two kinds of ADH evenly?

Usage (from the project folder):
  python -I tools/grayzone_adh_kinds.py   # -> outputs/grayzone/adh_kinds.json

V0 the adopted classifier; V1 groups A and C carry half of the ADH weight each; V2 the two groups are separate classes in training.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_features as gf
import short_lists as sl
from grayzone_slide_origin import SEVERITY, slide_labels
from train_classifier import ROOT, classifier, patient_draws, range95, rows

OUT = ROOT / "outputs/grayzone"
UDH, FEA, ADH, DCIS = 2, 3, 4, 5
C = sl.ENGINE["C"]


def fit_predict(kind, xa, ya, ga, xb):
    """ga: True where an ADH training region comes from an ADH-only slide (group A). Returns 7 probabilities per region."""
    if kind == "V0":
        return classifier(C).fit(xa, ya).predict_proba(xb)
    adh = ya == ADH
    if kind == "V1":
        w = np.ones(len(ya))
        for grp in (adh & ga, adh & ~ga):
            if grp.any():
                w[grp] = 0.5 * adh.sum() / grp.sum()
        return classifier(C).fit(xa, ya, logisticregression__sample_weight=w).predict_proba(xb)
    y8 = ya.copy()
    y8[adh & ga] = 7                      # V2: group A becomes its own class
    clf = classifier(C).fit(xa, y8)
    p = clf.predict_proba(xb)
    out = np.zeros((len(xb), 7))
    for j, c in enumerate(clf.classes_):
        out[:, ADH if c == 7 else c] += p[:, j]
    return out


def measures(prob, y, grp_a, i=slice(None)):
    prob, y, a = prob[i], y[i], grp_a[i]
    top = np.array(gf.GRAY)[prob[:, gf.GRAY].argmax(1)]
    macro, adh_f1 = gf.gray_f1(prob, y)
    m = (y == UDH) | ((y == ADH) & a)
    return {"group_A_named_ADH": float((top[(y == ADH) & a] == ADH).mean()), "group_C_named_ADH": float((top[(y == ADH) & ~a] == ADH).mean()),
            "gray_macro_f1": macro, "adh_f1": adh_f1, "udh_named_UDH": float((top[y == UDH] == UDH).mean()), "fea_named_FEA": float((top[y == FEA] == FEA).mean()),
            "dcis_named_DCIS": float((top[y == DCIS] == DCIS).mean()),
            "udh_against_group_A_auc": float(roc_auc_score((y[m] == ADH), np.log(prob[m, ADH] / prob[m, UDH]))) if 0 < (y[m] == ADH).sum() < m.sum() else float("nan")}


def main():
    lab = slide_labels()
    saved, y, patient, folds, _, vy, vpatient = sl.load()
    train, held = rows("train"), rows("val")
    x, vx = sl.engine_features("train"), sl.engine_features("val")
    grp_a = np.array([SEVERITY[lab[r["slide"]]] == ADH for r in train]) & (y == ADH)
    vgrp_a = np.array([SEVERITY[lab[r["slide"]]] == ADH for r in held]) & (vy == ADH)
    rng = np.random.default_rng(0)
    draws = patient_draws(patient, rng, 1000)
    draws = [i for i in draws if ((y[i] == ADH) & grp_a[i]).any() and ((y[i] == ADH) & ~grp_a[i]).any()]
    result = {"C": C, "regions": {"group_A": int(grp_a.sum()), "group_C": int(((y == ADH) & ~grp_a).sum())}, "train": {}, "val": {}}
    probs = {}
    for kind in ("V0", "V1", "V2"):
        p = np.zeros((len(y), 7))
        for a, b in folds:
            p[b] = fit_predict(kind, x[a], y[a], grp_a[a], x[b])
        probs[kind] = p
        result["train"][kind] = {k: round(v, 4) for k, v in measures(p, y, grp_a).items()}
        result["val"][kind] = {k: round(v, 4) for k, v in measures(fit_predict(kind, x, y, grp_a, vx), vy, vgrp_a).items()}
        print(kind, result["train"][kind], flush=True)
    result["V0_reproduces_saved"] = bool(np.allclose(probs["V0"], saved, atol=1e-6))
    for kind in ("V1", "V2"):
        d = {k: [] for k in ("group_A_named_ADH", "gray_macro_f1", "udh_named_UDH", "fea_named_FEA", "adh_f1")}
        for i in draws:
            m1, m0 = measures(probs[kind], y, grp_a, i), measures(probs["V0"], y, grp_a, i)
            for k in d:
                d[k].append(m1[k] - m0[k])
        r = {k: {"difference": round(result["train"][kind][k] - result["train"]["V0"][k], 4), "range95": range95(v)} for k, v in d.items()}
        up = r["group_A_named_ADH"]["range95"][0] > 0
        r["verdict"] = "adopted" if r["gray_macro_f1"]["range95"][0] > 0 else ("a trade" if up and r["gray_macro_f1"]["range95"][0] <= 0 <= r["gray_macro_f1"]["range95"][1]
                                                                         else ("failed" if not up else "group A up, gray zone down"))
        result[f"{kind}_minus_V0"] = r
    (OUT / "adh_kinds.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "train"}, indent=1))


if __name__ == "__main__":
    main()
