#!/usr/bin/env python3
"""Ledger entry 56: does the classifier read unseen patients better when the suspect labels of the training patients are
removed or softened?

Usage (from the project folder):
  python -I tools/grayzone_label_noise.py   # -> outputs/grayzone/label_noise.json

V0 the adopted classifier; V1 suspects left out; V2 suspects count 0.7 for the given type and 0.3 for the type the model
prefers; V3 as V2, with the ADH suspects taken inside each of the two kinds of ADH. Suspects are chosen from out-of-fold
probabilities of the training part only (nested), so no held-out label helps to choose.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import folds as foldlib
import grayzone_adh_kinds as ak
import short_lists as sl
from grayzone_slide_origin import SEVERITY, slide_labels
from train_classifier import ROOT, classifier, patient_draws, range95, rows

OUT = ROOT / "outputs/grayzone"
ADH, C, SHARE = 4, sl.ENGINE["C"], 0.10


def suspects(prob, y, grp_a, capped):
    """Per class, the SHARE of its regions with the largest margin (best other type minus given type), where the margin is positive.
    With capped=True the ADH regions are ranked inside group A and inside group C separately."""
    other = prob.copy()
    other[np.arange(len(y)), y] = -1
    margin, rival = other.max(1) - prob[np.arange(len(y)), y], other.argmax(1)
    out = np.zeros(len(y), bool)
    groups = [y == c for c in np.unique(y) if not (capped and c == ADH)] + ([(y == ADH) & grp_a, (y == ADH) & ~grp_a] if capped else [])
    for g in groups:
        idx = np.flatnonzero(g)
        take = idx[np.argsort(-margin[idx])][:int(round(SHARE * len(idx)))]
        out[take[margin[take] > 0]] = True
    return out, rival


def fit_predict(kind, xa, ya, pa, ga, inner_prob, xb):
    if kind == "V0":
        return classifier(C).fit(xa, ya).predict_proba(xb), None
    s, rival = suspects(inner_prob, ya, ga, capped=kind == "V3")
    if kind == "V1":
        return classifier(C).fit(xa[~s], ya[~s]).predict_proba(xb), s
    x2, y2 = np.vstack([xa, xa[s]]), np.concatenate([ya, rival[s]])       # a suspect appears twice: once per type it may be
    w = np.concatenate([np.where(s, 0.7, 1.0), np.full(s.sum(), 0.3)])
    return classifier(C).fit(x2, y2, logisticregression__sample_weight=w).predict_proba(xb), s


def inner_oof(x, y, patient):
    p = np.zeros((len(y), 7))
    for a, b in foldlib.split(4, y, patient):
        p[b] = classifier(C).fit(x[a], y[a]).predict_proba(x[b])
    return p


def main():
    lab = slide_labels()
    saved, y, patient, folds, _, vy, vpatient = sl.load()
    train, held = rows("train"), rows("val")
    x, vx = sl.engine_features("train"), sl.engine_features("val")
    grp_a = np.array([SEVERITY[lab[r["slide"]]] == ADH for r in train]) & (y == ADH)
    vgrp_a = np.array([SEVERITY[lab[r["slide"]]] == ADH for r in held]) & (vy == ADH)
    kinds = ("V0", "V1", "V2", "V3")
    probs = {k: np.zeros((len(y), 7)) for k in kinds}
    for n, (a, b) in enumerate(folds):
        inner = inner_oof(x[a], y[a], patient[a])
        for k in kinds:
            probs[k][b], _ = fit_predict(k, x[a], y[a], patient[a], grp_a[a], inner, x[b])
        print("outer fold", n + 1, "done", flush=True)
    result = {"C": C, "suspect_share_per_class": SHARE, "V0_reproduces_saved": bool(np.allclose(probs["V0"], saved, atol=1e-6)), "train": {}, "val": {}}
    # the suspects when all training patients are used (for the validation models and for the description)
    wrong146 = np.isin(y, [2, 4]) & ((np.log(saved[:, 4] / saved[:, 2]) > 0) != (y == 4))
    for k in kinds:
        vp, s = fit_predict(k, x, y, patient, grp_a, saved, vx)
        result["train"][k] = {m: round(v, 4) for m, v in ak.measures(probs[k], y, grp_a).items()}
        result["val"][k] = {m: round(v, 4) for m, v in ak.measures(vp, vy, vgrp_a).items()}
        if s is not None:
            result.setdefault("suspects_all_training", {})[k] = {
                "regions": int(s.sum()), "by_class": {str(c): int((s & (y == c)).sum()) for c in range(7)},
                "ADH_group_A": f"{int((s & grp_a).sum())} of {int(grp_a.sum())}", "ADH_group_C": f"{int((s & (y == ADH) & ~grp_a).sum())} of {int(((y == ADH) & ~grp_a).sum())}",
                "of_the_146_wrong_UDH_ADH_regions": f"{int((s & wrong146).sum())} of {int(wrong146.sum())}"}
        print(k, result["train"][k], flush=True)
    rng = np.random.default_rng(0)
    draws = [i for i in patient_draws(patient, rng, 1000) if ((y[i] == ADH) & grp_a[i]).any() and ((y[i] == ADH) & ~grp_a[i]).any()]
    for k in kinds[1:]:
        d = {m: [] for m in ("gray_macro_f1", "adh_f1", "group_A_named_ADH", "udh_named_UDH")}
        for i in draws:
            m1, m0 = ak.measures(probs[k], y, grp_a, i), ak.measures(probs["V0"], y, grp_a, i)
            for m in d:
                d[m].append(m1[m] - m0[m])
        r = {m: {"difference": round(result["train"][k][m] - result["train"]["V0"][m], 4), "range95": range95(v)} for m, v in d.items()}
        gave_up = r["group_A_named_ADH"]["range95"][1] < 0
        r["verdict"] = ("bought by giving up the hard ADH" if gave_up else "adopted" if r["gray_macro_f1"]["range95"][0] > 0
                        else "a lead" if r["adh_f1"]["range95"][0] > 0 else "failed")
        result[f"{k}_minus_V0"] = r
    (OUT / "label_noise.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "train"}, indent=1))


if __name__ == "__main__":
    main()
