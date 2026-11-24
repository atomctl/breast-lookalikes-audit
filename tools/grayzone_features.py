#!/usr/bin/env python3
"""Ledger entry 37: do other tissue-model features move the gray zone? UNI against a candidate and against both joined.

Usage (from the project folder):
  python -I tools/grayzone_features.py midnight   # needs data/features/midnight -> outputs/grayzone/features_midnight.json

The strength C is chosen inside each fold (inner patient-grouped folds, by gray-zone macro F1), so no held-out patient helps choose it.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import folds as foldlib
import short_lists as sl
from train_classifier import STRENGTHS, classifier, patient_draws, pooled, range95, rows, weighted_f1

GRAY = [2, 3, 4, 5]
PAIRS = {"UDH-ADH": (2, 4), "FEA-ADH": (3, 4), "ADH-DCIS": (4, 5)}


def gray_f1(prob, y, i=slice(None)):
    prob, y = prob[i], y[i]
    g = np.isin(y, GRAY)
    f = f1_score(np.searchsorted(GRAY, y[g]), prob[g][:, GRAY].argmax(1), average=None, labels=range(4))
    return float(f.mean()), float(f[2])


def out_of_fold(x, y, patient, folds):
    prob, chosen = np.zeros((len(y), 7)), []
    for a, b in folds:
        inner = list(foldlib.split(4, y[a], patient[a]))
        best = None
        for c in STRENGTHS:
            p = np.zeros((len(a), 7))
            for ia, ib in inner:
                p[ib] = classifier(c).fit(x[a][ia], y[a][ia]).predict_proba(x[a][ib])
            s = gray_f1(p, y[a])[0]
            if best is None or s > best[0]:
                best = (s, c)
        chosen.append(best[1])
        prob[b] = classifier(best[1]).fit(x[a], y[a]).predict_proba(x[b])
    return prob, chosen


def main():
    cand = sys.argv[1]
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    a, b = pooled("uni", "train", 0.5), pooled(cand, "train", 0.5)
    sets = {"A uni": a, f"B {cand}": b, f"C uni + {cand}": np.hstack([a, b])}
    draws = patient_draws(patient, np.random.default_rng(0))
    result, probs = {"patients": int(len(np.unique(patient))), "regions": len(y), "candidate": cand}, {}
    for name, x in sets.items():
        prob, chosen = out_of_fold(x, y, patient, folds)
        probs[name] = prob
        macro, adh = gray_f1(prob, y)
        r = {"numbers_per_region": int(x.shape[1]), "C_per_fold": chosen, "gray_macro_f1": round(macro, 4),
             "gray_macro_f1_range95": range95([gray_f1(prob, y, i)[0] for i in draws]), "adh_f1": round(adh, 4),
             "weighted_f1_7_types": round(weighted_f1(y, prob.argmax(1)), 4), "pair_auc": {}}
        for pn, (u, v) in PAIRS.items():
            m = np.isin(y, [u, v])
            r["pair_auc"][pn] = round(float(roc_auc_score(y[m] == v, prob[m, v] / (prob[m, u] + prob[m, v]))), 3)
        result[name] = r
        print(name, r, flush=True)
    base = "A uni"
    for name in sets:
        if name != base:
            d = [gray_f1(probs[name], y, i)[0] - gray_f1(probs[base], y, i)[0] for i in draws]
            e = [gray_f1(probs[name], y, i)[1] - gray_f1(probs[base], y, i)[1] for i in draws]
            result[name].update({"gray_macro_f1_minus_uni": round(result[name]["gray_macro_f1"] - result[base]["gray_macro_f1"], 4), "range95_of_difference": range95(d),
                                 "adh_f1_minus_uni": round(result[name]["adh_f1"] - result[base]["adh_f1"], 4), "adh_range95_of_difference": range95(e),
                                 "adopted": bool(np.percentile(d, 2.5) > 0)})
            np.save(sl.ROOT / f"outputs/grayzone/oof_{name.replace(' ', '_').replace('+', 'plus')}.npy", probs[name])
    (sl.ROOT / f"outputs/grayzone/features_{cand}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != base and isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
