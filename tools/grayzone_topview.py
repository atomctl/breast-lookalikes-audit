#!/usr/bin/env python3
"""Ledger entry 35: three checks on the gray zone. Are the labels the limit, is the data the limit, is the system safe there?

Usage (from the project folder):
  python -I tools/grayzone_topview.py   # -> outputs/grayzone/topview.json
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl
from train_classifier import classifier

GRAY = [2, 3, 4, 5]  # UDH, FEA, ADH, DCIS
T = sl.TYPES


def gray_scores(prob, y):
    g = np.isin(y, GRAY)
    p4 = prob[g][:, GRAY].argmax(1)
    y4 = np.searchsorted(GRAY, y[g])
    f = f1_score(y4, p4, average=None, labels=range(4))
    return float(f.mean()), float(f[2])


def errors(prob, y):
    g = np.isin(y, GRAY)
    p, yy = prob[g], y[g]
    top = p.argmax(1)
    second = np.argsort(-p, axis=1)[:, 1]
    wrong = top != yy
    conf = p.max(1) >= 0.7
    return {"gray_regions": int(g.sum()), "wrong": int(wrong.sum()), "wrong_share": round(float(wrong.mean()), 3),
            "wrong_with_top_probability_0.7_or_more": round(float(conf[wrong].mean()), 3),
            "right_with_top_probability_0.7_or_more": round(float(conf[~wrong].mean()), 3),
            "wrong_where_agreed_type_is_runner_up": round(float((second[wrong] == yy[wrong]).mean()), 3),
            "confident_wrong_regions": int((wrong & conf).sum())}


def safety(prob, y, patient):
    act, lists = pp.apply_system(prob, patient)
    top = prob.argmax(1)
    out = {}
    for name, m in [(T[k], y == k) for k in range(7)] + [("gray zone", np.isin(y, GRAY)), ("all", np.ones(len(y), bool))]:
        alone, asked = act & m, ~act & m
        hit = lists[np.arange(len(y)), y].astype(bool)
        out[name] = {"regions": int(m.sum()), "asked_share": round(float(asked.sum() / max(m.sum(), 1)), 3),
                     "answered_alone": int(alone.sum()), "answered_alone_wrong": int((alone & (top != y)).sum()),
                     "error_among_answered_alone": round(float((alone & (top != y)).sum() / max(alone.sum(), 1)), 3),
                     "list_holds_agreed_type_among_asked": round(float((asked & hit).sum() / max(asked.sum(), 1)), 3)}
    for k in (4, 5):
        bad = act & (y == k) & (top != y)
        out[T[k]]["wrong_lone_answers_called"] = {T[j]: int((top[bad] == j).sum()) for j in range(7) if (top[bad] == j).any()}
    return out


def main():
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    x = sl.engine_features("train")
    result = {"1_errors_training_patients": errors(prob, y), "1_errors_validation_patients": errors(vprob, vy)}
    rng = np.random.default_rng(0)
    curve = {}
    for share in (0.25, 0.5, 0.75, 1.0):
        runs = []
        for draw in range(1 if share == 1.0 else 5):
            p = np.zeros_like(prob)
            for a, b in folds:
                ids = np.unique(patient[a])
                keep = rng.choice(ids, max(int(round(share * len(ids))), 7), replace=False) if share < 1 else ids
                a2 = a[np.isin(patient[a], keep)]
                if len(np.unique(y[a2])) < 7:
                    a2 = a
                m = classifier(sl.ENGINE["C"]).fit(x[a2], y[a2])
                p[b] = m.predict_proba(x[b])
            runs.append(gray_scores(p, y))
        r = np.array(runs)
        curve[str(share)] = {"training_patients_about": int(round(share * len(np.unique(patient)) * 0.8)), "gray_macro_f1": round(float(r[:, 0].mean()), 4),
                             "gray_macro_f1_min_max": [round(float(r[:, 0].min()), 4), round(float(r[:, 0].max()), 4)], "adh_f1": round(float(r[:, 1].mean()), 4)}
        print(share, curve[str(share)], flush=True)
    result["2_learning_curve"] = curve
    result["2_gain_75_to_100"] = round(curve["1.0"]["gray_macro_f1"] - curve["0.75"]["gray_macro_f1"], 4)
    result["2_gain_50_to_100"] = round(curve["1.0"]["gray_macro_f1"] - curve["0.5"]["gray_macro_f1"], 4)
    result["3_safety_training_patients"] = safety(prob, y, patient)
    result["3_safety_validation_patients"] = safety(vprob, vy, vpatient)
    (sl.ROOT / "outputs/grayzone/topview.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("2_learning")}, indent=1))


if __name__ == "__main__":
    main()
