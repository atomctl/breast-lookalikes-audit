#!/usr/bin/env python3
"""Ledger entry 18: do look-alike cases from other patients carry correct information, and in which feature space?

Usage (from the project folder):
  python -I tools/evidence_lookalikes.py   # -> outputs/evidence/lookalikes.json

Training and validation patients only. Every query is embedded by a classifier that never saw its patient.
"""
import json
import pathlib
import sys

import joblib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl
from train_classifier import classifier, patient_draws, pooled, range95, rows

OUT = sl.ROOT / "outputs/evidence"
K_SHOW, K_TYPE = 5, 3


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def spaces(split):
    tiles, view = pooled("uni", split, 0.5), np.load(sl.ROOT / "outputs/m1/thunder" / f"uni_{split}.npy")
    return {"S1 UNI tile average": tiles, "S2 UNI whole-region view": view, "S5 H-optimus-0 tile average": pooled("hoptimus0", split, 0.0)}, np.hstack([tiles, view])


def similarity(name, q, lib):
    """Similarity of every query to every library case (higher = more alike)."""
    if name.startswith("S4"):
        return -np.sqrt(((q[:, None, :] - lib[None, :, :]) ** 2).sum(2))
    return unit(q) @ unit(lib).T


def evidence(sim, lib_y):
    """Per query and type: the mean similarity to the K_TYPE nearest library cases of that type."""
    out = np.zeros((sim.shape[0], 7))
    for k in range(7):
        s = sim[:, lib_y == k]
        out[:, k] = np.sort(s, axis=1)[:, -K_TYPE:].mean(1)
    return out


def main():
    train, held = rows("train"), rows("val")
    y_t = np.array([int(r["label"]) for r in train])
    y_v = np.array([int(r["label"]) for r in held])
    prob, y, patient, folds, vprob, _, vpatient = sl.load()
    sp_t, x_t = spaces("train")
    sp_v, x_v = spaces("val")
    names = [*sp_t, "S3 both, standardised", "S4 the classifier's 7 scores"]
    n = len(y_t) + len(y_v)
    top5 = {s: np.zeros((n, K_SHOW), int) for s in names}   # labels of the 5 nearest
    ev = {s: np.zeros((n, 7)) for s in names}               # look-alike evidence per type
    vote = {s: np.zeros(n, int) for s in names}
    random_share = np.zeros(n)

    def run(q_idx, lib_idx, model, xq, xl, spq, spl, yq, yl, offset):
        scaler, clf = model[0], model[1]
        q = {**{s: spq[s][q_idx] for s in spq}, "S3 both, standardised": scaler.transform(xq[q_idx]),
             "S4 the classifier's 7 scores": model.decision_function(xq[q_idx])}
        lib = {**{s: spl[s][lib_idx] for s in spl}, "S3 both, standardised": scaler.transform(xl[lib_idx]),
               "S4 the classifier's 7 scores": model.decision_function(xl[lib_idx])}
        ly = yl[lib_idx]
        rows_ = offset + q_idx
        random_share[rows_] = np.bincount(ly, minlength=7)[yq[q_idx]] / len(ly)
        for s in names:
            sim = similarity(s, q[s], lib[s])
            order = np.argsort(-sim, axis=1)[:, :10]
            top5[s][rows_] = ly[order[:, :K_SHOW]]
            w = np.take_along_axis(sim, order, axis=1)
            w = w - w.min(1, keepdims=True) + 1e-9
            vote[s][rows_] = np.array([np.bincount(ly[o], weights=ww, minlength=7).argmax() for o, ww in zip(order, w)])
            ev[s][rows_] = evidence(sim, ly)

    for a, b in folds:
        run(b, a, classifier(sl.ENGINE["C"]).fit(x_t[a], y_t[a]), x_t, x_t, sp_t, sp_t, y_t, y_t, 0)
        print("fold done", flush=True)
    final = joblib.load(sl.ROOT / "outputs/m2/engine.joblib")
    run(np.arange(len(y_v)), np.arange(len(y_t)), final, x_v, x_t, sp_v, sp_t, y_v, y_t, len(y_t))

    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, y_v]), np.concatenate([patient, vpatient])
    act, lists = pp.apply_system(prob, patient)
    asked = ~act
    draws = patient_draws(patient, np.random.default_rng(0), 1000)
    rate = lambda v, m, i=slice(None): float(v[i][m[i]].mean())

    def first_right(score):
        """Is the type ranked first among those on the short list the right one?"""
        return (np.where(lists, score, -np.inf).argmax(1) == y)

    def ranks(score):
        return np.where(lists, score, -np.inf).argsort(1).argsort(1).astype(float)  # higher = better

    by_classifier = first_right(prob)
    result = {"asked_cases": int(asked.sum()), "random_share_among_asked": round(float(random_share[asked].mean()), 4),
              "classifier_first_on_list_right_among_asked": round(rate(by_classifier, asked), 4), "spaces": {}}
    for s in names:
        p5 = (top5[s] == y[:, None]).mean(1)
        by_ev = first_right(ev[s])
        both = first_right(ranks(ev[s]) + ranks(prob) + 1e-3 * ranks(prob))
        gain = [rate(both, asked, i) - rate(by_classifier, asked, i) for i in draws]
        result["spaces"][s] = {
            "precision_at_5_among_asked": round(rate(p5, asked), 4), "precision_at_5_range95": range95([rate(p5, asked, i) for i in draws]),
            "precision_at_5_all_cases": round(float(p5.mean()), 4),
            "precision_at_5_among_asked_by_type": {t: round(float(p5[asked & (y == k)].mean()), 3) for k, t in enumerate(sl.TYPES)},
            "first_on_list_right_by_lookalikes": round(rate(by_ev, asked), 4), "first_on_list_right_by_both": round(rate(both, asked), 4),
            "both_minus_classifier": round(rate(both, asked) - rate(by_classifier, asked), 4), "both_minus_classifier_range95": range95(gain),
            "neighbour_vote_accuracy_among_asked": round(rate(vote[s] == y, asked), 4)}
        print(s, result["spaces"][s], flush=True)
    result["classifier_accuracy_among_asked"] = round(rate(prob.argmax(1) == y, asked), 4)
    best = max(names, key=lambda s: result["spaces"][s]["precision_at_5_among_asked"])
    b = result["spaces"][best]
    result["chosen_space"] = best
    result["shown_as_evidence"] = bool(b["precision_at_5_among_asked"] >= 2 * result["random_share_among_asked"])
    result["used_to_order_the_list"] = bool(result["shown_as_evidence"] and b["both_minus_classifier_range95"][0] > 0)
    # self-check: a query is its own nearest case when it is put into the library
    s0 = similarity("S1", sp_t["S1 UNI tile average"][:50], sp_t["S1 UNI tile average"][:200])
    assert (s0.argmax(1) == np.arange(50)).all()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lookalikes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "spaces"}, indent=2))


if __name__ == "__main__":
    main()
