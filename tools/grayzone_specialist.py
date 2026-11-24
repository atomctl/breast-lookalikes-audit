#!/usr/bin/env python3
"""Ledger entry 24: a specialist for the four gray-zone types, with lesion size and slide context.

Usage (from the project folder):
  python -I tools/grayzone_specialist.py   # -> outputs/grayzone/specialist.json

Cross-validation on the 106 training patients, then one look at the 15 validation patients. No test data.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import folds as foldlib
import short_lists as sl
from train_classifier import FEATS, classifier, patient_draws, range95, rows

OUT = sl.ROOT / "outputs/grayzone"
GRAY = [2, 3, 4, 5]  # UDH, FEA, ADH, DCIS
NAMES = [sl.TYPES[k] for k in GRAY]
PAIRS = [(0, 2), (1, 2), (2, 3)]  # positions in GRAY: UDH-ADH, FEA-ADH, ADH-DCIS
STRENGTHS = [0.0003, 0.001, 0.003, 0.01]


def size_features(held):
    out = []
    for r in held:
        z = np.load(FEATS / "uni" / (r["file"][:-4] + ".npz"))
        w, h = z["size"]
        out.append([np.log(w * h * 0.25 / 1e6), np.log(w), np.log(h), np.log(max(int((z["tissue"] >= 0.5).sum()), 1))])
    return np.array(out)


def slide_context(held, prob):
    """What the current classifier makes of the OTHER regions on the same slide. Uses no labels."""
    slide = np.array([r["slide"] for r in held])
    out = np.zeros((len(held), 16))
    for s in np.unique(slide):
        idx = np.flatnonzero(slide == s)
        for i in idx:
            other = idx[idx != i]
            if len(other):
                out[i] = [*prob[other].mean(0), *prob[other].max(0), np.log1p(len(other)), 1.0]
    return out


def scores(p4, y4):
    """p4: probabilities over the four gray-zone types; y4: position of the agreed type in GRAY."""
    pred = p4.argmax(1)
    out = {"macro_f1": round(float(f1_score(y4, pred, average="macro")), 4),
           "f1_by_type": dict(zip(NAMES, (round(float(v), 3) for v in f1_score(y4, pred, average=None, labels=range(4))))),
           "dcis_read_as_adh": int(((y4 == 3) & (pred == 2)).sum()), "adh_read_as_dcis": int(((y4 == 2) & (pred == 3)).sum())}
    for a, b in PAIRS:
        m = (y4 == a) | (y4 == b)
        out[f"auc_{NAMES[a]}_{NAMES[b]}"] = round(float(roc_auc_score(y4[m] == a, np.log(p4[m, a] + 1e-9) - np.log(p4[m, b] + 1e-9))), 3)
    return out


def main():
    train, held = rows("train"), rows("val")
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    x, xv = sl.engine_features("train"), sl.engine_features("val")
    blocks = {"size": (size_features(train), size_features(held)), "context": (slide_context(train, prob), slide_context(held, vprob))}
    gray, grayv = np.isin(y, GRAY), np.isin(vy, GRAY)
    y4 = np.searchsorted(GRAY, y[gray])
    y4v = np.searchsorted(GRAY, vy[grayv])
    variants = {"B1 same features": [], "S + size": ["size"], "X + slide context": ["context"], "SX + both": ["size", "context"],
                "audit: size alone": ["size!"], "audit: context alone": ["context!"]}

    def design(parts, which):
        base = (x, xv)[which]
        if parts and parts[0].endswith("!"):
            return blocks[parts[0][:-1]][which]
        return np.hstack([base] + [blocks[p][which] for p in parts])

    b0 = prob[gray][:, GRAY] / prob[gray][:, GRAY].sum(1, keepdims=True)
    b0v = vprob[grayv][:, GRAY] / vprob[grayv][:, GRAY].sum(1, keepdims=True)
    draws = patient_draws(patient[gray], np.random.default_rng(0), 1000)
    macro = lambda p, i=slice(None): f1_score(y4[i], p[i].argmax(1), average="macro")
    result = {"gray_zone_regions_train": int(gray.sum()), "patients_train": int(len(np.unique(patient[gray]))),
              "gray_zone_regions_val": int(grayv.sum()), "cross_validation": {"B0 current classifier": scores(b0, y4)}, "validation": {"B0 current classifier": scores(b0v, y4v)}}
    print("B0", result["cross_validation"]["B0 current classifier"], flush=True)
    pos = np.cumsum(gray) - 1  # row of a gray-zone region inside the gray-only arrays
    for name, parts in variants.items():
        d = design(parts, 0)
        best = None
        for c in STRENGTHS:
            p4 = np.zeros((int(gray.sum()), 4))
            for a, b in folds:
                a, b = a[gray[a]], b[gray[b]]
                p4[pos[b]] = classifier(c).fit(d[a], np.searchsorted(GRAY, y[a])).predict_proba(d[b])
            if best is None or macro(p4) > best[0]:
                best = (macro(p4), c, p4)
        _, c, p4 = best
        diff = [macro(p4, i) - macro(b0, i) for i in draws]
        entry = {"C": c, **scores(p4, y4), "minus_B0": round(macro(p4) - macro(b0), 4), "range95": range95(diff), "adopted": bool(np.percentile(diff, 2.5) > 0)}
        result["cross_validation"][name] = entry
        print(name, entry, flush=True)
        clf = classifier(c).fit(d[gray], y4)
        result["validation"][name] = scores(clf.predict_proba(design(parts, 1)[grayv]), y4v)
    adopted = [k for k, v in result["cross_validation"].items() if v.get("adopted") and not k.startswith("audit")]
    result["winner"] = max(adopted, key=lambda k: result["cross_validation"][k]["macro_f1"]) if adopted else "B0 current classifier"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "specialist.json").write_text(json.dumps(result, indent=2))
    print("winner:", result["winner"])
    print("validation:", json.dumps(result["validation"], indent=1))


def stack():
    """Ledger entry 25: a second small layer combines the specialist's four scores with size and slide context."""
    train = rows("train")
    prob, y, patient, folds, *_ = sl.load()
    x = sl.engine_features("train")
    extra = {"size": size_features(train), "context": slide_context(train, prob)}
    gray = np.isin(y, GRAY)
    xg, pg, y4 = x[gray], patient[gray], np.searchsorted(GRAY, y[gray])
    pos = np.cumsum(gray) - 1
    variants = {"T0 scores only": [], "TS + size": ["size"], "TX + slide context": ["context"], "TSX + both": ["size", "context"]}
    out = {k: np.zeros((len(y4), 4)) for k in variants}
    for a, b in folds:
        a, b = pos[a[gray[a]]], pos[b[gray[b]]]
        inner = np.zeros((len(a), 4))  # stage-one scores for the training regions, each from a model that never saw its patient
        for ia, ib in foldlib.split(4, y4[a], pg[a]):
            inner[ib] = classifier(0.001).fit(xg[a][ia], y4[a][ia]).predict_proba(xg[a][ib])
        held = classifier(0.001).fit(xg[a], y4[a]).predict_proba(xg[b])
        for name, parts in variants.items():
            za = np.hstack([np.log(inner + 1e-6)] + [extra[p][gray][a] for p in parts])
            zb = np.hstack([np.log(held + 1e-6)] + [extra[p][gray][b] for p in parts])
            out[name][b] = classifier(1.0).fit(za, y4[a]).predict_proba(zb)
        print("fold done", flush=True)
    b0 = prob[gray][:, GRAY] / prob[gray][:, GRAY].sum(1, keepdims=True)
    draws = patient_draws(pg, np.random.default_rng(0), 1000)
    macro = lambda p, i=slice(None): f1_score(y4[i], p[i].argmax(1), average="macro")
    result = {"B0 current classifier": scores(b0, y4)}
    for name, p4 in out.items():
        diff = [macro(p4, i) - macro(b0, i) for i in draws]
        result[name] = {**scores(p4, y4), "minus_B0": round(macro(p4) - macro(b0), 4), "range95": range95(diff), "adopted": bool(np.percentile(diff, 2.5) > 0)}
        print(name, result[name], flush=True)
    adopted = [k for k, v in result.items() if v.get("adopted")]
    result["winner"] = max(adopted, key=lambda k: result[k]["macro_f1"]) if adopted else "B0 current classifier"
    (OUT / "specialist_stack.json").write_text(json.dumps(result, indent=2))
    print("winner:", result["winner"])


if __name__ == "__main__":
    stack() if "stack" in sys.argv else main()
