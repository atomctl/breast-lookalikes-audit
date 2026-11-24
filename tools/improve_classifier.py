#!/usr/bin/env python3
"""M1, ledger entry 5: try three ways to improve the classifier, by cross-validation on the training patients.

Usage (from the project folder):
  python -I tools/improve_classifier.py   # -> outputs/m1/improve.json

Validation and test are not read. The rule that decides is entry 5 in LEDGER.md.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
from train_classifier import FEATS, OUT, classifier, patient_draws, pooled, range95, rows, weighted_f1
import folds as foldlib

STRENGTHS = [0.0003, 0.001, 0.003, 0.01, 0.03]
ADH = 4


def pooled_std(model, split, min_tissue):
    """One row per region: the standard deviation of its tile features (zeros for a single tile)."""
    cache = OUT / "pooled" / f"{model}_{split}_{min_tissue}_std.npy"
    if cache.exists():
        return np.load(cache)
    x = []
    for r in rows(split):
        z = np.load(FEATS / model / (r["file"][:-4] + ".npz"))
        feats, keep = z["feats"].astype(np.float32), z["tissue"] >= min_tissue
        x.append((feats[keep] if keep.any() else feats).std(0))
    np.save(cache, np.array(x))
    return np.array(x)


def size_features(split):
    """What a region's size alone says: log width, log height, log tile count (at 20x)."""
    x = []
    for r in rows(split):
        z = np.load(FEATS / "uni" / (r["file"][:-4] + ".npz"))
        x.append([np.log(z["size"][0]), np.log(z["size"][1]), np.log(len(z["coords"]))])
    return np.array(x)


def main():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    folds = list(foldlib.split(5, y, patient))  # the same folds as entries 3 and 4, from outputs/folds.json
    draws = patient_draws(patient, np.random.default_rng(0))

    def oof(x, c):
        """Out-of-fold probabilities: every region is scored by a classifier that never saw its patient."""
        prob = np.zeros((len(y), 7))
        for a, b in folds:
            prob[b] = classifier(c).fit(x[a], y[a]).predict_proba(x[b])
        return prob

    def best(x, name):
        scored = {c: oof(x, c) for c in STRENGTHS}
        c = max(scored, key=lambda k: weighted_f1(y, scored[k].argmax(1)))
        print(name, {k: round(weighted_f1(y, v.argmax(1)), 4) for k, v in scored.items()}, flush=True)
        return c, scored[c]

    uni = pooled("uni", "train", 0.5)
    base = oof(uni, 0.003)
    hopt = oof(pooled("hoptimus0", "train", 0.0), 0.003)
    wide = np.hstack([uni, pooled("uni_10x", "train", 0.5), pooled("uni_5x", "train", 0.5)])
    c2, v2 = best(wide, "V2 wide view")
    c3, v3 = best(np.hstack([uni, pooled_std("uni", "train", 0.5)]), "V3 spread")
    variants = {"V1 UNI + H-optimus-0": (None, (base + hopt) / 2), "V2 UNI 20x+10x+5x": (c2, v2), "V3 UNI average + spread": (c3, v3)}

    def report(prob):
        pred, ref = prob.argmax(1), base.argmax(1)
        diff = [weighted_f1(y[i], pred[i]) - weighted_f1(y[i], ref[i]) for i in draws]
        return {"weighted_f1": round(weighted_f1(y, pred), 4), "adh_f1": round(float(f1_score(y, pred, average=None)[ADH]), 4),
                "minus_baseline": round(weighted_f1(y, pred) - weighted_f1(y, ref), 4), "range95": range95(diff),
                "adopted": bool(np.percentile(diff, 2.5) > 0)}

    result = {"baseline": {"weighted_f1": round(weighted_f1(y, base.argmax(1)), 4),
                           "adh_f1": round(float(f1_score(y, base.argmax(1), average=None)[ADH]), 4)}}
    for name, (c, prob) in variants.items():
        result[name] = {"C": c, **report(prob)}
    if result["V1 UNI + H-optimus-0"]["adopted"] and result["V2 UNI 20x+10x+5x"]["adopted"]:
        result["V4 V1 + V2"] = {"C": c2, **report((v2 + hopt) / 2)}
    adopted = [k for k in result if result[k].get("adopted")]
    result["winner"] = max(adopted, key=lambda k: result[k]["weighted_f1"]) if adopted else "baseline"
    cs, size = best(size_features("train"), "audit: size only")
    result["audit_size_only"] = {"C": cs, "weighted_f1": round(weighted_f1(y, size.argmax(1)), 4),
                                 "chance_weighted_f1": round(float(np.mean([weighted_f1(y, np.random.default_rng(s).permutation(y)) for s in range(20)])), 4)}
    (OUT / "improve.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def whole():
    """Ledger entry 10: the whole region shrunk to 224 px (THUNDER's view), alone and next to the tile average."""
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    folds = list(foldlib.split(5, y, patient))
    draws = patient_draws(patient, np.random.default_rng(0))

    def oof(x, c):
        prob = np.zeros((len(y), 7))
        for a, b in folds:
            prob[b] = classifier(c).fit(x[a], y[a]).predict_proba(x[b])
        return prob

    tiles = pooled("uni", "train", 0.5)
    view = np.load(OUT / "thunder" / "uni_train.npy")
    assert len(view) == len(y)
    base = oof(tiles, 0.003).argmax(1)
    result = {"baseline": {"weighted_f1": round(weighted_f1(y, base), 4), "adh_f1": round(float(f1_score(y, base, average=None)[ADH]), 4)}}
    for name, x in {"W1 whole-region view": view, "W2 tile average + whole-region view": np.hstack([tiles, view])}.items():
        scored = {c: oof(x, c).argmax(1) for c in STRENGTHS}
        print(name, {c: round(weighted_f1(y, p), 4) for c, p in scored.items()}, flush=True)
        c = max(scored, key=lambda k: weighted_f1(y, scored[k]))
        diff = [weighted_f1(y[i], scored[c][i]) - weighted_f1(y[i], base[i]) for i in draws]
        result[name] = {"C": c, "weighted_f1": round(weighted_f1(y, scored[c]), 4),
                        "adh_f1": round(float(f1_score(y, scored[c], average=None)[ADH]), 4),
                        "f1_by_type": [round(float(v), 3) for v in f1_score(y, scored[c], average=None)],
                        "minus_baseline": round(weighted_f1(y, scored[c]) - weighted_f1(y, base), 4), "range95": range95(diff),
                        "adopted": bool(np.percentile(diff, 2.5) > 0)}
    adopted = [k for k in result if result[k].get("adopted")]
    result["winner"] = max(adopted, key=lambda k: result[k]["weighted_f1"]) if adopted else "baseline"
    (OUT / "improve_whole.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    whole() if sys.argv[1:] == ["whole"] else main()
