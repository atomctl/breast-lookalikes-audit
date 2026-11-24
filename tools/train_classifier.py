#!/usr/bin/env python3
"""M1: a small classifier on frozen features, compared across the 4 feature models.

A region's features are the average of its tiles' features. The classifier is one linear layer
(class-weighted logistic regression). The test set is never read: this script has no code path for it.

Usage (from the project folder):
  python -I tools/train_classifier.py cv    # grouped 5-fold CV on the training patients -> outputs/m1/cv.json
  python -I tools/train_classifier.py val   # fit on train with the CV choice, score validation once -> outputs/m1/val.json

The rule for this experiment is entry 3 in LEDGER.md and was written before the first run.
"""
import csv
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import folds as foldlib  # the patient folds, frozen in outputs/folds.json (tools/folds.py)
FEATS = ROOT / "data/features"
OUT = ROOT / "outputs/m1"
MODELS = {"resnet50": "general", "vit_l_imagenet": "general", "uni": "tissue", "hoptimus0": "tissue"}
FILTERS = [0.0, 0.5]          # minimum share of tissue a tile needs to count
STRENGTHS = [0.0001, 0.001, 0.003, 0.01, 0.1, 1, 10]  # C: smaller = stronger pull towards small weights (ledger entry 4 extended it downward)
TYPES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]


def rows(split):
    assert split in ("train", "val")  # the test set stays locked
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def pooled(model, split, min_tissue):
    """One row per region: the average of its tile features. Cached, because loading takes a while."""
    cache = OUT / "pooled" / f"{model}_{split}_{min_tissue}.npy"
    if cache.exists():
        return np.load(cache)
    x = []
    for r in rows(split):
        z = np.load(FEATS / model / (r["file"][:-4] + ".npz"))
        feats, keep = z["feats"].astype(np.float32), z["tissue"] >= min_tissue
        x.append(feats[keep].mean(0) if keep.any() else feats.mean(0))  # a region with no tile above the bar keeps all its tiles
    x = np.array(x)
    assert np.isfinite(x).all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return x


def classifier(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))


def weighted_f1(y, p):
    return f1_score(y, p, average="weighted")


def patient_draws(patient, rng, n=2000):
    """Row indices for n resamples of whole patients."""
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    return [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(n)]


def range95(values):
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


def cv():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    folds = list(foldlib.split(5, y, patient))
    assert all(not set(patient[a]) & set(patient[b]) for a, b in folds)  # no patient on both sides of a fold
    result, best_pred = {"grid": {}}, {}
    for model in MODELS:
        best = None
        for min_tissue in FILTERS:
            x = pooled(model, "train", min_tissue)
            for c in STRENGTHS:
                pred = np.empty_like(y)
                for a, b in folds:
                    pred[b] = classifier(c).fit(x[a], y[a]).predict(x[b])
                score = weighted_f1(y, pred)
                result["grid"][f"{model} tissue>={min_tissue} C={c}"] = round(score, 4)
                print(f"{model} tissue>={min_tissue} C={c}: {score:.4f}", flush=True)
                if best is None or score > best[0]:
                    best, best_pred[model] = (score, min_tissue, c), pred
        result[model] = {"kind": MODELS[model], "weighted_f1": round(best[0], 4), "min_tissue": best[1], "C": best[2]}
    draws = patient_draws(patient, np.random.default_rng(0))
    for model in MODELS:
        result[model]["range95"] = range95([weighted_f1(y[i], best_pred[model][i]) for i in draws])
    top = {k: max((m for m in MODELS if MODELS[m] == k), key=lambda m: result[m]["weighted_f1"]) for k in ("tissue", "general")}
    diff = [weighted_f1(y[i], best_pred[top["tissue"]][i]) - weighted_f1(y[i], best_pred[top["general"]][i]) for i in draws]
    result["comparison"] = {"best_tissue": top["tissue"], "best_general": top["general"],
                            "difference": round(result[top["tissue"]]["weighted_f1"] - result[top["general"]]["weighted_f1"], 4),
                            "range95": range95(diff), "tissue_models_win": bool(np.percentile(diff, 2.5) > 0)}
    result["patients"], result["regions"] = int(len(np.unique(patient))), len(y)
    (OUT / "cv.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "grid"}, indent=2))


def val():
    import joblib
    choice = json.loads((OUT / "cv.json").read_text())
    train, held = rows("train"), rows("val")
    y = np.array([int(r["label"]) for r in train])
    yv = np.array([int(r["label"]) for r in held])
    patient = np.array([int(r["patient"]) for r in held])
    draws = patient_draws(patient, np.random.default_rng(0))
    result = {"patients": int(len(np.unique(patient))), "regions": len(yv)}
    for model in MODELS:
        s = choice[model]
        clf = classifier(s["C"]).fit(pooled(model, "train", s["min_tissue"]), y)
        prob = clf.predict_proba(pooled(model, "val", s["min_tissue"]))
        pred = prob.argmax(1)
        joblib.dump(clf, OUT / f"classifier_{model}.joblib")
        np.savez(OUT / f"val_probs_{model}.npz", prob=prob, label=yv, patient=patient, region=[r["file"] for r in held])
        result[model] = {"weighted_f1": round(weighted_f1(yv, pred), 4),
                         "range95": range95([weighted_f1(yv[i], pred[i]) for i in draws]),
                         "macro_f1": round(f1_score(yv, pred, average="macro"), 4),
                         "f1_by_type": dict(zip(TYPES, (round(float(v), 3) for v in f1_score(yv, pred, average=None)))),
                         "confusion_rows_true_cols_predicted": confusion_matrix(yv, pred).tolist()}
    (OUT / "val.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"cv": cv, "val": val}[sys.argv[1]]()
