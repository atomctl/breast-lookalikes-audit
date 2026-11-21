#!/usr/bin/env python3
"""Ledger entry 19: "why A and not B". The exact split of the classifier's preference into tile shares, and a pointing test.

Usage (from the project folder):
  python -I tools/evidence_why.py   # -> outputs/evidence/why.json

Training and validation patients only; tile shares are computed with classifiers that never saw the patient.
"""
import itertools
import json
import pathlib
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import FEATS, classifier, pooled, range95, rows

OUT = sl.ROOT / "outputs/evidence"
D = 1024  # UNI numbers per tile; the classifier's input is [tile average (D), whole-region view (D)]


def tissue_tiles(file):
    z = np.load(FEATS / "uni" / (file[:-4] + ".npz"))
    feats, keep = z["feats"].astype(np.float32), z["tissue"] >= sl.ENGINE["min_tissue"]
    return (feats[keep], z["coords"][keep]) if keep.any() else (feats, z["coords"])


def split_scores(model, feats, view):
    """Per type: (tile shares, whole-view share, baseline). Their sum is the classifier's score for that type."""
    scaler, clf = model[0], model[1]
    w_t, w_v = clf.coef_[:, :D], clf.coef_[:, D:]
    tiles = ((feats - scaler.mean_[:D]) / scaler.scale_[:D]) @ w_t.T / len(feats)   # (tiles, 7)
    whole = ((view - scaler.mean_[D:]) / scaler.scale_[D:]) @ w_v.T                  # (7,)
    return tiles, whole, clf.intercept_


def main():
    train, held = rows("train"), rows("val")
    y_t = np.array([int(r["label"]) for r in train])
    patient_t = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, _, y_v, patient_v = sl.load()
    x_t = sl.engine_features("train")
    x_v = sl.engine_features("val")
    final = joblib.load(sl.ROOT / "outputs/m2/engine.joblib")
    view_t, view_v = x_t[:, D:], x_v[:, D:]

    # (a) exactness, on validation with the final classifier
    worst = 0.0
    for i, r in enumerate(held):
        feats, _ = tissue_tiles(r["file"])
        tiles, whole, base = split_scores(final, feats, view_v[i])
        worst = max(worst, float(np.abs(tiles.sum(0) + whole + base - final.decision_function(x_v[i:i + 1])[0]).max()))
    result = {"exactness_largest_gap": round(worst, 6), "exact": bool(worst < 1e-3)}
    print("largest gap between the split and the classifier's own scores:", worst, flush=True)

    # (b) pointing test: tile shares from classifiers that never saw the patient
    tile_scores, tile_label, tile_patient, region_scores, share_tiles = [], [], [], [], []

    def collect(model, idx, set_rows, views, ys, ps):
        for i in idx:
            feats, _ = tissue_tiles(set_rows[i]["file"])
            tiles, whole, _ = split_scores(model, feats, views[i])
            tile_scores.append(tiles * len(feats))  # per-tile score, not divided by the tile count, so regions of any size compare
            tile_label.append(np.full(len(feats), ys[i]))
            tile_patient.append(np.full(len(feats), ps[i]))
            region_scores.append((ys[i], ps[i], tiles.sum(0), whole))

    for a, b in folds:
        collect(classifier(sl.ENGINE["C"]).fit(x_t[a], y_t[a]), b, train, view_t, y_t, patient_t)
        print("fold done", flush=True)
    collect(final, range(len(held)), held, view_v, y_v, patient_v)
    ts, tl, tp = np.vstack(tile_scores), np.concatenate(tile_label), np.concatenate(tile_patient)
    ry = np.array([r[0] for r in region_scores])
    rt, rw = np.array([r[2] for r in region_scores]), np.array([r[3] for r in region_scores])
    ids = np.unique(tp)
    where = {p: np.flatnonzero(tp == p) for p in ids}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(200)]
    result["tiles"], result["pairs"] = int(len(tl)), {}
    for a, b in itertools.combinations(range(7), 2):
        m = (tl == a) | (tl == b)
        s, lab = ts[:, a] - ts[:, b], tl == a
        auc = roc_auc_score(lab[m], s[m])
        boot = []
        for i in draws:
            mi = m[i]
            if lab[i][mi].any() and not lab[i][mi].all():
                boot.append(roc_auc_score(lab[i][mi], s[i][mi]))
        rm = (ry == a) | (ry == b)
        tile_part, view_part = rt[rm, a] - rt[rm, b], rw[rm, a] - rw[rm, b]
        lo, hi = range95(boot)
        result["pairs"][f"{sl.TYPES[a]} vs {sl.TYPES[b]}"] = {
            "tile_auc": round(float(auc), 3), "range95": [lo, hi],
            "region_auc": round(float(roc_auc_score(ry[rm] == a, tile_part + view_part)), 3),
            "region_auc_tiles_only": round(float(roc_auc_score(ry[rm] == a, tile_part)), 3),
            "share_of_decision_from_tiles": round(float(np.abs(tile_part).mean() / (np.abs(tile_part).mean() + np.abs(view_part).mean())), 3),
            "map_shown": bool(auc >= 0.60 and lo > 0.5)}
        print(sl.TYPES[a], "vs", sl.TYPES[b], result["pairs"][f"{sl.TYPES[a]} vs {sl.TYPES[b]}"], flush=True)
    result["maps_shown"] = sum(v["map_shown"] for v in result["pairs"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "why.json").write_text(json.dumps(result, indent=2))
    print("maps shown for", result["maps_shown"], "of", len(result["pairs"]), "pairs")


if __name__ == "__main__":
    main()
