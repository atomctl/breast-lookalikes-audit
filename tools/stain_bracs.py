#!/usr/bin/env python3
"""Reading the biology, gate 3 (ledger entry 49): the stain model of entry 48 applied inside the ducts of the BRACS gray-zone regions.

Usage (from the project folder):
  python -I tools/stain_bracs.py fit           # the stain model on all four Xenium patients -> outputs/stain/model.joblib
  python -I tools/stain_bracs.py score train   # a predicted stain score for every duct-cell patch -> outputs/stain/bracs_train.npz
  python -I tools/stain_bracs.py score val
  python -I tools/stain_bracs.py test          # -> outputs/stain/gate3.json

Real tissue pictures stay local.
"""
import json
import pathlib
import sys

import joblib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import stain_xenium as sx
from train_classifier import ROOT, classifier, patient_draws, range95, rows

OUT = ROOT / "outputs/stain"
SRC = ROOT / "data/bracs/BRACS_RoI/latest_version"
GRAY = {"UDH": 2, "FEA": 3, "ADH": 4, "DCIS": 5}
CELLS = [0, 3, 7, 8, 9]   # duct-cell kinds of entry 41
MIN_PATCHES = 20


def fit():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    names = [s for s in sx.SAMPLES if (sx.DATA / s / "features.npz").exists()]
    data = [sx.load(s) for s in names]
    x, y = np.vstack([d["x"] for d in data]), np.hstack([d["y"] for d in data])
    g = np.hstack([d["block"] + k * 10 ** 9 for k, d in enumerate(data)])
    best = None
    for c in (0.00001, 0.0001, 0.001, 0.01, 0.1):
        p = np.zeros(len(y))
        for i, j in GroupKFold(3).split(x, y, g):
            p[j] = classifier(c).fit(x[i], y[i]).decision_function(x[j])
        s = roc_auc_score(y, p)
        print("C", c, "block-held-out AUC", round(s, 4), flush=True)
        if best is None or s > best[0]:
            best = (s, c)
    joblib.dump({"model": classifier(best[1]).fit(x, y), "C": best[1], "samples": names, "cells": int(len(y)), "positive": int(y.sum())}, OUT / "model.joblib")
    print("fitted on", names, "cells", len(y), "positive", int(y.sum()), "C", best[1])


def score(split):
    import duct_tokens as dt
    import extract_features as ef
    model = joblib.load(OUT / "model.joblib")["model"]
    kinds = joblib.load(ROOT / "outputs/ducts/kinds.joblib")
    tok = dt.Tokens()
    todo = [r for r in rows(split) if r["label_name"] in GRAY]
    out = {"region": [], "patches": [], "mean": [], "share_positive": [], "p90": []}
    for n, r in enumerate(todo):
        tiles, coords, tissue, size, padded = ef.region_tiles(SRC / r["file"])
        t = tok(tiles)                                            # (n, 14, 14, 1024)
        kind = kinds["kmeans"].predict(kinds["pca"].transform(t.reshape(-1, t.shape[-1]))).reshape(len(tiles), sx.GRID, sx.GRID)
        val = {}                                                  # one score per patch of the region; overlapping tiles overwrite
        for b, (x0, y0) in enumerate(coords):
            rr, cc = np.nonzero(np.isin(kind[b], CELLS))
            if not len(rr):
                continue
            centre = t[b, rr, cc]
            around = np.stack([t[b, max(i - 1, 0):min(i + 2, sx.GRID), max(j - 1, 0):min(j + 2, sx.GRID)].mean((0, 1)) for i, j in zip(rr, cc)])
            s = model.decision_function(np.hstack([centre, around]).astype(np.float32))
            for i, j, v in zip(rr, cc, s):
                val[(y0 // 16 + i, x0 // 16 + j)] = v
        v = np.array(list(val.values()))
        out["region"].append(r["file"]); out["patches"].append(len(v))
        out["mean"].append(float(v.mean()) if len(v) else np.nan)
        out["share_positive"].append(float((v > 0).mean()) if len(v) else np.nan)
        out["p90"].append(float(np.percentile(v, 90)) if len(v) else np.nan)
        if n % 100 == 0:
            print(f"{split}: {n}/{len(todo)}", flush=True)
    np.savez(OUT / f"bracs_{split}.npz", **{k: np.array(v) for k, v in out.items()})
    print(split, "regions", len(todo))


def auc(y, s, patient, rng):
    from sklearn.metrics import roc_auc_score
    d = [roc_auc_score(y[i], s[i]) for i in patient_draws(patient, rng, 1000) if 0 < y[i].sum() < len(i)]
    return {"auc": round(float(roc_auc_score(y, s)), 3), "range95": range95(d)}


def analyse(split, prob, with_size):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from grayzone_slide_origin import SEVERITY, slide_labels
    rng = np.random.default_rng(0)
    z = np.load(OUT / f"bracs_{split}.npz")
    all_rows = rows(split)
    where = {r["file"]: i for i, r in enumerate(all_rows)}
    lab = slide_labels()
    idx = np.array([where[f] for f in z["region"]])
    ok = z["patches"] >= MIN_PATCHES
    y = np.array([int(all_rows[i]["label"]) for i in idx])
    patient = np.array([int(all_rows[i]["patient"]) for i in idx])
    slab = np.array([SEVERITY[lab[all_rows[i]["slide"]]] for i in idx])
    s, e = z["mean"], np.log(prob[idx, 4] / prob[idx, 2])
    res = {"regions": {k: int(((y == v) & ok).sum()) for k, v in GRAY.items()}, "dropped_under_20_patches": {k: int(((y == v) & ~ok).sum()) for k, v in GRAY.items()},
           "median_S": {k: round(float(np.median(s[(y == v) & ok])), 3) for k, v in GRAY.items()},
           "median_share_positive": {k: round(float(np.median(z["share_positive"][(y == v) & ok])), 3) for k, v in GRAY.items()}}
    m = ok & np.isin(y, [2, 4])
    udh, sp, pp, ep = (y[m] == 2).astype(int), s[m], patient[m], e[m]
    res["test1_udh_above_adh"] = auc(udh, sp, pp, rng)
    res["test1_udh_above_adh"]["confirmed"] = bool(res["test1_udh_above_adh"]["auc"] >= 0.70 and res["test1_udh_above_adh"]["range95"][0] > 0.5)
    res["share_positive_udh_above_adh"] = auc(udh, z["share_positive"][m], pp, rng)
    size = {"log_patches": np.log(z["patches"][m])}
    if with_size:
        import duct_measure as dm
        meas = np.load(ROOT / "outputs/ducts/measurements_train.npy")
        size["thickness_mean"] = meas[idx[m], dm.NAMES.index("thickness_mean_mm")]
    draws = [i for i in patient_draws(pp, rng, 1000) if 0 < udh[i].sum() < len(i)]
    res["test1b_size"] = {}
    for name, v in size.items():
        a = roc_auc_score(udh, v)
        v2 = v if a >= 0.5 else -v          # a size number may point either way; its better direction is used
        d = [roc_auc_score(udh[i], sp[i]) - roc_auc_score(udh[i], v2[i]) for i in draws]
        res["test1b_size"][name] = {"auc_best_direction": round(float(max(a, 1 - a)), 3), "udh_is": "higher" if a >= 0.5 else "lower",
                                    "S_minus_size": round(float(roc_auc_score(udh, sp) - max(a, 1 - a)), 3), "range95": range95(d)}
    res["test1b_stain_not_size"] = bool(all(v["range95"][0] > 0 for v in res["test1b_size"].values()))
    res["rank_correlation_S_with_classifier_adh_score"] = round(float(spearmanr(sp, ep)[0]), 3)
    res["classifier_udh_against_adh"] = auc(1 - udh, ep, pp, rng)
    wrong = (ep > 0) != (udh == 0)
    res["test2_witness"] = dict(auc(udh[wrong], sp[wrong], pp[wrong], rng), regions=int(wrong.sum()))
    res["test2_witness"]["witness"] = bool(res["test2_witness"]["range95"][0] > 0.5)
    if with_size:                           # test 3 needs the project's patient folds, which exist for the training patients
        import short_lists as sl
        folds = sl.load()[3]
        fold_of = np.zeros(len(all_rows), int)
        for f, (_, b) in enumerate(folds):
            fold_of[b] = f
        fo, adh = fold_of[idx[m]], 1 - udh
        both = np.zeros(len(adh))
        x2 = np.column_stack([ep, sp])
        for f in np.unique(fo):
            both[fo == f] = classifier(1.0).fit(x2[fo != f], adh[fo != f]).decision_function(x2[fo == f])
        d = [roc_auc_score(adh[i], both[i]) - roc_auc_score(adh[i], ep[i]) for i in draws]
        res["test3_joined"] = {"auc": round(float(roc_auc_score(adh, both)), 3), "minus_classifier": round(float(roc_auc_score(adh, both) - roc_auc_score(adh, ep)), 4),
                               "range95": range95(d), "adopted": bool(np.percentile(d, 2.5) > 0)}
    a_grp, c_grp = ok & (y == 4) & (slab == 4), ok & (y == 4) & (slab >= 5)
    res["test4_two_kinds_of_adh"] = {"regions": {"A": int(a_grp.sum()), "C": int(c_grp.sum())},
                                     "median_S": {"A": round(float(np.median(s[a_grp])), 3) if a_grp.any() else None, "C": round(float(np.median(s[c_grp])), 3) if c_grp.any() else None}}
    if a_grp.sum() >= 5 and c_grp.sum() >= 5:
        k = a_grp | c_grp
        res["test4_two_kinds_of_adh"]["A_above_C"] = auc(a_grp[k].astype(int), s[k], patient[k], rng)
        k = a_grp | (ok & (y == 2))
        res["test4_two_kinds_of_adh"]["udh_above_group_A"] = auc((y[k] == 2).astype(int), s[k], patient[k], rng)
        k = c_grp | (ok & (y == 2))
        res["test4_two_kinds_of_adh"]["udh_above_group_C"] = auc((y[k] == 2).astype(int), s[k], patient[k], rng)
    for name, (lo, hi) in {"FEA above ADH": (4, 3), "ADH above DCIS": (5, 4), "UDH above DCIS": (5, 2), "UDH above FEA": (3, 2)}.items():
        k = ok & np.isin(y, [lo, hi])
        res[name] = auc((y[k] == hi).astype(int), s[k], patient[k], rng)
    return res


def test():
    import short_lists as sl
    prob, _, _, _, vprob, _, _ = sl.load()
    result = {"model": {k: v for k, v in joblib.load(OUT / "model.joblib").items() if k != "model"}, "train": analyse("train", prob, True)}
    if (OUT / "bracs_val.npz").exists():
        result["val"] = analyse("val", vprob, False)
    (OUT / "gate3.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"fit": fit, "score": score, "test": test}[sys.argv[1]](*sys.argv[2:])
