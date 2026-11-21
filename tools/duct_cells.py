#!/usr/bin/env python3
"""Duct instrument, step 3 (ledger entry 46): single nuclei inside the duct cells of the gray-zone training regions.

Usage (from the project folder):
  .venv-nuclei/bin/python -I tools/duct_cells.py measure [shard of]   # StarDist on the 40x pixels -> data/features/duct_cells/<region>.npz
  .venv/bin/python -I tools/duct_cells.py test                        # 15 numbers per region and their tests -> outputs/ducts/cells.json

measure keeps every nucleus (place, size, shape, stain), so the numbers can be changed later without running StarDist again.
Real tissue pictures stay local.
"""
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data/bracs/BRACS_RoI/latest_version"
MAPS = ROOT / "data/features/duct_maps"
CELLS_DIR = ROOT / "data/features/duct_cells"
OUT = ROOT / "outputs/ducts"
CELLS = [0, 3, 7, 8, 9]   # duct-cell kinds of entry 41
PATCH = 32                # one map patch in 40x pixels (16 pixels at 20x)
WIN = 16                  # window side in patches: 512 pixels at 40x, 128 micrometres
MAX_WIN, MIN_SHARE, MIN_NUCLEI = 12, 0.2, 30
UM = 0.25                 # micrometres per 40x pixel
NEAR = 25.0               # neighbours count up to this distance, in micrometres
COLS = ["x", "y", "area", "ecc", "solidity", "angle", "stain"]
NAMES = ["log_count", "density", "area_median", "area_spread", "area_p90_p10", "area_vs_neighbours", "elong_mean", "elong_spread",
         "long_share", "solidity", "stain_mean", "stain_spread", "gap_median", "gap_spread", "alignment"]
GRAY = {"UDH": 2, "FEA": 3, "ADH": 4, "DCIS": 5}
# the textbook, UDH against ADH: the measurement, and True if UDH is expected to be the higher one
TEXTBOOK = {"area_spread": True, "elong_mean": True, "alignment": True, "gap_spread": True}


def gray_rows():
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "train" and r["label_name"] in GRAY]  # the test set stays locked


def key(r):
    return r["file"][:-4].replace("/", "__")


def windows(m):
    """Up to MAX_WIN windows (row, column in patches) with the largest share of duct-cell patches."""
    cells = np.isin(m, CELLS)
    cand = []
    for r in range(0, max(m.shape[0] - WIN // 2, 1), WIN):
        for c in range(0, max(m.shape[1] - WIN // 2, 1), WIN):
            share = cells[r:r + WIN, c:c + WIN].mean()
            if share >= MIN_SHARE:
                cand.append((-share, r, c))
    return [(r, c) for _, r, c in sorted(cand)[:MAX_WIN]], cells


def measure():
    from csbdeep.utils import normalize
    from PIL import Image
    from skimage.color import rgb2hed
    from skimage.measure import regionprops
    from skimage.segmentation import mark_boundaries
    from stardist.models import StarDist2D
    Image.MAX_IMAGE_PIXELS = 400_000_000
    shard, of = (int(v) for v in sys.argv[2:4]) if len(sys.argv) > 3 else (0, 1)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    todo = gray_rows()[shard::of]
    for n, r in enumerate(todo):
        dest = CELLS_DIR / (key(r) + ".npz")
        if dest.exists():
            continue
        wins, cells = windows(np.load(MAPS / (key(r) + ".npy")).astype(int))
        img = Image.open(SRC / r["file"]).convert("RGB")
        nuclei, cell_patches = [], 0
        for r0, c0 in wins:
            x0, y0 = c0 * PATCH, r0 * PATCH
            rgb = np.asarray(img.crop((x0, y0, min(x0 + WIN * PATCH, img.width), min(y0 + WIN * PATCH, img.height))))
            if min(rgb.shape[:2]) < 64:
                continue
            cell_patches += int(cells[r0:r0 + WIN, c0:c0 + WIN].sum())
            labels, _ = model.predict_instances(normalize(rgb, 1, 99.8, axis=(0, 1, 2)))
            h, w = labels.shape
            kept = np.zeros_like(labels)
            for p in regionprops(labels, intensity_image=rgb2hed(rgb)[..., 0]):
                a, b, c, d = p.bbox
                cy, cx = p.centroid
                row, col = int((y0 + cy) // PATCH), int((x0 + cx) // PATCH)
                if a == 0 or b == 0 or c == h or d == w or row >= cells.shape[0] or col >= cells.shape[1] or not cells[row, col]:
                    continue
                nuclei.append([x0 + cx, y0 + cy, p.area, p.eccentricity, p.solidity, p.orientation, p.intensity_mean])
                kept[labels == p.label] = p.label
            look = OUT / f"cells_outline_{r['label_name']}.png"
            if not look.exists() and kept.any():  # one outline picture per type, for a check by eye
                Image.fromarray((mark_boundaries(rgb, kept, color=(0, 1, 0)) * 255).astype(np.uint8)).save(look)
        np.savez_compressed(dest, nuclei=np.array(nuclei, np.float32).reshape(-1, len(COLS)), windows=len(wins), cell_patches=cell_patches)
        if n % 25 == 0:
            print(f"shard {shard}: {n + 1}/{len(todo)} {r['file']} windows {len(wins)} nuclei {len(nuclei)}", flush=True)
    print(f"shard {shard}: done", flush=True)


def numbers(z):
    """15 numbers from the nuclei of one region, or None if it holds too few."""
    from scipy.spatial import cKDTree
    nu = z["nuclei"]
    if len(nu) < MIN_NUCLEI:
        return None
    xy, area, ecc, sol, ang, stain = nu[:, :2] * UM, nu[:, 2] * UM ** 2, nu[:, 3], nu[:, 4], nu[:, 5], nu[:, 6]
    dist, idx = cKDTree(xy).query(xy, k=6)
    dist, idx = dist[:, 1:], idx[:, 1:]
    near = dist <= NEAR
    la = np.log(area)
    with np.errstate(invalid="ignore", divide="ignore"):
        nb_la = np.where(near, la[idx], 0).sum(1) / near.sum(1)
        long_ = ecc > 0.6  # a round nucleus has no direction
        pair = near & long_[idx] & long_[:, None]
        align = np.where(pair, np.cos(2 * (ang[:, None] - ang[idx])), 0).sum() / max(pair.sum(), 1)
    has = near.any(1)
    gap = dist[:, 0]
    p10, p90 = np.percentile(area, [10, 90])
    return [np.log(len(nu)), len(nu) / max(float(z["cell_patches"]) * (PATCH * UM) ** 2, 1) * 1000, np.median(area), area.std() / area.mean(),
            np.log(p90 / p10), float(np.abs(la - nb_la)[has].mean()) if has.any() else 0.0, ecc.mean(), ecc.std(), (ecc > 0.8).mean(),
            sol.mean(), stain.mean(), stain.std(), np.median(gap), gap.std() / gap.mean(), float(align)]


def auc_range(y, s, draws):
    from sklearn.metrics import roc_auc_score
    d = [roc_auc_score(y[i], s[i]) for i in draws if 0 < y[i].sum() < len(i)]
    return round(float(roc_auc_score(y, s)), 3), [round(float(v), 3) for v in np.percentile(d, [2.5, 97.5])]


def pair_score(x, y, patient, fold):
    """Out-of-fold score for 'ADH, not UDH', the strength C chosen inside each fold by AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
    from train_classifier import STRENGTHS, classifier
    score = np.zeros(len(y))
    for f in np.unique(fold):
        a, b = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=0).split(y[a], y[a], patient[a]))
        best = None
        for c in STRENGTHS:
            p = np.zeros(len(a))
            for ia, ib in inner:
                p[ib] = classifier(c).fit(x[a][ia], y[a][ia]).decision_function(x[a][ib])
            s = roc_auc_score(y[a], p)
            if best is None or s > best[0]:
                best = (s, c)
        score[b] = classifier(best[1]).fit(x[a], y[a]).decision_function(x[b])
    return score


def test():
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import short_lists as sl
    from train_classifier import patient_draws, rows
    train = rows("train")
    prob, _, _, folds, *_ = sl.load()
    fold_of = np.zeros(len(train), int)
    for f, (_, b) in enumerate(folds):
        fold_of[b] = f
    where = {r["file"]: i for i, r in enumerate(train)}
    x, y, patient, fold, e, dropped, missing = [], [], [], [], [], {k: 0 for k in GRAY}, 0
    for r in gray_rows():
        path = CELLS_DIR / (key(r) + ".npz")
        if not path.exists():
            missing += 1
            continue
        v = numbers(np.load(path))
        if v is None:
            dropped[r["label_name"]] += 1
            continue
        i = where[r["file"]]
        x.append(v); y.append(GRAY[r["label_name"]]); patient.append(int(r["patient"])); fold.append(fold_of[i])
        e.append(np.log(prob[i, 4] / prob[i, 2]))
    assert missing == 0, f"{missing} regions not measured yet"
    x, y, patient, fold, e = np.array(x), np.array(y), np.array(patient), np.array(fold), np.array(e)
    assert np.isfinite(x).all() and x.shape[1] == len(NAMES)
    np.save(OUT / "cells_train.npy", x)
    rng = np.random.default_rng(0)
    result = {"regions": {k: int((y == v).sum()) for k, v in GRAY.items()}, "dropped_under_30_nuclei": dropped, "measurements": NAMES,
              "medians_by_type": {n: {k: round(float(np.median(x[y == v, j])), 3) for k, v in GRAY.items()} for j, n in enumerate(NAMES)}}

    # reported: every measurement for the three pairs (AUC for 'the second type is the higher one')
    result["single_measurements_auc"] = {}
    for name, (lo, hi) in {"UDH-ADH": (2, 4), "FEA-ADH": (3, 4), "ADH-DCIS": (4, 5)}.items():
        mk = np.isin(y, [lo, hi])
        result["single_measurements_auc"][name] = {n: round(float(roc_auc_score(y[mk] == hi, x[mk, j])), 3) for j, n in enumerate(NAMES)}

    mk = np.isin(y, [2, 4])
    xp, adh, pp, fp, ep = x[mk], (y[mk] == 4).astype(int), patient[mk], fold[mk], e[mk]
    draws = patient_draws(pp, rng)

    # test 1: the textbook
    t1 = {}
    for n, udh_higher in TEXTBOOK.items():
        s = xp[:, NAMES.index(n)]
        auc, rg = auc_range(1 - adh if udh_higher else adh, s, draws)
        t1[n] = {"expected": "higher in UDH" if udh_higher else "higher in ADH", "auc": auc, "range95": rg, "confirmed": bool(auc >= 0.65 and rg[0] > 0.5)}
    result["test1_textbook"] = t1
    result["test1_works"] = sum(v["confirmed"] for v in t1.values()) >= 2

    # test 2: the 15 numbers alone, next to the adopted classifier on the same regions
    cells = pair_score(xp, adh, pp, fp)
    result["test2_udh_against_adh"] = {"cell measurements": dict(zip(("auc", "range95"), auc_range(adh, cells, draws))),
                                       "adopted classifier": dict(zip(("auc", "range95"), auc_range(adh, ep, draws)))}

    # test 3: the witness, where the adopted classifier prefers the wrong one of the two
    wrong = (ep > 0) != (adh == 1)
    wd = patient_draws(pp[wrong], rng)
    auc, rg = auc_range(adh[wrong], cells[wrong], wd)
    result["test3_witness"] = {"regions_classifier_wrong": int(wrong.sum()), "of": int(len(adh)), "auc": auc, "range95": rg, "witness": bool(rg[0] > 0.5)}

    # test 4: the classifier's score joined with the 15 numbers
    both = pair_score(np.column_stack([ep, xp]), adh, pp, fp)
    d = [roc_auc_score(adh[i], both[i]) - roc_auc_score(adh[i], ep[i]) for i in draws if 0 < adh[i].sum() < len(i)]
    result["test4_joined"] = {"auc": round(float(roc_auc_score(adh, both)), 3), "minus_classifier": round(float(roc_auc_score(adh, both) - roc_auc_score(adh, ep)), 4),
                              "range95_of_difference": [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])], "adopted": bool(np.percentile(d, 2.5) > 0)}
    (OUT / "cells.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"measure": measure, "test": test}[sys.argv[1]]()
