#!/usr/bin/env python3
"""Ledger entry 27: measure the extent of DCIS-like tissue across whole slides, and see whether it separates ADH from DCIS slides.

Usage (from the project folder):
  python -I tools/wsi_extent.py              # reads data/features/uni_wsi/*.npz -> outputs/grayzone/wsi_extent.json
  python -I tools/wsi_extent.py <list.json>  # only the slides in that list -> outputs/grayzone/wsi_extent_<list>.json

A tile is scored by the tile part of a region classifier that never saw the slide's patient.
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import classifier, range95, rows

WSI = sl.ROOT / "data/features/uni_wsi"
OUT = sl.ROOT / "outputs/grayzone"
D, ADH, DCIS = 1024, 4, 5
TILE_MM = 0.112  # one 224-pixel tile at 0.5 um per pixel


def tile_logits(model, feats):
    """The classifier's score per type from one tile alone (the whole-region view is left out)."""
    scaler, clf = model[0], model[1]
    return ((feats - scaler.mean_[:D]) / scaler.scale_[:D]) @ clf.coef_[:, :D].T + clf.intercept_


def largest_group(mask_xy, step):
    """Largest connected group of flagged tiles (touching sides or corners): (tiles, longest side in tiles)."""
    if not len(mask_xy):
        return 0, 0
    g = (mask_xy // step).astype(int)
    g -= g.min(0)
    grid = np.zeros(g.max(0) + 1, bool)
    grid[g[:, 0], g[:, 1]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    sizes = ndimage.sum(grid, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    xs, ys = np.nonzero(lab == best)
    return int(sizes[best - 1]), int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)


def main():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    x = sl.engine_features("train")
    final = joblib.load(sl.ROOT / "outputs/m2/engine.joblib")
    fold_of, models = {}, []
    for n, (a, b) in enumerate(folds):
        models.append(classifier(sl.ENGINE["C"]).fit(x[a], y[a]))
        fold_of.update({int(p): n for p in np.unique(patient[b])})
    info = pd.read_excel(sl.ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    who = dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))
    table = []
    for f in sorted(WSI.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        label = "DCIS" if "Type_DCIS" in str(z["remote"]) else "ADH" if "Type_ADH" in str(z["remote"]) else None
        if label is None or not len(z["feats"]):
            continue
        p = int(who[f.stem])
        model = models[fold_of[p]] if p in fold_of else final  # a classifier that never saw this patient
        logits = tile_logits(model, z["feats"].astype(np.float32))
        top = logits.argmax(1)
        step = int(z["step"])
        row = {"slide": f.stem, "label": label, "patient": p, "tissue_tiles": int(len(top))}
        for name, mask in (("dcis_like", top == DCIS), ("adh_like", top == ADH), ("malignant_like", top >= DCIS)):
            n, side = largest_group(z["coords"][mask], step)
            row.update({f"{name}_tiles": int(mask.sum()), f"{name}_share": round(float(mask.mean()), 4),
                        f"{name}_largest_mm2": round(n * TILE_MM ** 2, 3), f"{name}_longest_mm": round(side * TILE_MM, 2)})
        row["baseline_mean_dcis_minus_adh"] = round(float((logits[:, DCIS] - logits[:, ADH]).mean()), 4)
        table.append(row)
    t = pd.DataFrame(table)
    out = OUT / "wsi_extent.json"
    if len(sys.argv) > 1:  # only the slides in this list
        keep = {pathlib.Path(r).stem for r in json.loads(pathlib.Path(sys.argv[1]).read_text())}
        t = t[t["slide"].isin(keep)].reset_index(drop=True)
        table = t.to_dict("records")
        out = OUT / f"wsi_extent_{pathlib.Path(sys.argv[1]).stem}.json"
    is_dcis = (t["label"] == "DCIS").to_numpy()
    rng = np.random.default_rng(0)
    ids = t["patient"].unique()
    result = {"slides": int(len(t)), "adh_slides": int((~is_dcis).sum()), "dcis_slides": int(is_dcis.sum()), "patients": int(len(ids)), "measures": {}}
    for m in ["dcis_like_tiles", "dcis_like_share", "dcis_like_largest_mm2", "dcis_like_longest_mm", "malignant_like_largest_mm2", "adh_like_share", "baseline_mean_dcis_minus_adh"]:
        v = t[m].to_numpy(float)
        boot = []
        for _ in range(2000):
            i = np.concatenate([np.flatnonzero(t["patient"].to_numpy() == p) for p in rng.choice(ids, len(ids))])
            if is_dcis[i].any() and not is_dcis[i].all():
                boot.append(roc_auc_score(is_dcis[i], v[i]))
        result["measures"][m] = {"auc_dcis_vs_adh": round(float(roc_auc_score(is_dcis, v)), 3), "range95": range95(boot),
                                 "median_adh": round(float(np.median(v[~is_dcis])), 3), "median_dcis": round(float(np.median(v[is_dcis])), 3)}
    diff = []  # largest DCIS-like area against the baseline, on the same patient draws
    area, base = t["dcis_like_largest_mm2"].to_numpy(float), t["baseline_mean_dcis_minus_adh"].to_numpy(float)
    for _ in range(2000):
        i = np.concatenate([np.flatnonzero(t["patient"].to_numpy() == p) for p in rng.choice(ids, len(ids))])
        if is_dcis[i].any() and not is_dcis[i].all():
            diff.append(roc_auc_score(is_dcis[i], area[i]) - roc_auc_score(is_dcis[i], base[i]))
    result["largest_area_minus_baseline_auc"] = {"difference": round(float(roc_auc_score(is_dcis, area) - roc_auc_score(is_dcis, base)), 3), "range95": range95(diff)}
    rule = t["dcis_like_longest_mm"].to_numpy() >= 2.0
    result["two_mm_rule"] = {"dcis_slides_called_dcis": int((rule & is_dcis).sum()), "adh_slides_called_dcis": int((rule & ~is_dcis).sum())}
    result["table"] = table
    # self-check: a 3 x 2 block of tiles is one group of 6 with longest side 3
    blk = np.array([[0, 0], [448, 0], [896, 0], [0, 448], [448, 448], [896, 448], [4480, 4480]])
    assert largest_group(blk, 448) == (6, 3)
    out.write_text(json.dumps(result, indent=2))
    print(t[["slide", "label", "tissue_tiles", "dcis_like_share", "dcis_like_largest_mm2", "dcis_like_longest_mm", "adh_like_share", "baseline_mean_dcis_minus_adh"]].to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
