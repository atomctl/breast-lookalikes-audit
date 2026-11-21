#!/usr/bin/env python3
"""Ledger entry 29: does the tile scoring find the marked DCIS on the whole slides, and where are its false alarms?

Usage (from the project folder):
  python -I tools/wsi_diagnose.py   # needs outputs/grayzone/wsi_regions.json -> outputs/grayzone/wsi_diagnose.json
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import classifier, rows
from wsi_extent import ADH, DCIS, WSI, tile_logits

OUT = sl.ROOT / "outputs/grayzone"


def fold_models():
    """One classifier per fold, and which fold never saw each patient (as in entry 27)."""
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    x = sl.engine_features("train")
    fold_of, models = {}, []
    for n, (a, b) in enumerate(folds):
        models.append(classifier(sl.ENGINE["C"]).fit(x[a], y[a]))
        fold_of.update({int(p): n for p in np.unique(patient[b])})
    info = pd.read_excel(sl.ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    return models, fold_of, dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))


def main():
    models, fold_of, who = fold_models()
    final = joblib.load(sl.ROOT / "outputs/m2/engine.joblib")
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    inside = {"DCIS": [], "ADH": []}   # per marked region: (tiles, DCIS-like tiles, ADH-like tiles)
    outside = {"DCIS": [0, 0], "ADH": [0, 0]}  # per slide label: DCIS-like tiles outside every marked region, all DCIS-like tiles
    sides = []
    for f in sorted(WSI.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        label = "DCIS" if "Type_DCIS" in str(z["remote"]) else "ADH"
        p = int(who[f.stem])
        top = tile_logits(models[fold_of[p]] if p in fold_of else final, z["feats"].astype(np.float32)).argmax(1)
        centre = z["coords"] + int(z["step"]) / 2
        marked = np.zeros(len(top), bool)
        for r in regions[f.stem]["matched"]:
            x0, y0, x1, y1 = r["box"]
            m = (centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)
            marked |= m
            if r["label"] in inside and m.any():
                inside[r["label"]].append((int(m.sum()), int((top[m] == DCIS).sum()), int((top[m] == ADH).sum())))
            if r["label"] == "DCIS":
                sides.append(max(x1 - x0, y1 - y0) * float(z["mpp"]) / 1000)
        outside[label][0] += int(((top == DCIS) & ~marked).sum())
        outside[label][1] += int((top == DCIS).sum())
    d, a = np.array(inside["DCIS"]), np.array(inside["ADH"])
    result = {
        "marked_dcis_regions": int(len(d)), "tiles_in_marked_dcis": int(d[:, 0].sum()),
        "1_share_of_tiles_in_marked_dcis_scored_dcis_like": round(float(d[:, 1].sum() / d[:, 0].sum()), 4),
        "1_share_of_tiles_in_marked_dcis_scored_adh_like": round(float(d[:, 2].sum() / d[:, 0].sum()), 4),
        "1_share_of_marked_dcis_regions_at_least_half_dcis_like": round(float((d[:, 1] >= d[:, 0] / 2).mean()), 4),
        "marked_adh_regions": int(len(a)), "tiles_in_marked_adh": int(a[:, 0].sum()),
        "2_share_of_tiles_in_marked_adh_scored_dcis_like": round(float(a[:, 1].sum() / a[:, 0].sum()), 4),
        "2_share_of_tiles_in_marked_adh_scored_adh_like": round(float(a[:, 2].sum() / a[:, 0].sum()), 4),
        "3_share_of_dcis_like_tiles_outside_marked_regions": {k: round(v[0] / max(v[1], 1), 4) for k, v in outside.items()},
        "3_dcis_like_tiles": {k: v[1] for k, v in outside.items()},
        "4_marked_dcis_longest_side_mm_median": round(float(np.median(sides)), 2),
        "4_share_of_marked_dcis_at_least_2mm": round(float((np.array(sides) >= 2).mean()), 4),
    }
    (OUT / "wsi_diagnose.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
