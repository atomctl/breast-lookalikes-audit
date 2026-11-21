#!/usr/bin/env python3
"""Ledger entry 34: find atypical duct tissue (ADH and DCIS as one group), then let the measured extent decide between them.

Usage (from the project folder):
  python -I tools/wsi_atypia_extent.py   # needs outputs/grayzone/wsi_lesions.csv (entry 33) -> outputs/grayzone/wsi_atypia_extent.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import wsi_lesion_extent as le
from train_classifier import range95
from wsi_extent import TILE_MM

OUT = le.OUT
FAR = 20  # tiles


def far_from_lesions(s, regions):
    """Unmarked tiles more than FAR tiles away from every marked ADH, DCIS or FEA region."""
    centre = s["coords"] + s["step"] / 2
    far = s["mark"] == ""
    pad = FAR * s["step"]
    for r in regions[s["slide"]]["matched"]:
        if r["label"] in ("ADH", "DCIS", "FEA"):
            x0, y0, x1, y1 = r["box"]
            far &= ~((centre[:, 0] >= x0 - pad) & (centre[:, 0] < x1 + pad) & (centre[:, 1] >= y0 - pad) & (centre[:, 1] < y1 + pad))
    return far


def main():
    slides = le.load()
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    patients = np.array([s["patient"] for s in slides])
    lesions, tile_y, tile_p, far_flag, fea_flag = [], [], [], [0, 0], [0, 0]
    for k, (train, test) in enumerate(GroupKFold(5).split(slides, groups=patients)):
        x, y = [], []
        for i in train:
            s = slides[i]
            pos = np.isin(s["mark"], ["ADH", "DCIS"])
            neg = np.isin(s["mark"], ["N", "PB", "UDH"])
            if s["label"] == "ADH":
                neg |= far_from_lesions(s, regions)
            x.append(s["feats"][pos | neg]); y.append(pos[pos | neg])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000)).fit(np.concatenate(x), np.concatenate(y))
        for i in test:
            s = slides[i]
            prob = model.predict_proba(s["feats"])[:, 1]
            for row in le.lesion_extents(s, prob):
                lesions.append(row)
            pos, neg = np.isin(s["mark"], ["ADH", "DCIS"]), np.isin(s["mark"], ["N", "PB", "UDH"])
            tile_y.append(pos[pos | neg]); tile_p.append(prob[pos | neg])
            fea = s["mark"] == "FEA"
            fea_flag[0] += int((prob[fea] >= 0.5).sum()); fea_flag[1] += int(fea.sum())
            if s["label"] == "ADH":
                far = far_from_lesions(s, regions)
                far_flag[0] += int((prob[far] >= 0.5).sum()); far_flag[1] += int(far.sum())
        print("fold", k, "done", flush=True)
    t = pd.DataFrame(lesions).rename(columns={"mean_probability": "mean_atypia_probability"})
    old = pd.read_csv(OUT / "wsi_lesions.csv")
    assert len(t) == len(old) and (t["slide"] == old["slide"]).all() and (t["label"] == old["label"]).all() and (t["tiles"] == old["tiles"]).all()
    t["entry33_dcis_probability"] = old["mean_probability"]
    t["entry33_extent_mm"] = old["extent_mm"]
    # the marked region's own longest side, from its box
    box_side = []
    for s in slides:
        centre = s["coords"] + s["step"] / 2
        for r in s["regions"]:
            x0, y0, x1, y1 = r["box"]
            if ((centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)).any():
                box_side.append(max(x1 - x0, y1 - y0) / s["step"] * TILE_MM)
    t["region_side_mm"] = box_side
    rng = np.random.default_rng(0)

    def compare(tt):
        y = (tt["label"] == "DCIS").to_numpy()
        ids, pat = tt["patient"].unique(), tt["patient"].to_numpy()
        auc = lambda col, i=slice(None): float(roc_auc_score(y[i], tt[col].to_numpy()[i]))
        draws = []
        while len(draws) < 2000:
            i = np.concatenate([np.flatnonzero(pat == p) for p in rng.choice(ids, len(ids))])
            if y[i].any() and not y[i].all():
                draws.append(i)
        e = tt["extent_mm"].to_numpy()
        return {"dcis_regions": int(y.sum()), "adh_regions": int((~y).sum()),
                "auc_atypia_extent": round(auc("extent_mm"), 3), "auc_entry33_dcis_probability": round(auc("entry33_dcis_probability"), 3),
                "auc_entry33_extent": round(auc("entry33_extent_mm"), 3), "auc_region_own_side": round(auc("region_side_mm"), 3),
                "auc_mean_atypia_probability": round(auc("mean_atypia_probability"), 3),
                "extent_minus_entry33_probability": round(auc("extent_mm") - auc("entry33_dcis_probability"), 3),
                "range95": range95([auc("extent_mm", i) - auc("entry33_dcis_probability", i) for i in draws]),
                "extent_minus_region_own_side": round(auc("extent_mm") - auc("region_side_mm"), 3),
                "range95_vs_own_side": range95([auc("extent_mm", i) - auc("region_side_mm", i) for i in draws]),
                "median_extent_mm": {"ADH": round(float(np.median(e[~y])), 2), "DCIS": round(float(np.median(e[y])), 2)},
                "rule_2mm": {"dcis_called_dcis": round(float((e[y] >= 2).mean()), 3), "adh_called_dcis": round(float((e[~y] >= 2).mean()), 3),
                             "accuracy": round(float(((e >= 2) == y).mean()), 3)}}

    ty, tp = np.concatenate(tile_y), np.concatenate(tile_p)
    primary = compare(t)
    result = {"slides": len(slides), "patients": int(len(np.unique(patients))),
              "detector": {"tile_auc_atypia_vs_marked_benign": round(float(roc_auc_score(ty, tp)), 3), "atypia_tiles_found": round(float((tp[ty] >= 0.5).mean()), 3),
                           "marked_benign_tiles_flagged": round(float((tp[~ty] >= 0.5).mean()), 3),
                           "far_unmarked_tiles_on_adh_slides_flagged": round(far_flag[0] / far_flag[1], 4), "fea_tiles_flagged": round(fea_flag[0] / fea_flag[1], 3)},
              "primary": primary, "adh_regions_on_adh_slides_only": compare(t[(t["label"] == "DCIS") | (t["slide_label"] == "ADH")].reset_index(drop=True)),
              "verdict": "PASS" if primary["range95"][0] > 0 else "FAIL"}
    t.to_csv(OUT / "wsi_lesions_atypia.csv", index=False)
    (OUT / "wsi_atypia_extent.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
