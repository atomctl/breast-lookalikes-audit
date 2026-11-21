#!/usr/bin/env python3
"""Ledger entry 33: a tile scorer shown all unmarked tissue of ADH slides, and the extent of DCIS-like tissue around each marked lesion.

Usage (from the project folder):
  python -I tools/wsi_lesion_extent.py   # -> outputs/grayzone/wsi_lesion_extent.json
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
from train_classifier import ROOT, range95
from wsi_extent import TILE_MM, WSI, largest_group

OUT = ROOT / "outputs/grayzone"


def load():
    info = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    who = dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    slides = []
    for remote in json.loads((OUT / "wsi_all68.json").read_text()) + json.loads((OUT / "wsi_val17.json").read_text()):
        name = pathlib.Path(remote).stem
        z = np.load(WSI / f"{name}.npz", allow_pickle=True)
        step = int(z["step"])
        centre = z["coords"] + step / 2
        mark = np.full(len(centre), "", dtype=object)
        for r in regions[name]["matched"]:
            x0, y0, x1, y1 = r["box"]
            mark[(centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)] = r["label"]
        slides.append({"slide": name, "label": "DCIS" if "Type_DCIS" in remote else "ADH", "patient": int(who[name]), "feats": z["feats"].astype(np.float32),
                       "coords": z["coords"], "step": step, "mark": mark, "regions": [r for r in regions[name]["matched"] if r["label"] in ("ADH", "DCIS")]})
    return slides


def lesion_extents(s, prob):
    """Per marked ADH or DCIS region: mean tile probability inside it and the longest side (mm) of the DCIS-like groups touching it."""
    like = prob >= 0.5
    g = (s["coords"] // s["step"]).astype(int)
    lo = g.min(0)
    g = g - lo + 1  # one empty cell around the grid
    grid = np.zeros(g.max(0) + 2, bool)
    grid[g[like, 0], g[like, 1]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    side = {k + 1: max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for k, sl in enumerate(ndimage.find_objects(lab))}
    centre = s["coords"] + s["step"] / 2
    out = []
    for r in s["regions"]:
        x0, y0, x1, y1 = r["box"]
        inside = (centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)
        if not inside.any():
            continue
        cells = np.zeros_like(grid)
        cells[g[inside, 0], g[inside, 1]] = True
        touched = np.unique(lab[ndimage.binary_dilation(cells, structure=np.ones((3, 3)))])
        extent = max([side[k] for k in touched if k], default=0) * TILE_MM
        out.append({"slide": s["slide"], "slide_label": s["label"], "patient": s["patient"], "label": r["label"], "tiles": int(inside.sum()),
                    "mean_probability": float(prob[inside].mean()), "extent_mm": float(extent)})
    return out


def main():
    slides = load()
    patients = np.array([s["patient"] for s in slides])
    lesions, per_slide, found, alarm, looked = [], [], [0, 0], [0, 0], {}
    fold_of = {}
    for k, (train, test) in enumerate(GroupKFold(5).split(slides, groups=patients)):
        x, y = [], []
        for i in train:
            s = slides[i]
            keep = (s["mark"] != "") | (s["label"] == "ADH")  # marked tiles everywhere, and all unmarked tissue of ADH slides
            x.append(s["feats"][keep]); y.append(s["mark"][keep] == "DCIS")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000)).fit(np.concatenate(x), np.concatenate(y))
        for i in test:
            s = slides[i]
            fold_of[s["patient"]] = k
            prob = model.predict_proba(s["feats"])[:, 1]
            like = prob >= 0.5
            lesions += lesion_extents(s, prob)
            n, side = largest_group(s["coords"][like], s["step"])
            per_slide.append({"slide": s["slide"], "label": s["label"], "patient": s["patient"], "largest_mm2": n * TILE_MM ** 2})
            m = s["mark"] == "DCIS"
            found[0] += int(like[m].sum()); found[1] += int(m.sum())
            if s["label"] == "ADH":
                f = s["mark"] == ""
                alarm[0] += int(like[f].sum()); alarm[1] += int(f.sum())
            if s["slide"] in ("BRACS_1797", "BRACS_1977"):
                looked[s["slide"]] = int(like.sum())
        print("fold", k, "done", flush=True)
    t = pd.DataFrame(lesions)
    t["fold"] = t["patient"].map(fold_of)
    t["both"] = 0.0
    feats = np.c_[t["mean_probability"], np.log1p(t["extent_mm"])]
    for k in range(5):  # the two-number model, fitted on the other patient groups
        a, b = (t["fold"] != k).to_numpy(), (t["fold"] == k).to_numpy()
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(feats[a], t["label"][a] == "DCIS")
        t.loc[b, "both"] = m.predict_proba(feats[b])[:, 1]
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
        return {"dcis_regions": int(y.sum()), "adh_regions": int((~y).sum()), "patients": int(len(ids)),
                "auc_mean_probability": round(auc("mean_probability"), 3), "auc_extent_alone": round(auc("extent_mm"), 3), "auc_both": round(auc("both"), 3),
                "both_minus_mean_probability": round(auc("both") - auc("mean_probability"), 3),
                "range95": range95([auc("both", i) - auc("mean_probability", i) for i in draws]),
                "median_extent_mm": {"ADH": round(float(tt["extent_mm"][~y].median()), 2), "DCIS": round(float(tt["extent_mm"][y].median()), 2)},
                "share_extent_at_least_2mm": {"ADH": round(float((tt["extent_mm"][~y] >= 2).mean()), 3), "DCIS": round(float((tt["extent_mm"][y] >= 2).mean()), 3)}}

    ps = pd.DataFrame(per_slide)
    primary = compare(t)
    result = {"slides": len(slides), "patients": int(len(np.unique(patients))),
              "a_share_unmarked_tiles_on_adh_slides_scored_dcis_like": round(alarm[0] / alarm[1], 4),
              "a_share_marked_dcis_tiles_found": round(found[0] / found[1], 4), "a_flagged_tiles_on_looked_slides": looked,
              "a_slide_level_auc_largest_patch": round(float(roc_auc_score(ps["label"] == "DCIS", ps["largest_mm2"])), 3),
              "b_primary": primary, "b_adh_regions_on_adh_slides_only": compare(t[(t["label"] == "DCIS") | (t["slide_label"] == "ADH")].reset_index(drop=True)),
              "verdict": "PASS" if primary["range95"][0] > 0 else "FAIL"}
    t.to_csv(OUT / "wsi_lesions.csv", index=False)
    (OUT / "wsi_lesion_extent.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
