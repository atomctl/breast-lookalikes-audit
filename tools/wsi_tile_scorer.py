#!/usr/bin/env python3
"""Ledger entry 30: a scorer trained on single tiles, then the extent of DCIS-like tissue per slide.

Usage (from the project folder):
  python -I tools/wsi_tile_scorer.py   # -> outputs/grayzone/wsi_tile_scorer.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
from train_classifier import ROOT, range95
from wsi_extent import TILE_MM, WSI, largest_group

OUT = ROOT / "outputs/grayzone"
UNMARKED_PER_ADH_SLIDE = 1000


def load():
    info = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    who = dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    slides = []
    for f in sorted(WSI.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "/train/" not in str(z["remote"]):  # validation slides are kept out of training (entry 31)
            continue
        centre = z["coords"] + int(z["step"]) / 2
        mark = np.full(len(centre), "", dtype=object)  # the type of the marked region a tile lies in, or "" for unmarked tissue
        for r in regions[f.stem]["matched"]:
            x0, y0, x1, y1 = r["box"]
            mark[(centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)] = r["label"]
        slides.append({"slide": f.stem, "label": "DCIS" if "Type_DCIS" in str(z["remote"]) else "ADH", "patient": int(who[f.stem]),
                       "feats": z["feats"].astype(np.float32), "coords": z["coords"], "step": int(z["step"]), "mark": mark})
    return slides


def training_tiles(slides, rng):
    x, y = [], []
    for s in slides:
        marked = s["mark"] != ""
        keep = marked.copy()
        if s["label"] == "ADH":  # unmarked tissue of an ADH slide is not DCIS
            free = np.flatnonzero(~marked)
            keep[rng.choice(free, min(UNMARKED_PER_ADH_SLIDE, len(free)), replace=False)] = True
        x.append(s["feats"][keep])
        y.append(s["mark"][keep] == "DCIS")
    return np.concatenate(x), np.concatenate(y)


def main():
    slides = load()
    patients = np.array([s["patient"] for s in slides])
    rng = np.random.default_rng(0)
    table, in_dcis, free_adh = [], [0, 0], [0, 0]
    for train, test in GroupKFold(5).split(slides, groups=patients):
        x, y = training_tiles([slides[i] for i in train], rng)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)).fit(x, y)
        for i in test:
            s = slides[i]
            prob = model.predict_proba(s["feats"])[:, 1]
            like = prob >= 0.5
            n, side = largest_group(s["coords"][like], s["step"])
            table.append({"slide": s["slide"], "label": s["label"], "patient": s["patient"], "tissue_tiles": int(len(like)),
                          "dcis_like_share": round(float(like.mean()), 4), "dcis_like_largest_mm2": round(n * TILE_MM ** 2, 3),
                          "dcis_like_longest_mm": round(side * TILE_MM, 2), "mean_probability": round(float(prob.mean()), 4)})
            m = s["mark"] == "DCIS"
            in_dcis[0] += int(like[m].sum()); in_dcis[1] += int(m.sum())
            if s["label"] == "ADH":
                f = s["mark"] == ""
                free_adh[0] += int(like[f].sum()); free_adh[1] += int(f.sum())
    t = pd.DataFrame(table).sort_values("slide").reset_index(drop=True)
    old = pd.DataFrame(json.loads((OUT / "wsi_extent_wsi_all68.json").read_text())["table"]).set_index("slide")
    t["old_baseline"] = old.loc[t["slide"], "baseline_mean_dcis_minus_adh"].to_numpy()
    is_dcis = (t["label"] == "DCIS").to_numpy()
    ids, pat = t["patient"].unique(), t["patient"].to_numpy()
    draws = []
    while len(draws) < 2000:
        i = np.concatenate([np.flatnonzero(pat == p) for p in rng.choice(ids, len(ids))])
        if is_dcis[i].any() and not is_dcis[i].all():
            draws.append(i)
    auc = lambda col, i=slice(None): float(roc_auc_score(is_dcis[i], t[col].to_numpy(float)[i]))
    result = {"slides": int(len(t)), "patients": int(len(ids)),
              "share_of_tiles_in_marked_dcis_scored_dcis_like": round(in_dcis[0] / in_dcis[1], 4),
              "share_of_unmarked_tiles_on_adh_slides_scored_dcis_like": round(free_adh[0] / free_adh[1], 4), "auc": {}}
    for col in ["dcis_like_largest_mm2", "dcis_like_longest_mm", "dcis_like_share", "mean_probability", "old_baseline"]:
        result["auc"][col] = {"auc": round(auc(col), 3), "range95": range95([auc(col, i) for i in draws]),
                              "median_adh": round(float(t[col][~is_dcis].median()), 3), "median_dcis": round(float(t[col][is_dcis].median()), 3)}
    for rival in ["old_baseline", "mean_probability"]:
        result[f"largest_area_minus_{rival}"] = {"difference": round(auc("dcis_like_largest_mm2") - auc(rival), 3),
                                                 "range95": range95([auc("dcis_like_largest_mm2", i) - auc(rival, i) for i in draws])}
    rule = t["dcis_like_longest_mm"].to_numpy() >= 2.0
    result["two_mm_rule"] = {"dcis_slides_called_dcis": int((rule & is_dcis).sum()), "dcis_slides": int(is_dcis.sum()),
                             "adh_slides_called_dcis": int((rule & ~is_dcis).sum()), "adh_slides": int((~is_dcis).sum())}
    a = result["auc"]["dcis_like_largest_mm2"]["auc"]
    lo = result["largest_area_minus_old_baseline"]["range95"][0]
    result["verdict"] = "PASS" if a >= 0.75 and lo > 0 else "FAIL"
    result["table"] = t.to_dict("records")
    (OUT / "wsi_tile_scorer.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
