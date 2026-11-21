#!/usr/bin/env python3
"""Ledger entry 32: where the frozen tile scorer's DCIS-like tiles sit on the validation slides.

Usage (from the project folder):
  python -I tools/wsi_diagnose_val.py   # -> outputs/grayzone/wsi_diagnose_val.json
"""
import collections
import json
import pathlib
import sys

import joblib
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
from train_classifier import ROOT
from wsi_extent import WSI

OUT = ROOT / "outputs/grayzone"


def main():
    model = joblib.load(OUT / "tile_scorer.joblib")
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    where = collections.Counter()   # on ADH slides: where the DCIS-like tiles lie
    inside = {"ADH": [0, 0], "DCIS": [0, 0]}  # DCIS-like tiles, all tiles, inside marked regions of that type
    p_dcis, p_adh, patches, placed = [], [], {}, [0, 0]
    for remote in json.loads((OUT / "wsi_val17.json").read_text()):
        name = pathlib.Path(remote).stem
        label = "DCIS" if "Type_DCIS" in remote else "ADH"
        z = np.load(WSI / f"{name}.npz", allow_pickle=True)
        prob = model.predict_proba(z["feats"].astype(np.float32))[:, 1]
        like = prob >= 0.5
        step = int(z["step"])
        centre = z["coords"] + step / 2
        mark = np.full(len(prob), "unmarked", dtype=object)
        placed[0] += len(regions[name]["matched"]); placed[1] += regions[name]["regions"]
        for r in regions[name]["matched"]:
            x0, y0, x1, y1 = r["box"]
            mark[(centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)] = r["label"]
        for k in inside:
            inside[k][0] += int(like[mark == k].sum()); inside[k][1] += int((mark == k).sum())
        p_dcis.append(prob[mark == "DCIS"]); p_adh.append(prob[mark == "ADH"])
        if label == "ADH":
            where.update(mark[like].tolist())
            g = (z["coords"][like] // step).astype(int)
            if len(g):
                g -= g.min(0)
                grid = np.zeros(g.max(0) + 1, bool); grid[g[:, 0], g[:, 1]] = True
                lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
                best = int(np.argmax(ndimage.sum(grid, lab, range(1, n + 1)))) + 1
                in_best = lab[g[:, 0], g[:, 1]] == best
                patches[name] = {"tiles": int(in_best.sum()), "made_of": dict(collections.Counter(mark[like][in_best].tolist())),
                                 "mean_probability_in_patch": round(float(prob[like][in_best].mean()), 3)}
    p_dcis, p_adh = np.concatenate(p_dcis), np.concatenate(p_adh)
    total = sum(where.values())
    result = {"regions_placed": placed,
              "1_dcis_like_tiles_on_adh_slides": total, "1_where": {k: round(v / total, 4) for k, v in where.most_common()},
              "2_share_dcis_like_inside_marked_adh": round(inside["ADH"][0] / inside["ADH"][1], 4), "2_tiles_in_marked_adh": inside["ADH"][1],
              "2_share_dcis_like_inside_marked_dcis": round(inside["DCIS"][0] / inside["DCIS"][1], 4), "2_tiles_in_marked_dcis": inside["DCIS"][1],
              "3_tile_auc_marked_dcis_vs_marked_adh": round(float(roc_auc_score(np.r_[np.ones(len(p_dcis)), np.zeros(len(p_adh))], np.r_[p_dcis, p_adh])), 3),
              "4_largest_patch_per_adh_slide": patches}
    (OUT / "wsi_diagnose_val.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
