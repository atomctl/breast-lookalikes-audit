#!/usr/bin/env python3
"""Ledger entry 31: the frozen tile scorer on whole slides of the validation patients.

Usage (from the project folder):
  python -I tools/wsi_validate.py freeze   # train the entry-30 scorer on all 68 training slides -> outputs/grayzone/tile_scorer.joblib
  python -I tools/wsi_validate.py score    # score the validation slides -> outputs/grayzone/wsi_validate.json
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import wsi_tile_scorer as ts
from train_classifier import ROOT, range95, rows
from wsi_extent import TILE_MM, WSI, largest_group

OUT = ROOT / "outputs/grayzone"
MODEL = OUT / "tile_scorer.joblib"


def freeze():
    assert not MODEL.exists(), "already frozen"
    slides = ts.load()
    assert len(slides) == 68
    x, y = ts.training_tiles(slides, np.random.default_rng(0))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)).fit(x, y)
    joblib.dump(model, MODEL)
    print("frozen on", len(slides), "slides,", len(y), "tiles,", int(y.sum()), "of them DCIS")


def score():
    assert not (OUT / "wsi_validate.json").exists(), "entry 31 runs once"
    model = joblib.load(MODEL)
    info = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    who = dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))
    seen = {int(r["patient"]) for r in rows("train")} | {s["patient"] for s in ts.load()}
    table = []
    for remote in json.loads((OUT / "wsi_val17.json").read_text()):
        name = pathlib.Path(remote).stem
        z = np.load(WSI / f"{name}.npz", allow_pickle=True)
        assert int(who[name]) not in seen, f"{name}: patient also in training"
        prob = model.predict_proba(z["feats"].astype(np.float32))[:, 1]
        like = prob >= 0.5
        n, side = largest_group(z["coords"][like], int(z["step"]))
        table.append({"slide": name, "label": "DCIS" if "Type_DCIS" in remote else "ADH", "patient": int(who[name]), "tissue_tiles": int(len(like)),
                      "dcis_like_share": round(float(like.mean()), 4), "dcis_like_largest_mm2": round(n * TILE_MM ** 2, 3),
                      "dcis_like_longest_mm": round(side * TILE_MM, 2), "mean_probability": round(float(prob.mean()), 4)})
    t = pd.DataFrame(table)
    assert len(t) == 17
    is_dcis = (t["label"] == "DCIS").to_numpy()
    ids, pat = t["patient"].unique(), t["patient"].to_numpy()
    rng = np.random.default_rng(0)
    draws = []
    while len(draws) < 2000:
        i = np.concatenate([np.flatnonzero(pat == p) for p in rng.choice(ids, len(ids))])
        if is_dcis[i].any() and not is_dcis[i].all():
            draws.append(i)
    result = {"slides": 17, "adh_slides": int((~is_dcis).sum()), "dcis_slides": int(is_dcis.sum()), "patients": int(len(ids)), "auc": {}}
    for col in ["dcis_like_largest_mm2", "dcis_like_longest_mm", "dcis_like_share", "mean_probability"]:
        v = t[col].to_numpy(float)
        result["auc"][col] = {"auc": round(float(roc_auc_score(is_dcis, v)), 3), "range95": range95([roc_auc_score(is_dcis[i], v[i]) for i in draws]),
                              "median_adh": round(float(np.median(v[~is_dcis])), 3), "median_dcis": round(float(np.median(v[is_dcis])), 3)}
    rule = t["dcis_like_longest_mm"].to_numpy() >= 2.0
    result["two_mm_rule"] = {"dcis_slides_called_dcis": int((rule & is_dcis).sum()), "adh_slides_called_dcis": int((rule & ~is_dcis).sum())}
    result["verdict"] = "PASS" if result["auc"]["dcis_like_largest_mm2"]["auc"] >= 0.75 else "FAIL"
    result["table"] = table
    (OUT / "wsi_validate.json").write_text(json.dumps(result, indent=1))
    print(t.to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    {"freeze": freeze, "score": score}[sys.argv[1]]()
