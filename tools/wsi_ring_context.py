#!/usr/bin/env python3
"""Ledger entry 44: does the tissue just around a marked region, read from the whole slide, help to name the region?

Usage (from the project folder):
  python -I tools/wsi_ring_context.py   # -> outputs/grayzone/ring_context.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_features as gf
from train_classifier import ROOT, STRENGTHS, TYPES, classifier, patient_draws, range95
from wsi_extent import WSI
import folds as foldlib

OUT = ROOT / "outputs/grayzone"
RINGS_MM = [0.5, 1.5]


def load():
    info = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    who = dict(zip(info["WSI Filename"].str.strip(), info["Patient Id"]))
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    inside, rings, y, patient = [], {r: [] for r in RINGS_MM}, [], []
    for remote in json.loads((OUT / "wsi_all68.json").read_text()) + json.loads((OUT / "wsi_val17.json").read_text()):
        name = pathlib.Path(remote).stem
        z = np.load(WSI / f"{name}.npz", allow_pickle=True)
        feats, step, mpp = z["feats"].astype(np.float32), int(z["step"]), float(z["mpp"])
        c = z["coords"] + step / 2
        for r in regions[name]["matched"]:
            x0, y0, x1, y1 = r["box"]
            box = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
            if not box.any():
                continue
            bands = []
            for mm in RINGS_MM:
                pad = mm * 1000 / mpp
                band = (c[:, 0] >= x0 - pad) & (c[:, 0] < x1 + pad) & (c[:, 1] >= y0 - pad) & (c[:, 1] < y1 + pad) & ~box
                bands.append(feats[band].mean(0) if band.any() else None)
            if any(b is None for b in bands):
                continue
            inside.append(feats[box].mean(0)); y.append(TYPES.index(r["label"])); patient.append(int(who[name]))
            for mm, b in zip(RINGS_MM, bands):
                rings[mm].append(b)
    return np.array(inside), {k: np.array(v) for k, v in rings.items()}, np.array(y), np.array(patient)


def out_of_fold(x, y, patient, folds):
    """As grayzone_features.out_of_fold, for data where not all 7 types occur."""
    prob = np.zeros((len(y), 7))
    for a, b in folds:
        inner = list(foldlib.split(4, y[a], patient[a]))
        best = None
        for cc in STRENGTHS:
            p = np.zeros((len(a), 7))
            for ia, ib in inner:
                m = classifier(cc).fit(x[a][ia], y[a][ia])
                p[np.ix_(ib, m.classes_)] = m.predict_proba(x[a][ib])
            s = gf.gray_f1(p, y[a])[0]
            if best is None or s > best[0]:
                best = (s, cc)
        m = classifier(best[1]).fit(x[a], y[a])
        prob[np.ix_(b, m.classes_)] = m.predict_proba(x[b])
    return prob


def main():
    inside, rings, y, patient = load()
    folds = list(foldlib.split(5, y, patient))
    sets = {"B inside the region": inside, "B + ring 0.5 mm": np.hstack([inside, rings[0.5]]), "B + ring 1.5 mm": np.hstack([inside, rings[1.5]]),
            "B + both rings": np.hstack([inside, rings[0.5], rings[1.5]]), "ring 0.5 mm alone": rings[0.5], "ring 1.5 mm alone": rings[1.5]}
    draws = patient_draws(patient, np.random.default_rng(0))
    result = {"regions": int(len(y)), "patients": int(len(np.unique(patient))), "by_type": {TYPES[k]: int((y == k).sum()) for k in range(7) if (y == k).any()}}
    probs = {}
    for name, x in sets.items():
        p = probs[name] = out_of_fold(x, y, patient, folds)
        macro, adh = gf.gray_f1(p, y)
        r = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "pair_auc": {}}
        for pn, (u, v) in gf.PAIRS.items():
            m = np.isin(y, [u, v])
            r["pair_auc"][pn] = round(float(roc_auc_score(y[m] == v, p[m, v] / (p[m, u] + p[m, v] + 1e-12))), 3)
        if name != "B inside the region":
            base = probs["B inside the region"]
            d = [gf.gray_f1(p, y, i)[0] - gf.gray_f1(base, y, i)[0] for i in draws]
            e = [gf.gray_f1(p, y, i)[1] - gf.gray_f1(base, y, i)[1] for i in draws]
            r.update({"gray_macro_f1_minus_B": round(macro - result["B inside the region"]["gray_macro_f1"], 4), "range95_of_difference": range95(d),
                      "adh_f1_minus_B": round(adh - result["B inside the region"]["adh_f1"], 4), "adh_range95_of_difference": range95(e)})
            if name.startswith("B +"):
                r["adopted"] = bool(np.percentile(d, 2.5) > 0)
        result[name] = r
        print(name, r, flush=True)
    (OUT / "ring_context.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
