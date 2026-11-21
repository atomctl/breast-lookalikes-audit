#!/usr/bin/env python3
"""Duct instrument, step 2 (ledger entry 41): tissue-kind maps for every training region, measurements per region, and their test.

Usage (from the project folder):
  python -I tools/duct_measure.py maps   # -> data/features/duct_maps/<region>.npy (one tissue kind per 16-pixel patch)
  python -I tools/duct_measure.py test   # -> outputs/ducts/measure.json, outputs/ducts/measurements_train.npy
"""
import json
import pathlib
import sys

import joblib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
from train_classifier import ROOT, rows

OUT = ROOT / "outputs/ducts"
MAPS = ROOT / "data/features/duct_maps"
SRC = ROOT / "data/bracs/BRACS_RoI/latest_version"
STROMA, RIM, SPACE, CELLS = [2, 4], [1], [5, 6], [0, 3, 7, 8, 9]  # named from the montage and two overlays (entry 41)
MM = 0.008  # one patch: 16 pixels at 0.5 micrometres per pixel
NAMES = ["ducts", "cell_area_mm2", "largest_duct_mm", "fullness_mean", "fullness_largest", "ducts_90pct_full", "spaces", "space_roundness",
         "space_size_spread", "space_share", "thickness_mean_mm", "thickness_max_mm"] + [f"kind_{k}" for k in range(10)] + [f"cells_kind_{k}" for k in CELLS]


def maps():
    import duct_tokens as dt
    kinds = joblib.load(OUT / "kinds.joblib")
    tok = dt.Tokens()
    MAPS.mkdir(parents=True, exist_ok=True)
    train = rows("train")
    for n, r in enumerate(train):
        dest = MAPS / (r["file"][:-4].replace("/", "__") + ".npy")
        if not dest.exists():
            _, m = dt.region_map(tok, kinds, SRC / r["file"])
            np.save(dest, m.astype(np.int8))
        if n % 300 == 0:
            print(n, "/", len(train), flush=True)


def measure(m):
    """26 numbers from one map of tissue kinds."""
    seen = m >= 0
    cells, space = np.isin(m, CELLS), np.isin(m, SPACE)
    # open space enclosed by duct cells: space components that do not touch the border and are mostly bordered by cells or rim
    lab, n = ndimage.label(space)
    enclosed = np.zeros_like(space)
    areas, rounds = [], []
    border = np.zeros_like(space); border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if (border[sl] & comp).any():
            continue
        ring = ndimage.binary_dilation(comp, iterations=1) & ~comp
        grown = np.zeros_like(space); grown[sl] = comp
        ring = ndimage.binary_dilation(grown) & ~grown
        if np.isin(m[ring], CELLS + RIM).mean() < 0.7:
            continue
        enclosed |= grown
        a = comp.sum()
        per = (ndimage.binary_dilation(grown) & ~grown).sum()
        areas.append(a); rounds.append(min(4 * np.pi * a / max(per, 1) ** 2, 1.0))
    duct = ndimage.binary_closing(cells | enclosed, iterations=1) | cells | enclosed
    dl, nd = ndimage.label(duct, structure=np.ones((3, 3)))
    sizes = ndimage.sum(duct, dl, range(1, nd + 1)) if nd else np.array([])
    keep = [k + 1 for k in range(nd) if sizes[k] >= 12]  # ducts under 12 patches (about 770 square micrometres) are specks
    full, diam = [], []
    for k in keep:
        sl = ndimage.find_objects(dl)[k - 1]
        comp = dl[sl] == k
        c, e = (cells[sl] & comp).sum(), (enclosed[sl] & comp).sum()
        full.append(c / max(c + e, 1))
        diam.append(max(comp.shape) * MM)
    big = int(np.argmax([sizes[k - 1] for k in keep])) if keep else 0
    thick = ndimage.distance_transform_edt(cells) * 2 * MM
    peaks = thick[cells & (thick == ndimage.maximum_filter(thick, 5))]
    total = max(seen.sum(), 1)
    out = [len(keep), cells.sum() * MM ** 2, max(diam, default=0.0), float(np.mean(full)) if full else 0.0, full[big] if full else 0.0,
           sum(f >= 0.9 for f in full), len(areas), float(np.mean(rounds)) if rounds else 0.0,
           float(np.std(areas) / np.mean(areas)) if len(areas) > 1 else 0.0, enclosed.sum() / max(duct.sum(), 1),
           float(peaks.mean()) if len(peaks) else 0.0, float(thick.max())]
    out += [(m == k).sum() / total for k in range(10)]
    out += [(m == k).sum() / max(cells.sum(), 1) for k in CELLS]
    return np.array(out, np.float32)


def test():
    from sklearn.metrics import roc_auc_score
    import grayzone_features as gf
    import short_lists as sl
    from train_classifier import patient_draws, range95, weighted_f1
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    x = np.array([measure(np.load(MAPS / (r["file"][:-4].replace("/", "__") + ".npy")).astype(int)) for r in train])
    assert np.isfinite(x).all() and x.shape[1] == len(NAMES)
    np.save(OUT / "measurements_train.npy", x)
    _, _, _, folds, *_ = sl.load()
    e = sl.engine_features("train")
    draws = patient_draws(patient, np.random.default_rng(0))
    result, probs = {"regions": len(y), "measurements": NAMES}, {}
    xl = np.hstack([x, np.log1p(x[:, :3])])
    for name, feats in (("E adopted inputs", e), ("measurements alone", xl), ("E + measurements", np.hstack([e, xl]))):
        probs[name], chosen = gf.out_of_fold(feats, y, patient, folds)
        macro, adh = gf.gray_f1(probs[name], y)
        result[name] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "weighted_f1_7_types": round(weighted_f1(y, probs[name].argmax(1)), 4), "C_per_fold": chosen}
        print(name, result[name], flush=True)
    d = [gf.gray_f1(probs["E + measurements"], y, i)[0] - gf.gray_f1(probs["E adopted inputs"], y, i)[0] for i in draws]
    result["E + measurements"].update({"gray_macro_f1_minus_E": round(result["E + measurements"]["gray_macro_f1"] - result["E adopted inputs"]["gray_macro_f1"], 4),
                                       "range95_of_difference": range95(d), "adopted": bool(np.percentile(d, 2.5) > 0)})
    T = sl.TYPES
    result["by_type_medians"] = {n: {T[k]: round(float(np.median(x[y == k, j])), 3) for k in range(7)} for j, n in enumerate(NAMES[:12])}
    pairs = {"fullness_largest: DCIS above ADH": ("fullness_largest", 5, 4), "fullness_mean: DCIS above ADH": ("fullness_mean", 5, 4),
             "space_roundness: ADH above UDH": ("space_roundness", 4, 2), "thickness_max_mm: ADH above FEA": ("thickness_max_mm", 4, 3),
             "thickness_max_mm: DCIS above ADH": ("thickness_max_mm", 5, 4), "largest_duct_mm: DCIS above ADH": ("largest_duct_mm", 5, 4)}
    result["single_measurements_auc"] = {}
    for label, (n, hi, lo) in pairs.items():
        mk = np.isin(y, [hi, lo])
        result["single_measurements_auc"][label] = round(float(roc_auc_score(y[mk] == hi, x[mk, NAMES.index(n)])), 3)
    (OUT / "measure.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("E + measurements", "by_type_medians", "single_measurements_auc")}, indent=1))


if __name__ == "__main__":
    {"maps": maps, "test": test}[sys.argv[1]]()
