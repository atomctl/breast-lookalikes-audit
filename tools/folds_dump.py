"""Dump every fold assignment the tools derive, one block per call site, so the
before/after pair can be diffed. Read-only: nothing is written outside the file named on argv.

Usage (from the project folder):
    .venv/bin/python -I tools/folds_dump.py <out.json>

Shape A is the five patient folds; B, C and D are the inner splits the gray-zone tools take
inside each outer training part.
"""
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

USE_FILE = (ROOT / "tools/folds.py").exists()
if USE_FILE:
    import folds as F


def spl(n, y, p):
    """The split, from the file once it exists, else re-derived the way the tools did before."""
    if USE_FILE:
        return list(F.split(n, y, p))
    from sklearn.model_selection import StratifiedGroupKFold
    return [(np.asarray(a), np.asarray(b)) for a, b in StratifiedGroupKFold(n, shuffle=True, random_state=0).split(y, y, p)]


def record(n, y, p):
    """What must stay equal: per fold, the sorted held-out patients and the held-out row count."""
    out = []
    for i, (a, b) in enumerate(spl(n, y, p)):
        out.append([i, int(len(a)), int(len(b)), sorted(int(v) for v in np.unique(p[b]))])
    assert sorted(np.concatenate([np.asarray(b) for _, b in spl(n, y, p)]).tolist()) == list(range(len(y))), "test folds do not cover every row once"
    return out


def train_rows():
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "train"]


def main(out):
    train = train_rows()
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    dump = {"folds_module_present": USE_FILE, "sites": {}}

    # shape A: the five M1/patient folds on the whole training table
    # train_classifier.cv, improve_classifier.baseline/whole, short_lists.load,
    # within_slide.load, within_slide.centred, wsi_ring_context.main
    for site in ["train_classifier.cv", "improve_classifier.baseline", "improve_classifier.whole", "short_lists.load",
                 "within_slide.load", "within_slide.centred", "wsi_ring_context.main"]:
        dump["sites"][site] = {"n_splits": 5, "rows": int(len(y)), "folds": record(5, y, patient)}

    outer = spl(5, y, patient)
    dump["sites"]["short_lists.load"]["outer_folds"] = np.asarray(y).__len__() and dump["sites"]["short_lists.load"]["folds"]

    # shape B: the inner split inside one outer training part, full label set
    # grayzone_features.out_of_fold, wsi_ring_context.out_of_fold, grayzone_label_noise.inner_oof,
    # grayzone_finetune.main (best_c)
    gray = np.isin(y, [2, 3, 4, 5])
    y4 = np.searchsorted([2, 3, 4, 5], y[gray])
    pg = patient[gray]
    import duct_cells as dc
    rows_gray = dc.gray_rows()
    cells, adh = [], []
    for r in rows_gray:
        path = dc.CELLS_DIR / (dc.key(r) + ".npz")
        cells.append(path.exists() and dc.numbers(np.load(path)) is not None)  # fewer than MIN_NUCLEI nuclei is dropped
    measured = np.array(cells)
    print("gray rows:", len(rows_gray), "measured:", int(measured.sum()), flush=True)
    base_gray = np.isin(y, [2, 3, 4, 5])
    udh_adh = np.isin(y, [2, 4])
    for k, (a, b) in enumerate(outer):
        inner = {}
        inner["grayzone_features.out_of_fold / wsi_ring_context.out_of_fold"] = record(4, y[a], patient[a])
        inner["grayzone_label_noise.inner_oof / grayzone_finetune.main"] = record(4, y[a], patient[a])
        dump["sites"][f"inner.full.fold{k}"] = {"n_splits": 4, "rows": int(len(a)), "folds": inner["grayzone_features.out_of_fold / wsi_ring_context.out_of_fold"]}
        # shape C: same, on the gray-zone rows with GRAY positions as labels (grayzone_specialist.stack)
        ga = a[gray[a]]
        pos = np.cumsum(gray) - 1
        pg_a, y4_a = pg[pos[ga] - pos[ga][0] + pos[ga][0] - pos[ga] * 0], y4[pos[ga]] if False else y4[pos[ga]]
        dump["sites"][f"inner.gray.fold{k}"] = {"n_splits": 4, "rows": int(len(ga)), "folds": record(4, y4[pos[ga]], pg[pos[ga]])}
        # shape D: the UDH/ADH measured rows (duct_cells.pair_score)
        mk = udh_adh[a]
        sub = np.flatnonzero(mk)
        sub = sub[measured[sub]]
        dump["sites"][f"inner.udh_adh_measured.fold{k}"] = {"n_splits": 4, "rows": int(len(sub)),
                                                            "folds": record(4, (y[a][sub] == 4).astype(int), patient[a][sub])}
    del y4_a, pg_a
    pathlib.Path(out).write_text(json.dumps(dump, indent=1, sort_keys=True))
    print("sites:", len(dump["sites"]), "-> ", out)


if __name__ == "__main__":
    main(sys.argv[1])
