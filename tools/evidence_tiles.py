#!/usr/bin/env python3
"""Ledger entry 21: tile-level look-alikes in UNI's own feature space.

Usage (from the project folder):
  python -I tools/evidence_tiles.py   # -> outputs/evidence/tiles.json (builds the tile library on first run)

Library: tissue tiles of the training patients. Queries: tiles of validation patients. No test data.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import FEATS, range95, rows

OUT = sl.ROOT / "outputs/evidence"


def tiles_of(file):
    z = np.load(FEATS / "uni" / (file[:-4] + ".npz"))
    keep = z["tissue"] >= 0.5
    f = z["feats"][keep].astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True), z["coords"][keep]


def library():
    """Unit-length UNI features of every tissue tile of the training patients, with region index and position."""
    cache = OUT / "tile_library.npz"
    if not cache.exists():
        feats, region, coords = [], [], []
        for i, r in enumerate(rows("train")):
            f, c = tiles_of(r["file"])
            feats.append(f.astype(np.float16)); region.append(np.full(len(f), i, np.int32)); coords.append(c)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache, feats=np.concatenate(feats), region=np.concatenate(region), coords=np.concatenate(coords))
    z = np.load(cache)
    return z["feats"].astype(np.float32), z["region"], z["coords"]


def nearest(query, lib, k=5, block=512):
    """Indices of the k most similar library tiles for each query tile."""
    out = np.zeros((len(query), k), np.int64)
    for n in range(0, len(query), block):
        sim = query[n:n + block] @ lib.T
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        out[n:n + block] = np.take_along_axis(top, np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1), axis=1)
    return out


def main():
    import joblib
    lib, lib_region, _ = library()
    train, held = rows("train"), rows("val")
    y_t = np.array([int(r["label"]) for r in train])
    lib_y = y_t[lib_region]
    final = joblib.load(sl.ROOT / "outputs/m2/engine.joblib")
    wrong_region = final.predict(sl.engine_features("val")) != np.array([int(r["label"]) for r in held])
    q, q_y, q_p, q_wrong = [], [], [], []
    for i, r in enumerate(held):
        f, _ = tiles_of(r["file"])
        q.append(f); q_y += [int(r["label"])] * len(f); q_p += [int(r["patient"])] * len(f); q_wrong += [bool(wrong_region[i])] * len(f)
    q, q_y, q_p, q_wrong = np.concatenate(q), np.array(q_y), np.array(q_p), np.array(q_wrong)
    pick = np.random.default_rng(0).choice(len(q), 3000, replace=False)
    q, q_y, q_p, q_wrong = q[pick], q_y[pick], q_p[pick], q_wrong[pick]
    nn = nearest(q, lib)
    share = (lib_y[nn] == q_y[:, None]).mean(1)
    random_share = np.bincount(lib_y, minlength=7)[q_y] / len(lib_y)
    ids = np.unique(q_p)
    where = {p: np.flatnonzero(q_p == p) for p in ids}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(1000)]
    lo, hi = range95([share[i].mean() for i in draws])
    result = {"library_tiles": int(len(lib)), "query_tiles": int(len(q)), "share_same_type_at_5": round(float(share.mean()), 4), "range95": [lo, hi],
              "random_share": round(float(random_share.mean()), 4),
              "by_type": {t: round(float(share[q_y == k].mean()), 3) for k, t in enumerate(sl.TYPES)},
              "where_the_classifier_is_wrong": {"tiles": int(q_wrong.sum()), "share_same_type_at_5": round(float(share[q_wrong].mean()), 4),
                                               "random_share": round(float(random_share[q_wrong].mean()), 4),
                                               "range95": range95([share[i][q_wrong[i]].mean() for i in draws])},
              "where_the_classifier_is_right": round(float(share[~q_wrong].mean()), 4)}
    result["shown"] = bool(result["share_same_type_at_5"] >= 2 * result["random_share"] and lo > result["random_share"])
    assert (nearest(lib[:20], lib[:500], k=1)[:, 0] == np.arange(20)).all()  # a tile is its own nearest neighbour
    (OUT / "tiles.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
