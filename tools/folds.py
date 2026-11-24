"""The patient folds, read from `outputs/folds.json` instead of re-derived in every script.

The split itself is fixed by `tools/patient_split.py` (one patient never straddles two parts).
The *folds* around that split used to be re-derived at every call site with
`StratifiedGroupKFold(n, shuffle=True, random_state=0)`, so they coincided only as long as the row
order, the labels, the patient table and the sklearn version all stayed the same. This module reads
the frozen assignment instead; `tools/folds_build.py` writes it.

A record is keyed by `n_splits` plus the labels and the patient of every row in order, so a call on
a subset (the inner folds inside one outer training part) is its own record and replays exactly.
A key that is not in the file is derived once, appended and saved, so a new call site self-heals
instead of silently inventing a split.

Usage:
    import folds
    for train, test in folds.split(5, y, patient):
        ...
"""
import hashlib
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs/folds.json"
_CACHE = None


def _key(n_splits, y, groups):
    """sha1 over the split count and every (label, patient) pair in row order."""
    h = hashlib.sha1()
    h.update(str(int(n_splits)).encode())
    h.update(np.asarray(y, np.int64).tobytes())
    h.update(np.asarray(groups, np.int64).tobytes())
    return h.hexdigest()


def load():
    """key -> {"n_splits", "rows", "patients", "fold_of"}; read once."""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(PATH.read_text())["records"] if PATH.exists() else {}
    return _CACHE


def save():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = {"about": "Frozen patient folds. Written by tools/folds_build.py; loaded by folds.split().",
            "key": "(n_splits, labels, patient) of every row in order, sha1",
            "records": {k: load()[k] for k in sorted(load())}}
    tmp = PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=1, sort_keys=True))
    tmp.replace(PATH)


def _derive(n_splits, y, groups):
    """The old behaviour, used only to fill a key that the file does not hold yet."""
    from sklearn.model_selection import StratifiedGroupKFold
    return [(np.asarray(a), np.asarray(b))
            for a, b in StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(y, y, groups)]


def _replay(n_splits, groups, fold_of):
    """The recorded patient -> fold map back into train/test index arrays, in fold order."""
    g = np.asarray(groups)
    out = []
    for k in range(n_splits):
        test = np.flatnonzero(np.isin(g, [int(p) for p, f in fold_of.items() if f == k]))
        train = np.flatnonzero(~np.isin(g, [int(p) for p, f in fold_of.items() if f == k]))
        out.append((train, test))
    return out


def split(n_splits, y, groups):
    """(train, test) index pairs, byte-identical to the derivation this module replaced."""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError(f"y and groups differ in length: {len(y)} vs {len(groups)}")
    k = _key(n_splits, y, groups)
    rec = load().get(k)
    covered = rec is not None and len(set(int(p) for p in groups) - set(map(int, rec["fold_of"]))) == 0
    if not covered or rec["rows"] != len(y):
        derived = _derive(n_splits, y, groups)
        fold_of = {}
        for f, (_, b) in enumerate(derived):
            for p in np.unique(groups[b]):
                fold_of[str(int(p))] = f
        rec = {"n_splits": int(n_splits), "rows": len(y), "patients": len(set(int(p) for p in groups)), "fold_of": fold_of}
        load()[k] = rec
        save()
    mine = _replay(n_splits, groups, rec["fold_of"])
    if os.environ.get("FOLDS_VERIFY"):  # FOLDS_VERIFY=1 re-derives live and refuses to return a drifted split
        for (la, lb), (ma, mb) in zip(_derive(n_splits, y, groups), mine):
            assert np.array_equal(la, ma) and np.array_equal(lb, mb), f"folds.json drift at {k}"
    return mine
