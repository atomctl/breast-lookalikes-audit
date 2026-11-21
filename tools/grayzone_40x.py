#!/usr/bin/env python3
"""Ledger entry 26: do 40x features help the gray-zone specialist?

Usage (from the project folder):
  python -I tools/grayzone_40x.py extract   # UNI at 40x for gray-zone regions -> outputs/grayzone/uni40_<split>.npy
  python -I tools/grayzone_40x.py score     # -> outputs/grayzone/specialist_40x.json

Training and validation patients only. At most 64 tissue tiles per region (fixed seed).
"""
import json
import os
import pathlib
import sys

os.environ["BRACS_REDUCE"] = "1"  # 40x: read by extract_features when it is imported, also in the loader's workers
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_specialist as gs
import short_lists as sl
from train_classifier import classifier, patient_draws, range95, rows

OUT = gs.OUT
PER_REGION = 64


class Sampled:
    """At most PER_REGION tissue tiles of a region at 40x. Sampling happens in the worker, so little data crosses processes."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch

        import extract_features as ef
        tiles, coords, tissue, size, padded = ef.region_tiles(self.paths[i])
        keep = np.flatnonzero(tissue >= 0.5)
        keep = keep if len(keep) else np.arange(len(tiles))
        pick = np.sort(np.random.default_rng(i).choice(keep, min(PER_REGION, len(keep)), replace=False))
        return i, torch.from_numpy(tiles[pick]), len(keep)


def extract():
    import timm
    import torch

    import extract_features as ef
    assert ef.REDUCE == 1
    model, norm, embed, _ = ef.load_model("uni")
    cfg = timm.data.resolve_model_data_config(model)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.eval().to(dev).half()
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    for split in ("train", "val"):
        dest = OUT / f"uni40_{split}.npy"
        if dest.exists():
            continue
        held = [r for r in rows(split) if int(r["label"]) in gs.GRAY]
        paths = [sl.ROOT / "data/bracs/BRACS_RoI/latest_version" / r["file"] for r in held]
        loader = torch.utils.data.DataLoader(Sampled(paths), batch_size=None, shuffle=False, num_workers=4)
        x, total = [None] * len(held), 0
        for i, tiles, n_tissue in loader:
            with torch.inference_mode():
                t = tiles.to(dev).permute(0, 3, 1, 2).float().div_(255)
                x[i] = model(((t - mean) / std).half()).float().cpu().numpy().mean(0)
            total += len(tiles)
            if i % 200 == 0:
                print(split, i, "/", len(held), "tiles so far", total, flush=True)
        x = np.array(x)
        assert np.isfinite(x).all()
        np.save(dest, x)
        print(split, "done:", len(held), "regions,", total, "tiles", flush=True)


def score():
    from sklearn.metrics import f1_score
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    x, xv = sl.engine_features("train"), sl.engine_features("val")
    gray, grayv = np.isin(y, gs.GRAY), np.isin(vy, gs.GRAY)
    x40, x40v = np.load(OUT / "uni40_train.npy"), np.load(OUT / "uni40_val.npy")
    assert len(x40) == gray.sum() and len(x40v) == grayv.sum()
    xg, xgv = x[gray], xv[grayv]
    y4, y4v, pg = np.searchsorted(gs.GRAY, y[gray]), np.searchsorted(gs.GRAY, vy[grayv]), patient[gray]
    pos = np.cumsum(gray) - 1
    designs = {"B1 20x (reference)": (xg, xgv), "F40 40x tiles + whole view": (np.hstack([x40, xg[:, 1024:]]), np.hstack([x40v, xgv[:, 1024:]])),
               "F20+40 both + whole view": (np.hstack([xg[:, :1024], x40, xg[:, 1024:]]), np.hstack([xgv[:, :1024], x40v, xgv[:, 1024:]]))}
    b0 = prob[gray][:, gs.GRAY] / prob[gray][:, gs.GRAY].sum(1, keepdims=True)
    b0v = vprob[grayv][:, gs.GRAY] / vprob[grayv][:, gs.GRAY].sum(1, keepdims=True)
    draws = patient_draws(pg, np.random.default_rng(0), 1000)
    macro = lambda p, i=slice(None): f1_score(y4[i], p[i].argmax(1), average="macro")
    result = {"cross_validation": {"B0 current classifier": gs.scores(b0, y4)}, "validation": {"B0 current classifier": gs.scores(b0v, y4v)}}
    for name, (d, dv) in designs.items():
        best = None
        for c in gs.STRENGTHS:
            p4 = np.zeros((len(y4), 4))
            for a, b in folds:
                a, b = pos[a[gray[a]]], pos[b[gray[b]]]
                p4[b] = classifier(c).fit(d[a], y4[a]).predict_proba(d[b])
            if best is None or macro(p4) > best[0]:
                best = (macro(p4), c, p4)
        _, c, p4 = best
        diff = [macro(p4, i) - macro(b0, i) for i in draws]
        result["cross_validation"][name] = {"C": c, **gs.scores(p4, y4), "minus_B0": round(macro(p4) - macro(b0), 4), "range95": range95(diff),
                                            "adopted": bool(np.percentile(diff, 2.5) > 0) and not name.startswith("B1")}
        result["validation"][name] = gs.scores(classifier(c).fit(d, y4).predict_proba(dv), y4v)
        print(name, result["cross_validation"][name], flush=True)
    adopted = [k for k, v in result["cross_validation"].items() if v.get("adopted")]
    result["winner"] = max(adopted, key=lambda k: result["cross_validation"][k]["macro_f1"]) if adopted else "B0 current classifier"
    (OUT / "specialist_40x.json").write_text(json.dumps(result, indent=2))
    print("winner:", result["winner"])
    for k, v in result["validation"].items():
        print("VAL", k, v["macro_f1"], v["f1_by_type"], {a: b for a, b in v.items() if a.startswith("auc")})


if __name__ == "__main__":
    {"extract": extract, "score": score}[sys.argv[1]]()
