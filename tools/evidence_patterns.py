#!/usr/bin/env python3
"""Ledger entry 20: named patterns with a text-image model (QuiltNet-B-16).

Usage (from the project folder):
  python -I tools/evidence_patterns.py embed   # every region's tiles -> outputs/evidence/quilt/<split>.npz
  python -I tools/evidence_patterns.py score   # the pre-registered test -> outputs/evidence/patterns.json

The patterns and the types each is expected in were fixed in entry 20 before any tile was scored.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/evidence"
MODEL = "hf-hub:wisdomik/QuiltNet-B-16"
TYPES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
PATTERNS = {  # pattern -> types where it is expected to score highest (ledger entry 20)
    "normal terminal duct lobular unit": ["N"],
    "sclerosing adenosis": ["PB"],
    "apocrine metaplasia": ["PB"],
    "fibroadenoma": ["PB"],
    "usual ductal hyperplasia with streaming cells and slit-like spaces": ["UDH"],
    "columnar cell change with apical snouts": ["FEA"],
    "atypical ductal hyperplasia": ["ADH"],
    "cribriform architecture": ["ADH", "DCIS"],
    "comedo necrosis": ["DCIS"],
    "micropapillary architecture": ["ADH", "DCIS"],
    "invasive carcinoma infiltrating the stroma": ["IC"],
    "desmoplastic stroma": ["IC"],
}
TYPE_NAMES = ["normal breast tissue", "benign breast lesion", "usual ductal hyperplasia", "flat epithelial atypia",
              "atypical ductal hyperplasia", "ductal carcinoma in situ", "invasive carcinoma"]
TEMPLATES = ["a histopathology image of {}", "an H&E stained image of {}", "{}"]


def split_rows(split):
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def embed():
    import open_clip
    import torch

    import extract_features as ef
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.eval().to(dev)
    mean, std = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in (norm.mean, norm.std))
    (OUT / "quilt").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:  # embeddings only; nothing is scored against labels here
        dest = OUT / "quilt" / f"{split}.npz"
        if dest.exists():
            continue
        held = split_rows(split)
        paths = [ROOT / "data/bracs/BRACS_RoI/latest_version" / r["file"] for r in held]
        loader = torch.utils.data.DataLoader(ef.Regions(paths), batch_size=None, shuffle=False, num_workers=4)
        region_mean, tile_emb, tile_region = [None] * len(held), [], []
        for i, tiles, coords, tissue, size, padded in loader:
            out = []
            with torch.inference_mode():
                for n in range(0, len(tiles), 128):
                    x = tiles[n:n + 128].to(dev).permute(0, 3, 1, 2).float().div_(255)
                    e = model.encode_image((x - mean) / std)
                    out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            e = np.concatenate(out)
            keep = np.asarray(tissue) >= 0.5
            region_mean[i] = e[keep].mean(0) if keep.any() else e.mean(0)
            tile_emb.append(e.astype(np.float16))
            tile_region.append(np.full(len(e), i, np.int32))
            if i % 400 == 0:
                print(split, i, "/", len(held), flush=True)
        np.savez(dest, region_mean=np.array(region_mean), tile_emb=np.concatenate(tile_emb), tile_region=np.concatenate(tile_region))


def text_embeddings(phrases):
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    out = []
    with torch.inference_mode():
        for p in phrases:
            e = torch.nn.functional.normalize(model.encode_text(tok([t.format(p) for t in TEMPLATES])).float(), dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=0).numpy())
    return np.array(out)


def score():
    from sklearn.metrics import roc_auc_score
    held = split_rows("train") + split_rows("val")
    y = np.array([int(r["label"]) for r in held])
    patient = np.array([int(r["patient"]) for r in held])
    x = np.vstack([np.load(OUT / "quilt" / f"{s}.npz")["region_mean"] for s in ("train", "val")])
    text = text_embeddings(list(PATTERNS) + TYPE_NAMES)
    np.save(OUT / "quilt" / "text.npy", text)
    result = test_patterns(x @ text.T, y, patient)
    (OUT / "patterns.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "patterns"}, indent=2))


def test_patterns(sims, y, patient):
    """The pre-registered test of entry 20, for any model's region-to-text similarities."""
    from sklearn.metrics import roc_auc_score
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(300)]
    result = {"regions": len(y), "patterns": {}}
    for j, (name, expected) in enumerate(PATTERNS.items()):
        target = np.isin(y, [TYPES.index(t) for t in expected])
        auc = roc_auc_score(target, sims[:, j])
        boot = [roc_auc_score(target[i], sims[i, j]) for i in draws]
        lo, hi = (round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5]))
        means = [float(sims[y == k, j].mean()) for k in range(7)]
        result["patterns"][name] = {"expected": expected, "auc": round(float(auc), 3), "range95": [lo, hi],
                                    "type_with_highest_mean": TYPES[int(np.argmax(means))], "usable": bool(auc >= 0.65 and lo > 0.5)}
        print(name, result["patterns"][name], flush=True)
    result["usable_patterns"] = sum(v["usable"] for v in result["patterns"].values())
    result["in_viewer"] = bool(result["usable_patterns"] >= 6)
    zero = sims[:, len(PATTERNS):].argmax(1)
    result["zero_shot_accuracy_7_types"] = round(float((zero == y).mean()), 4)
    result["zero_shot_balanced_accuracy"] = round(float(np.mean([(zero[y == k] == k).mean() for k in range(7)])), 4)
    return result


if __name__ == "__main__":
    {"embed": embed, "score": score}[sys.argv[1]]()
