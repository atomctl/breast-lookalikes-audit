#!/usr/bin/env python3
"""Ledger entry 15: the frozen system on another hospital's images (BACH, ICIAR 2018).

Usage (from the project folder):
  python -I tools/second_hospital.py selfcheck   # the image-to-features path reproduces saved BRACS features
  python -I tools/second_hospital.py unpack      # take the 400 photos out of the downloaded zip
  python -I tools/second_hospital.py embed       # images -> classifier probabilities (outputs/bach/prob.npy)
  python -I tools/second_hospital.py measure     # -> outputs/bach/result.json

Everything about how BACH is read and scored was fixed in entry 15 before any BACH image was opened.
BACH: Aresta et al. 2019, CC BY-NC-ND 4.0. Images are used as they are; none is redistributed.
"""
import csv
import json
import pathlib
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACH = ROOT / "data/bach"
OUT = ROOT / "outputs/bach"
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
TYPES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
TO_CLASS = np.array([0, 1, 1, 1, 1, 2, 3])  # N -> normal; PB, UDH, FEA, ADH -> benign; DCIS -> in situ; IC -> invasive
BACH_MPP, TARGET_MPP = 0.42, 0.5


class Embedder:
    """One region image -> the classifier's input, built exactly as for BRACS."""

    def __init__(self):
        import timm
        import torch
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        import extract_features as ef
        self.torch, self.ef = torch, ef
        model, norm, embed, _ = ef.load_model("uni")
        assert norm is None and embed is None
        cfg = timm.data.resolve_model_data_config(model)
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.eval().to(self.dev)
        self.mean = torch.tensor(cfg["mean"], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.dev).view(1, 3, 1, 1)
        self.whole = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def __call__(self, image_20x, original):
        """image_20x: the region at 0.5 um per pixel (uint8 array). original: the PIL image as delivered."""
        torch, ef = self.torch, self.ef
        a = image_20x
        h, w = a.shape[:2]
        if h < ef.TILE or w < ef.TILE:
            b = np.full((max(h, ef.TILE), max(w, ef.TILE), 3), 255, np.uint8)
            b[:h, :w] = a
            a = b
        tiles = np.stack([a[y:y + ef.TILE, x:x + ef.TILE] for y in ef.starts(a.shape[0]) for x in ef.starts(a.shape[1])])
        tissue = (tiles.sum(axis=3, dtype=np.uint16) < 660).mean(axis=(1, 2))
        feats = []
        with torch.inference_mode():
            for n in range(0, len(tiles), 64):
                x = torch.from_numpy(tiles[n:n + 64]).to(self.dev).permute(0, 3, 1, 2).float().div_(255)
                feats.append(self.model((x - self.mean) / self.std).float().cpu().numpy())
            view = self.model(self.whole(original)[None].to(self.dev)).float().cpu().numpy()[0]
        feats = np.concatenate(feats)
        keep = tissue >= 0.5
        return np.hstack([feats[keep].mean(0) if keep.any() else feats.mean(0), view])


def selfcheck():
    """The path used for BACH must reproduce what was saved for BRACS, on BRACS validation regions."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with open(ROOT / "data/regions.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    pick = [0, 50, 100, 150, 200, 250]
    saved = np.hstack([np.load(ROOT / "outputs/m1/pooled/uni_val_0.5.npy"), np.load(ROOT / "outputs/m1/thunder/uni_val.npy")])
    e = Embedder()
    worst = 1.0
    for i in pick:
        img = Image.open(ROOT / "data/bracs/BRACS_RoI/latest_version" / rows[i]["file"]).convert("RGB")
        x = e(np.asarray(img.reduce(2)), img)
        for part in (slice(0, 1024), slice(1024, 2048)):
            cos = float(x[part] @ saved[i][part] / np.linalg.norm(x[part]) / np.linalg.norm(saved[i][part]))
            worst = min(worst, cos)
    print("lowest cosine between rebuilt and saved BRACS features:", round(worst, 5))
    assert worst > 0.999


def photos():
    return sorted(p for p in (BACH / "extracted").rglob("*.tif") if p.parent.name in CLASSES)


def unpack():
    with zipfile.ZipFile(BACH / "ICIAR2018_BACH_Challenge.zip") as z:
        names = [n for n in z.namelist() if "/Photos/" in n and n.lower().endswith((".tif", ".csv"))]
        z.extractall(BACH / "extracted", members=names)
    found = photos()
    print(len(found), "photos;", {c: sum(p.parent.name == c for p in found) for c in CLASSES})


def embed():
    import joblib
    from PIL import Image
    found = photos()
    assert len(found) == 400, len(found)
    e = Embedder()
    x = []
    for n, p in enumerate(found):
        img = Image.open(p).convert("RGB")
        scale = BACH_MPP / TARGET_MPP
        small = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        x.append(e(np.asarray(small), img))
        if n % 40 == 0:
            print(n, "/", len(found), p.parent.name, img.size, flush=True)
    prob = joblib.load(ROOT / "outputs/m2/engine.joblib").predict_proba(np.array(x))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "prob.npy", prob)
    (OUT / "files.json").write_text(json.dumps([str(p.relative_to(BACH / "extracted")) for p in found]))


def measure():
    import patient_promise as pp
    prob = np.load(OUT / "prob.npy")
    files = json.loads((OUT / "files.json").read_text())
    label = np.array([CLASSES.index(pathlib.Path(f).parent.name) for f in files])
    act, lists = pp.apply_system(prob, np.arange(len(prob)))  # no patient numbers: every image is its own patient
    top = TO_CLASS[prob.argmax(1)]
    wrong = top != label
    hit = np.array([(TO_CLASS[np.flatnonzero(row)] == c).any() for row, c in zip(lists, label)])
    asked = ~act
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(label), (1000, len(label)))
    rng95 = lambda fn: [round(float(v), 4) for v in np.percentile([fn(i) for i in draws], [2.5, 97.5])]
    ratio = lambda a, b: float(a.sum() / max(b.sum(), 1))
    strict_wrong = wrong | np.isin(prob.argmax(1), [3, 4])  # stricter reading: an atypia answer (FEA, ADH) is always wrong on BACH
    result = {
        "images": len(label), "mapping": "N->Normal; PB, UDH, FEA, ADH->Benign; DCIS->InSitu; IC->Invasive",
        "classifier_accuracy_4_classes": round(float(1 - wrong.mean()), 4), "accuracy_range95": rng95(lambda i: 1 - wrong[i].mean()),
        "asked_share": round(float(asked.mean()), 4), "asked_share_range95": rng95(lambda i: asked[i].mean()),
        "answered_alone": int(act.sum()), "answered_alone_wrong": int((wrong & act).sum()),
        "error_among_answered_alone": round(ratio(wrong & act, act), 4),
        "error_among_answered_alone_range95": rng95(lambda i: ratio((wrong & act)[i], act[i])),
        "error_among_answered_alone_strict_mapping": round(ratio(strict_wrong & act, act), 4),
        "hit_rate_among_asked": round(ratio(hit & asked, asked), 4), "hit_rate_among_asked_range95": rng95(lambda i: ratio((hit & asked)[i], asked[i])),
        "list_length_among_asked": round(float(lists[asked].sum(1).mean()), 3) if asked.any() else None,
        "mean_top_probability": round(float(prob.max(1).mean()), 4),
        "by_class": {c: {"images": int((label == k).sum()), "accuracy": round(float(1 - wrong[label == k].mean()), 4),
                         "asked_share": round(float(asked[label == k].mean()), 4),
                         "error_among_answered_alone": round(ratio((wrong & act)[label == k], act[label == k]), 4),
                         "hit_rate_among_asked": round(ratio((hit & asked)[label == k], asked[label == k]), 4)} for k, c in enumerate(CLASSES)},
        "top_type_by_class": {c: {t: int(((prob.argmax(1) == j) & (label == k)).sum()) for j, t in enumerate(TYPES)} for k, c in enumerate(CLASSES)},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"selfcheck": selfcheck, "unpack": unpack, "embed": embed, "measure": measure}[sys.argv[1]]()
