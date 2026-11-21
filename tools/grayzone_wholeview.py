#!/usr/bin/env python3
"""Ledger entry 40: show UNI the whole region at 448 and 672 pixels a side, next to the adopted 224-pixel view.

Usage (from the project folder):
  python -I tools/grayzone_wholeview.py   # -> outputs/grayzone/wholeview.json (views cached in outputs/grayzone/whole_uni_<size>_train.npy)
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_features as gf
import short_lists as sl
from train_classifier import patient_draws, range95, rows, weighted_f1

SRC = sl.ROOT / "data/bracs/BRACS_RoI/latest_version"
OUT = sl.ROOT / "outputs/grayzone"


class Whole(torch.utils.data.Dataset):
    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 200_000_000
        im = Image.open(SRC / self.files[i]).convert("RGB")
        side = max(im.size)
        sq = Image.new("RGB", (side, side), "white")  # pad to a square: nothing is cropped
        sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        return torch.from_numpy(np.asarray(sq.resize((self.size, self.size), Image.LANCZOS)).copy())


def view(size):
    cache = OUT / f"whole_uni_{size}_train.npy"
    if cache.exists():
        return np.load(cache)
    import extract_features as ef
    model, _, _, _ = ef.load_model("uni")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.eval().to(dev).half()
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    files = [r["file"] for r in rows("train")]
    loader = torch.utils.data.DataLoader(Whole(files, size), batch_size=8 if size <= 448 else 2, num_workers=6)
    x = []
    with torch.inference_mode():
        for n, b in enumerate(loader):
            b = b.to(dev).permute(0, 3, 1, 2).float().div_(255)
            x.append(model(((b - mean) / std).half()).float().cpu().numpy())
            if n % 100 == 0:
                print(size, len(x) * len(b), "/", len(files), flush=True)
    x = np.concatenate(x)
    assert np.isfinite(x).all() and len(x) == len(files)
    np.save(cache, x)
    return x


def main():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    e = sl.engine_features("train")
    w448, w672 = view(448), view(672)
    sets = {"E adopted inputs": e, "E + view 448": np.hstack([e, w448]), "E + view 672": np.hstack([e, w672]), "E + both views": np.hstack([e, w448, w672])}
    draws = patient_draws(patient, np.random.default_rng(0))
    result, probs = {}, {}
    for name, x in sets.items():
        probs[name], chosen = gf.out_of_fold(x, y, patient, folds)
        macro, adh = gf.gray_f1(probs[name], y)
        result[name] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "weighted_f1_7_types": round(weighted_f1(y, probs[name].argmax(1)), 4), "C_per_fold": chosen}
        if name != "E adopted inputs":
            d = [gf.gray_f1(probs[name], y, i)[0] - gf.gray_f1(probs["E adopted inputs"], y, i)[0] for i in draws]
            a = [gf.gray_f1(probs[name], y, i)[1] - gf.gray_f1(probs["E adopted inputs"], y, i)[1] for i in draws]
            result[name].update({"gray_macro_f1_minus_E": round(macro - result["E adopted inputs"]["gray_macro_f1"], 4), "range95_of_difference": range95(d),
                                 "adh_f1_minus_E": round(adh - result["E adopted inputs"]["adh_f1"], 4), "adh_range95_of_difference": range95(a),
                                 "adopted": bool(np.percentile(d, 2.5) > 0)})
        print(name, result[name], flush=True)
    (OUT / "wholeview.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
