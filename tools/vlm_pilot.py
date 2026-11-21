#!/usr/bin/env python3
"""Ledger entry 43: a vision-and-language reader with criteria and labelled look-alikes, on gray-zone regions of the validation patients.

Usage (from the project folder):
  python -I tools/vlm_pilot.py build   # -> outputs/vlm_pilot/q_001.jpg ... (no labels of the cases, no key on disk)
  python -I tools/vlm_pilot.py score   # reads outputs/vlm_pilot/answers_*.json -> outputs/vlm_pilot/result.json
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import pooled, rows

OUT = sl.ROOT / "outputs/vlm_pilot"
SRC = sl.ROOT / "data/bracs/BRACS_RoI/latest_version"
GRAY, NAMES = [2, 3, 4, 5], ["UDH", "FEA", "ADH", "DCIS"]
PER_TYPE, SEED = 10, 43


def chosen():
    """The 40 cases, in the order of the pictures: indices into the validation rows. Deterministic, so no key file is needed."""
    val = rows("val")
    y = np.array([int(r["label"]) for r in val])
    rng = np.random.default_rng(SEED)
    pick = np.concatenate([rng.choice(np.flatnonzero(y == k), PER_TYPE, replace=False) for k in GRAY])
    return rng.permutation(pick), y


def fit(im, box):
    im = im.copy()
    s = min(box / im.width, box / im.height, 2.0)
    return im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.LANCZOS)


def build():
    Image.MAX_IMAGE_PIXELS = 200_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    order, _ = chosen()
    val, train = rows("val"), rows("train")
    ty = np.array([int(r["label"]) for r in train])
    tp = np.array([int(r["patient"]) for r in train])
    xv, xt = pooled("uni", "val", 0.5), pooled("uni", "train", 0.5)
    xv, xt = xv / np.linalg.norm(xv, axis=1, keepdims=True), xt / np.linalg.norm(xt, axis=1, keepdims=True)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    for n, i in enumerate(order, 1):
        im = Image.open(SRC / val[i]["file"]).convert("RGB").reduce(2)
        sheet = Image.new("RGB", (1540, 1640), "white")
        d = ImageDraw.Draw(sheet)
        d.text((10, 6), f"CASE q_{n:03d}: whole region (width {im.width * 0.0005:.2f} mm)", fill="black", font=font)
        d.text((780, 6), "CASE detail at full 20x resolution, shown 2x", fill="black", font=font)
        sheet.paste(fit(im, 750), (10, 44))
        a = np.asarray(im.convert("L"), np.float32)
        w = min(375, im.width, im.height)
        if im.width > w or im.height > w:  # the window with the most dark (cell-rich) pixels
            dark = (a < 150).astype(np.float32)
            c = np.pad(dark.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            s = c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]
            yy, xx = np.unravel_index(int(s.argmax()), s.shape)
        else:
            yy = xx = 0
        sheet.paste(im.crop((xx, yy, xx + w, yy + w)).resize((750, 750), Image.LANCZOS), (780, 44))
        d.text((10, 806), "REFERENCE regions with confirmed types (the most similar ones in the archive, two per type):", fill="black", font=font)
        sim = xt @ xv[i]
        col = 0
        for k, name in zip(GRAY, NAMES):
            cand = np.flatnonzero(ty == k)
            best, seen = [], set()
            for j in cand[np.argsort(-sim[cand])]:
                if tp[j] not in seen:
                    best.append(j); seen.add(tp[j])
                if len(best) == 2:
                    break
            for row, j in enumerate(best):
                ref = Image.open(SRC / train[j]["file"]).convert("RGB").reduce(2)
                x0, y0 = 10 + col * 382, 846 + row * 396
                d.text((x0, y0), f"{name}  ({ref.width * 0.0005:.2f} mm wide)", fill=(160, 0, 0), font=small)
                sheet.paste(fit(ref, 370), (x0, y0 + 24))
            col += 1
        sheet.save(OUT / f"q_{n:03d}.jpg", quality=88)
    print(len(order), "pictures written to", OUT)


def score():
    from sklearn.metrics import f1_score
    order, y = chosen()
    truth = np.searchsorted(GRAY, y[order])
    _, _, _, _, vprob, vy, _ = sl.load()
    assert (vy == y).all()
    eng = vprob[order][:, GRAY]
    eng = eng / eng.sum(1, keepdims=True)
    reader = np.full((len(order), 4), np.nan)
    for f in sorted(OUT.glob("answers_*.json")):
        for a in json.loads(f.read_text()):
            n = int(a["case"].split("_")[1]) - 1
            p = np.array([float(a["probs"][t]) for t in NAMES])
            reader[n] = p / p.sum()
    done = ~np.isnan(reader[:, 0])
    assert done.all(), f"missing answers for {np.flatnonzero(~done) + 1}"
    both = (reader + eng) / 2
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(truth), (2000, len(truth)))
    macro = lambda p, i=slice(None): float(f1_score(truth[i], p[i].argmax(1), average="macro", labels=range(4), zero_division=0))
    result = {"cases": int(len(truth))}
    for name, p in (("reader (vision-and-language model)", reader), ("adopted classifier", eng), ("average of the two", both)):
        pred = p.argmax(1)
        result[name] = {"accuracy": round(float((pred == truth).mean()), 3), "right": int((pred == truth).sum()), "gray_macro_f1": round(macro(p), 3),
                        "gray_macro_f1_range95": [round(float(v), 3) for v in np.percentile([macro(p, i) for i in idx], [2.5, 97.5])],
                        "right_by_type": {t: int(((pred == truth) & (truth == k)).sum()) for k, t in enumerate(NAMES)},
                        "answers_by_type": {t: int((pred == k).sum()) for k, t in enumerate(NAMES)}}
    r, e = reader.argmax(1) == truth, eng.argmax(1) == truth
    result["overlap"] = {"both_right": int((r & e).sum()), "only_reader_right": int((r & ~e).sum()), "only_classifier_right": int((~r & e).sum()), "both_wrong": int((~r & ~e).sum())}
    d = [macro(both, i) - macro(eng, i) for i in idx]
    result["average_minus_classifier"] = {"gray_macro_f1": round(macro(both) - macro(eng), 3), "range95": [round(float(v), 3) for v in np.percentile(d, [2.5, 97.5])]}
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
