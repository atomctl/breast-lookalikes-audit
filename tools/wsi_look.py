#!/usr/bin/env python3
"""Look at what the frozen tile scorer flags as DCIS-like on validation ADH slides (after ledger entry 32).

Usage (from the project folder; the slide files must be in data/bracs_wsi):
  python -I tools/wsi_look.py BRACS_1797 BRACS_1977   # -> outputs/grayzone/look/*.png and look.json

Pictures: an overview per slide (flagged tiles, the largest patch, the marked regions), the 40 most confident
flagged tiles per slide, and 6 groups of flagged tiles with KEEP's nearest descriptions. Real tissue: local only.
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import KMeans

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import evidence_patterns as ep
import evidence_patterns_keep as kp
from train_classifier import ROOT
from wsi_extent import WSI

OUT = ROOT / "outputs/grayzone"
LOOK = OUT / "look"
SVS = ROOT / "data/bracs_wsi"
EXTRA = ["adipose tissue", "dense fibrous stroma", "lymphocytic infiltrate", "blood and hemorrhage", "skin with epidermis",
         "out-of-focus or folded tissue artifact", "pen ink marking", "necrosis", "lobular carcinoma", "intraductal papilloma",
         "apocrine cyst lining", "lactational change"]  # descriptions never tested on BRACS: a hint, not evidence


def scored(name, model, regions):
    z = np.load(WSI / f"{name}.npz", allow_pickle=True)
    prob = model.predict_proba(z["feats"].astype(np.float32))[:, 1]
    step = int(z["step"])
    centre = z["coords"] + step / 2
    marked = np.zeros(len(prob), bool)
    for r in regions[name]["matched"]:
        x0, y0, x1, y1 = r["box"]
        marked |= (centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)
    return z, prob, step, marked


def tile(slide, xy, step):
    return slide.read_region((int(xy[0]), int(xy[1])), 0, (step, step)).convert("RGB").resize((224, 224), Image.BOX)


def sheet(tiles, cols, path):
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (cols * 226, rows * 226), "white")
    for i, t in enumerate(tiles):
        im.paste(t, ((i % cols) * 226, (i // cols) * 226))
    im.save(path)


def main():
    import tiffslide
    names = sys.argv[1:]
    LOOK.mkdir(parents=True, exist_ok=True)
    model = joblib.load(OUT / "tile_scorer.joblib")
    regions = json.loads((OUT / "wsi_regions.json").read_text())
    report = {"slides": {}}
    slides = {}
    for name in names:
        z, prob, step, marked = scored(name, model, regions)
        slide = slides[name] = tiffslide.TiffSlide(SVS / f"{name}.svs")
        like = prob >= 0.5
        level = slide.get_best_level_for_downsample(32)
        lw, lh = slide.level_dimensions[level]
        scale = slide.dimensions[0] / lw
        small = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
        over = Image.new("RGBA", small.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for (x, y) in z["coords"][like]:
            d.rectangle([x / scale, y / scale, (x + step) / scale, (y + step) / scale], fill=(201, 114, 10, 110))
        g = (z["coords"][like] // step).astype(int)
        lo = g.min(0)
        grid = np.zeros(g.max(0) - lo + 1, bool)
        grid[g[:, 0] - lo[0], g[:, 1] - lo[1]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        best = int(np.argmax(ndimage.sum(grid, lab, range(1, n + 1)))) + 1
        xs, ys = np.nonzero(lab == best)
        box = [(xs.min() + lo[0]) * step / scale, (ys.min() + lo[1]) * step / scale, (xs.max() + lo[0] + 1) * step / scale, (ys.max() + lo[1] + 1) * step / scale]
        d.rectangle(box, outline=(180, 0, 0, 255), width=4)
        for r in regions[name]["matched"]:
            d.rectangle([v / scale for v in r["box"]], outline=(0, 120, 115, 255), width=4)
        pic = Image.alpha_composite(small.convert("RGBA"), over).convert("RGB")
        pic.thumbnail((1800, 1800))
        pic.save(LOOK / f"{name}_overview.png")
        crop = small.crop([int(v) for v in (box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40)])
        crop.thumbnail((1600, 1600))
        crop.save(LOOK / f"{name}_patch_plain.png")
        top = np.argsort(-np.where(like & ~marked, prob, -1))[:40]
        sheet([tile(slide, z["coords"][i], step) for i in top], 8, LOOK / f"{name}_top40.png")
        report["slides"][name] = {"tissue_tiles": int(len(prob)), "flagged": int(like.sum()), "flagged_unmarked": int((like & ~marked).sum()),
                                  "largest_patch_tiles": int(len(xs)), "marked_regions": [r["label"] for r in regions[name]["matched"]]}

    # groups of flagged, unmarked tiles over all 8 validation ADH slides
    feats, src = [], []
    for remote in json.loads((OUT / "wsi_val17.json").read_text()):
        if "Type_ADH" not in remote:
            continue
        name = pathlib.Path(remote).stem
        z, prob, step, marked = scored(name, model, regions)
        for i in np.flatnonzero((prob >= 0.5) & ~marked):
            feats.append(z["feats"][i].astype(np.float32))
            src.append((name, int(z["coords"][i][0]), int(z["coords"][i][1]), step))
    x = np.array(feats)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    km = KMeans(6, n_init=10, random_state=0).fit(x)
    keep, tok = kp.load_keep()
    with torch.inference_mode():
        words = list(ep.PATTERNS) + ep.TYPE_NAMES + EXTRA
        text = torch.stack([torch.nn.functional.normalize(keep.encode_text(tok([s.format(p) for s in ep.TEMPLATES], max_length=256, padding="max_length",
                                                                               truncation=True, return_tensors="pt")).mean(0), dim=0) for p in words])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    keep = keep.to(dev)
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    tested = {k for k, v in json.loads((ROOT / "outputs/evidence/patterns_keep.json").read_text())["patterns"].items() if v["usable"]}
    report["groups"] = []
    all_tiles = []
    for c in np.argsort(-np.bincount(km.labels_)):
        members = np.flatnonzero(km.labels_ == c)
        near = members[np.argsort(-(x[members] @ km.cluster_centers_[c]))]
        shown = [i for i in near if src[i][0] in slides][:12]
        tiles = [tile(slides[src[i][0]], src[i][1:3], src[i][3]) for i in shown]
        all_tiles += tiles + [Image.new("RGB", (224, 224), "white")] * (12 - len(tiles))
        group = {"tiles": int(len(members)), "share": round(len(members) / len(x), 3),
                 "by_slide": {n: int(sum(src[i][0] == n for i in members)) for n in sorted({s[0] for s in src})}, "shown": len(tiles)}
        if tiles:
            with torch.inference_mode():
                t = torch.from_numpy(np.stack([np.asarray(im) for im in tiles])).to(dev).permute(0, 3, 1, 2).float().div_(255)
                sim = (keep.encode_image((t - mean) / std).float().cpu().mean(0) @ text.T).numpy()
            group["keep_nearest"] = [{"words": words[j], "similarity": round(float(sim[j]), 3), "tested_on_bracs": words[j] in tested} for j in np.argsort(-sim)[:4]]
        report["groups"].append(group)
    sheet(all_tiles, 12, LOOK / "groups.png")
    (LOOK / "look.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
