#!/usr/bin/env python3
"""Ledger entry 55, milestone 2: the slide-level step.

A whole slide is read at 20x on a grid of 224-pixel tiles. One pass of UNI gives every tissue tile its feature and the
share of its 196 patches that look like duct cells (the token kinds of entry 41). Windows of 3 x 3 tiles (0.34 mm, the
size of a typical gray-zone region) with enough tissue and enough duct cells are the "duct regions". Each is scored by
the frozen classifier the way a BRACS region is: the average of its tiles next to UNI's view of the whole window.
Nothing is trained here.

Usage (from the project folder):
  python -I tools/slide_step.py check                          # the tile path must reproduce the saved BRACS region features
  python -I tools/slide_step.py slide <slide> <mpp> <out.npz>  # one slide -> windows with their 7 probabilities
"""
import pathlib
import sys

import joblib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
ROOT = pathlib.Path(__file__).resolve().parent.parent
TILE, GRID = 224, 14
WIN = 3            # a window is WIN x WIN tiles: 0.336 mm at 0.5 um per pixel (BRACS gray-zone regions: median side 0.24 to 0.47 mm)
MIN_TISSUE_TILES = 5   # of 9: tiles that are at least half tissue
MIN_DUCT = 0.20    # share of the window's patches that are duct-cell kinds (the bar entry 46 used for its windows)
CELLS = [0, 3, 7, 8, 9]   # duct-cell kinds of entry 41


class Uni:
    """UNI once: for a batch of 224-pixel tiles, the feature of each tile and its 14 x 14 patch tokens."""

    def __init__(self):
        import torch
        import extract_features as ef
        self.torch = torch
        model, _, _, _ = ef.load_model("uni")
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.eval().to(self.dev).half()
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.dev).view(1, 3, 1, 1)

    def __call__(self, tiles, tokens=True):
        feats, toks = [], []
        with self.torch.inference_mode():
            for s in range(0, len(tiles), 64):
                x = self.torch.from_numpy(tiles[s:s + 64]).to(self.dev).permute(0, 3, 1, 2).float().div_(255)
                t = self.model.forward_features(((x - self.mean) / self.std).half())
                feats.append(self.model.forward_head(t).float().cpu().numpy())
                if tokens:
                    toks.append(t[:, 1:].float().cpu().numpy())
        return np.concatenate(feats), (np.concatenate(toks) if tokens else None)


def tissue_share(tiles):
    return (tiles.sum(axis=3, dtype=np.uint16) < 660).mean(axis=(1, 2))   # the regions' rule: mean colour under 220


def whole_view(img):
    """UNI's own input transform on a whole image: shorter side to 224 (bicubic), centre crop."""
    from PIL import Image
    w, h = img.size
    k = TILE / min(w, h)
    img = img.resize((max(round(w * k), TILE), max(round(h * k), TILE)), Image.BICUBIC)
    w, h = img.size
    x0, y0 = (w - TILE) // 2, (h - TILE) // 2
    return np.asarray(img.crop((x0, y0, x0 + TILE, y0 + TILE)))


def process(path, mpp, uni, boxes=()):
    """One slide. Returns the windows (level-0 corner, probabilities, tissue tiles, duct share) and, for each given box
    (x0, y0, x1, y1 in level-0 pixels), the probabilities of that box scored the same way."""
    import tiffslide
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    kinds = joblib.load(ROOT / "outputs/ducts/kinds.joblib")
    engine = joblib.load(ROOT / "outputs/m2/engine.joblib")
    slide = tiffslide.TiffSlide(path)
    w, h = slide.dimensions
    step = round(TILE * 0.5 / mpp)   # level-0 pixels under one 224-pixel tile at 0.5 um per pixel
    level = slide.get_best_level_for_downsample(32)
    lw, lh = slide.level_dimensions[level]
    small = np.asarray(slide.read_region((0, 0), level, (lw, lh)).convert("RGB"), dtype=np.int16).sum(2)
    rough = (small < 660) & (small > 90)   # tissue on the overview; black background is not tissue
    scale = w / lw
    cand = [(ix, iy) for iy in range(h // step) for ix in range(w // step)
            if rough[int(iy * step / scale):max(int((iy + 1) * step / scale), int(iy * step / scale) + 1),
                     int(ix * step / scale):max(int((ix + 1) * step / scale), int(ix * step / scale) + 1)].mean() >= 0.1]
    pool = ThreadPoolExecutor(8)

    def read(c):
        return np.asarray(slide.read_region((c[0] * step, c[1] * step), 0, (step, step)).convert("RGB").resize((TILE, TILE), Image.BOX))

    feat, tissue, duct = {}, {}, {}
    for b in range(0, len(cand), 128):
        tiles = np.stack(list(pool.map(read, cand[b:b + 128])))
        f, t = uni(tiles)
        k = kinds["kmeans"].predict(kinds["pca"].transform(t.reshape(-1, t.shape[-1]))).reshape(len(tiles), -1)
        for c, fi, ti, ki in zip(cand[b:b + 128], f, tissue_share(tiles), k):
            feat[c], tissue[c], duct[c] = fi, float(ti), float(np.isin(ki, CELLS).mean())

    def score(tile_keys, image):
        good = [c for c in tile_keys if tissue[c] >= 0.5] or list(tile_keys)   # as for regions: with no tile above the bar, all tiles count
        return np.mean([feat[c] for c in good], 0), whole_view(image)

    wins, avgs, views = [], [], []
    for iy in range(0, h // step - WIN + 1, WIN):
        for ix in range(0, w // step - WIN + 1, WIN):
            keys = [(ix + a, iy + b) for b in range(WIN) for a in range(WIN) if (ix + a, iy + b) in feat]
            n_tissue = sum(tissue[c] >= 0.5 for c in keys)
            d = sum(duct[c] for c in keys) / WIN ** 2
            if n_tissue >= MIN_TISSUE_TILES and d >= MIN_DUCT:
                a, v = score(keys, slide.read_region((ix * step, iy * step), 0, (WIN * step, WIN * step)).convert("RGB"))
                wins.append((ix * step, iy * step, n_tissue, d)); avgs.append(a); views.append(v)
    out = {"step": step, "mpp": mpp, "dimensions": np.array([w, h]), "tiles": len(cand),
           "tissue_tiles": int(sum(v >= 0.5 for v in tissue.values())), "windows": np.array(wins, np.float32).reshape(-1, 4),
           "tile_xy": np.array(cand, np.int32).reshape(-1, 2), "tile_tissue": np.array([tissue[c] for c in cand], np.float16),
           "tile_duct": np.array([duct[c] for c in cand], np.float16)}
    out["prob"] = engine.predict_proba(np.hstack([np.array(avgs), uni(np.stack(views), tokens=False)[0]])) if wins else np.zeros((0, 7))
    if len(boxes):
        bp = np.full((len(boxes), 7), np.nan)
        for n, (x0, y0, x1, y1) in enumerate(boxes):
            keys = [c for c in feat if x0 <= (c[0] + 0.5) * step < x1 and y0 <= (c[1] + 0.5) * step < y1]
            keys = keys or [c for c in [(int((x0 + x1) / 2 // step), int((y0 + y1) / 2 // step))] if c in feat]   # a box smaller than a tile
            if keys:
                a, v = score(keys, slide.read_region((int(x0), int(y0)), 0, (int(x1 - x0), int(y1 - y0))).convert("RGB"))
                bp[n] = engine.predict_proba(np.hstack([a, uni(v[None], tokens=False)[0][0]])[None])[0]
        out["box_prob"] = bp
    slide.close()
    return out


def check():
    """The tile path of this file must give the features that were saved for the BRACS regions (same pixels, same model)."""
    import extract_features as ef
    from train_classifier import rows
    r = rows("train")[0]
    tiles, _, _, _, _ = ef.region_tiles(ROOT / "data/bracs/BRACS_RoI/latest_version" / r["file"])
    saved = np.load(ROOT / "data/features/uni" / (r["file"][:-4] + ".npz"))["feats"].astype(np.float32)
    mine = Uni()(tiles, tokens=False)[0]
    cos = (mine * saved).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(saved, axis=1)
    print(r["file"], "tiles", len(tiles), "| cosine with the saved features: min", round(float(cos.min()), 4))
    assert cos.min() > 0.995, "the tile path does not reproduce the saved region features"
    print("check passed")


def one(path, mpp, dest):
    out = process(path, float(mpp), Uni())
    np.savez(dest, **out)
    print(path, "| tiles", out["tiles"], "tissue tiles", out["tissue_tiles"], "| duct windows", len(out["windows"]))


def look(path, npz, dest, n=12):
    """A sheet for a check by eye: the n windows with the largest duct share, and n tissue windows that were not kept."""
    import tiffslide
    from PIL import Image
    z = np.load(npz)
    slide, step = tiffslide.TiffSlide(path), int(z["step"])
    kept = z["windows"][np.argsort(-z["windows"][:, 3])][:n]
    taken = {(int(x), int(y)) for x, y, _, _ in z["windows"]}
    rng = np.random.default_rng(0)
    tiss = z["tile_xy"][z["tile_tissue"] >= 0.5]
    rest = [(ix // WIN * WIN * step, iy // WIN * WIN * step) for ix, iy in tiss[rng.permutation(len(tiss))]]
    rest = list(dict.fromkeys(c for c in rest if c not in taken))[:n]
    sheet = Image.new("RGB", (n * 228, 2 * 228), "white")
    for row, boxes in enumerate(([(int(x), int(y)) for x, y, _, _ in kept], rest)):
        for i, (x, y) in enumerate(boxes):
            sheet.paste(slide.read_region((x, y), 0, (WIN * step, WIN * step)).convert("RGB").resize((224, 224), Image.BOX), (i * 228, row * 228))
    sheet.save(dest)
    print("top row: kept windows (most duct cells first); bottom row: tissue windows that were not kept ->", dest)


if __name__ == "__main__":
    {"check": check, "slide": one, "look": look}[sys.argv[1]](*sys.argv[2:])
