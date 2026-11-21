#!/usr/bin/env python3
"""Duct instrument, step 1: label every 16-pixel patch of a region (8 micrometres at 20x) by tissue kind, from UNI's patch tokens.

Usage (from the project folder):
  python -I tools/duct_tokens.py fit     # sample regions, fit PCA + k-means on patch tokens -> outputs/ducts/kinds.joblib, kinds_montage.png
  python -I tools/duct_tokens.py look    # label maps and overlays for a few gray-zone regions -> outputs/ducts/look_*.png

Real tissue pictures stay local.
"""
import pathlib
import sys

import joblib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import extract_features as ef
from train_classifier import ROOT, rows

SRC = ROOT / "data/bracs/BRACS_RoI/latest_version"
OUT = ROOT / "outputs/ducts"
K, GRID = 10, 14
COLOURS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
                    [170, 110, 40], [128, 128, 128], [0, 0, 0], [255, 255, 255]], np.uint8)


class Tokens:
    def __init__(self):
        model, _, _, _ = ef.load_model("uni")
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.eval().to(self.dev).half()
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.dev).view(1, 3, 1, 1)

    def __call__(self, tiles):
        """tiles: (n, 224, 224, 3) uint8 -> (n, 14, 14, 1024) patch tokens."""
        out = []
        with torch.inference_mode():
            for s in range(0, len(tiles), 64):
                x = torch.from_numpy(tiles[s:s + 64]).to(self.dev).permute(0, 3, 1, 2).float().div_(255)
                t = self.model.forward_features(((x - self.mean) / self.std).half())[:, 1:]
                out.append(t.float().cpu().numpy().reshape(len(t), GRID, GRID, -1))
        return np.concatenate(out)


def region_map(tok, kinds, path):
    """The image at 20x and a map with one tissue kind per 16-pixel patch (-1 where no tile covers it)."""
    tiles, coords, tissue, size, padded = ef.region_tiles(path)
    t = tok(tiles)
    lab = kinds["kmeans"].predict(kinds["pca"].transform(t.reshape(-1, t.shape[-1]))).reshape(len(tiles), GRID, GRID)
    h, w = max(int(size[1]), ef.TILE), max(int(size[0]), ef.TILE)
    img = np.full((h, w, 3), 255, np.uint8)
    m = np.full((h // 16 + 1, w // 16 + 1), -1, np.int16)
    for tile, (x, y), l in zip(tiles, coords, lab):
        img[y:y + ef.TILE, x:x + ef.TILE] = tile
        m[y // 16:y // 16 + GRID, x // 16:x // 16 + GRID] = l
    return img, m[:h // 16, :w // 16]


def fit():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = rows("train")
    pick = rng.choice(len(train), 280, replace=False)
    tok = Tokens()
    feats, patches = [], []
    for n, i in enumerate(pick):
        tiles, coords, tissue, size, padded = ef.region_tiles(SRC / train[i]["file"])
        keep = rng.choice(len(tiles), min(4, len(tiles)), replace=False)
        t = tok(tiles[keep])
        for a, k in enumerate(keep):
            for _ in range(40):
                r, c = rng.integers(GRID, size=2)
                feats.append(t[a, r, c])
                patches.append((tiles[k], r, c))
        if n % 70 == 0:
            print("sampled", n, "/", len(pick), flush=True)
    x = np.array(feats)
    pca = PCA(32, random_state=0).fit(x)
    km = KMeans(K, n_init=10, random_state=0).fit(pca.transform(x))
    joblib.dump({"pca": pca, "kmeans": km}, OUT / "kinds.joblib")
    # montage: for each kind, 16 example patches shown with their surroundings (a 64-pixel window, the patch in the middle)
    sheet = Image.new("RGB", (16 * 66, K * 66), "white")
    info = []
    for k in range(K):
        members = np.flatnonzero(km.labels_ == k)
        d = np.linalg.norm(pca.transform(x[members]) - km.cluster_centers_[k], axis=1)
        col = []
        for j, mi in enumerate(members[np.argsort(d)][:400:25]):
            tile, r, c = patches[mi]
            y0, x0 = min(max(r * 16 - 24, 0), 160), min(max(c * 16 - 24, 0), 160)
            sheet.paste(Image.fromarray(tile[y0:y0 + 64, x0:x0 + 64]), (j * 66, k * 66))
        rgb = np.mean([patches[mi][0][patches[mi][1] * 16:patches[mi][1] * 16 + 16, patches[mi][2] * 16:patches[mi][2] * 16 + 16].mean((0, 1)) for mi in members[:500]], axis=0)
        info.append((k, len(members), [int(v) for v in rgb]))
    sheet.save(OUT / "kinds_montage.png")
    for row in info:
        print("kind %d: %d sample patches, mean colour %s" % row)


def look():
    kinds = joblib.load(OUT / "kinds.joblib")
    tok = Tokens()
    rng = np.random.default_rng(1)
    train = rows("train")
    for name in ("UDH", "FEA", "ADH", "DCIS"):
        cand = [r for r in train if r["label_name"] == name]
        for j in rng.choice(len(cand), 3, replace=False):
            img, m = region_map(tok, kinds, SRC / cand[j]["file"])
            over = COLOURS[m].repeat(16, 0).repeat(16, 1)[:img.shape[0], :img.shape[1]]
            both = np.hstack([img[:over.shape[0], :over.shape[1]], (0.5 * img[:over.shape[0], :over.shape[1]] + 0.5 * over).astype(np.uint8)])
            im = Image.fromarray(both)
            im.thumbnail((1800, 1800))
            im.save(OUT / f"look_{name}_{pathlib.Path(cand[j]['file']).stem}.png")
            print(name, cand[j]["file"], img.shape, flush=True)


if __name__ == "__main__":
    {"fit": fit, "look": look}[sys.argv[1]]()
