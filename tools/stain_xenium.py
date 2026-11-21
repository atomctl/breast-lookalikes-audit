#!/usr/bin/env python3
"""Reading the biology, gate 1 (ledger entry 48): can the H&E image show which duct cells carry the CK5/6 stain's genes?

Data: 10x Xenium breast samples. Every cell has measured gene counts and a place on an H&E image of the same section.

Usage (from the project folder):
  python -I tools/stain_xenium.py cells <sample>      # gene counts inside the nucleus, cell classes -> data/xenium/<sample>/cells.npz
  python -I tools/stain_xenium.py align <sample>      # check where the cells fall on the H&E image -> data/xenium/<sample>/align.json
  python -I tools/stain_xenium.py shift <sample>      # how far the cells are off the image, per block (entry 51) -> outputs/stain/shift_<sample>.json
  python -I tools/stain_xenium.py features <sample>   # UNI patch tokens at every epithelial cell -> data/xenium/<sample>/features.npz
  python -I tools/stain_xenium.py cellfeatures <sample>  # one picture per labelled cell, nucleus in the middle (entry 52) -> features_cell.npz
  python -I tools/stain_xenium.py test                # train on some patients, test on another -> outputs/stain/gate1.json

Real tissue pictures stay local.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data/xenium"
OUT = ROOT / "outputs/stain"
XEN_UM = 0.2125  # micrometres per pixel of the Xenium run's own image
MPP = 0.5        # the project's working resolution (20x), micrometres per pixel
TILE, PATCH, GRID = 224, 16, 14
GROUPS = {"stain": ["KRT5", "KRT6B"], "basal": ["KRT5", "KRT6B", "KRT14"], "luminal": ["KRT8", "KRT7", "EPCAM"],
          "muscle": ["ACTA2", "MYH11", "MYLK", "OXTR"], "er": ["ESR1"]}
# one entry per sample: its patient, the files' prefix, and the cell table the H&E image belongs to
SAMPLES = {"breast_rep1": {"patient": 1, "prefix": "rep1"},
           "breast_ilc": {"patient": 3, "prefix": "ilc"},
           "breast_idc": {"patient": 2, "prefix": "idcaddon"},
           "breast_s2": {"patient": 4, "prefix": "s2"}}


def paths(sample):
    d, p = DATA / sample, SAMPLES[sample]["prefix"]
    return d, {k: d / f"{p}_{v}" for k, v in (("matrix", "cell_feature_matrix.h5"), ("cells", "cells.csv.gz"), ("tx", "transcripts.parquet"),
                                              ("he", "he_image.ome.tif"), ("align", "he_imagealignment.csv"))}


# ---------- cells: gene counts and classes ----------

def whole_cell_counts(path, genes):
    import h5py
    import scipy.sparse as sp
    m = h5py.File(path, "r")["matrix"]
    names = [x.decode() for x in m["features/name"][:]]
    x = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=m["shape"][:]).T.tocsr()
    ids = [b.decode() for b in m["barcodes"][:]]
    return ids, {g: np.asarray(x[:, names.index(g)].todense()).ravel() for g in genes}


def nucleus_counts(path, genes, ids):
    """Per cell, the number of transcripts of each gene that lie inside the nucleus outline (quality at least 20)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["cell_id", "overlaps_nucleus", "feature_name", "qv"])
    name = pc.cast(t["feature_name"], "string")
    keep = pc.and_(pc.and_(pc.equal(t["overlaps_nucleus"], 1), pc.greater_equal(t["qv"], 20)), pc.is_in(name, value_set=__import__("pyarrow").array(genes)))
    t = t.filter(keep)
    cell = np.array([str(c) for c in pc.cast(t["cell_id"], "string").to_pylist()])
    gene = np.array(pc.cast(t["feature_name"], "string").to_pylist())
    where = {c: i for i, c in enumerate(ids)}
    out = {g: np.zeros(len(ids), np.int32) for g in genes}
    idx = np.array([where.get(c, -1) for c in cell])
    for g in genes:
        m = (gene == g) & (idx >= 0)
        np.add.at(out[g], idx[m], 1)
    return out


def cells(sample):
    import pandas as pd
    d, p = paths(sample)
    genes = sorted({g for v in GROUPS.values() for g in v})
    ids, whole = whole_cell_counts(p["matrix"], genes)
    table = pd.read_csv(p["cells"])
    assert [str(c) for c in table["cell_id"]] == ids
    nuc = nucleus_counts(p["tx"], genes, ids)
    out = {"x": table["x_centroid"].values.astype(np.float32), "y": table["y_centroid"].values.astype(np.float32)}
    for k, v in GROUPS.items():
        out[f"whole_{k}"] = sum(whole[g] for g in v).astype(np.int32)
        out[f"nuc_{k}"] = sum(nuc[g] for g in v).astype(np.int32)
    np.savez_compressed(d / "cells.npz", **out)
    print(sample, "cells", len(ids))
    for k in GROUPS:
        print(f"  {k:8s} whole-cell: mean {out['whole_' + k].mean():.2f}, cells with any {100 * (out['whole_' + k] > 0).mean():.1f}% | "
              f"in nucleus: mean {out['nuc_' + k].mean():.2f}, cells with any {100 * (out['nuc_' + k] > 0).mean():.1f}%")


def classes(c):
    """Cell classes from gene counts inside the nucleus (thresholds fixed in ledger entry 48)."""
    myo = (c["nuc_basal"] >= 2) & (c["nuc_muscle"] >= 2)
    lum = (c["nuc_luminal"] >= 2) & ~myo
    return myo, lum


def duct_cells(c, myo, lum):
    """Luminal-type cells with at least 3 myoepithelial cells within 50 micrometres: the cells of ducts, not of invasive tumour."""
    from scipy.spatial import cKDTree
    xy = np.column_stack([c["x"], c["y"]])
    near = cKDTree(xy[myo]).query_ball_point(xy[lum], 50.0, return_length=True)
    out = np.zeros(len(xy), bool)
    out[np.flatnonzero(lum)[near >= 3]] = True
    return out


# ---------- align: where do the cells fall on the H&E image? ----------

class Slide:
    """The H&E image as a pyramid; level 0 is the full resolution."""

    def __init__(self, path):
        import tifffile
        import zarr
        self.tif = tifffile.TiffFile(path)
        opened = [zarr.open(lv.aszarr(), mode="r") for lv in self.tif.series[0].levels]
        self.levels = [z if hasattr(z, "shape") else z[sorted(z.array_keys())[0]] for z in opened]  # a level can open as a group of one array
        self.planar = self.levels[0].shape[0] in (3, 4) and self.levels[0].shape[-1] not in (3, 4)
        s = self.levels[0].shape
        self.h, self.w = (s[1], s[2]) if self.planar else (s[0], s[1])

    def size(self, level):
        s = self.levels[level].shape
        return (s[1], s[2]) if self.planar else (s[0], s[1])

    def read(self, level, y0, y1, x0, x1):
        """RGB pixels of a box at one level; outside the image is white."""
        z = self.levels[level]
        h, w = self.size(level)
        out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
        a, b, c, d = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
        if a < b and c < d:
            px = np.asarray(z[:3, a:b, c:d]).transpose(1, 2, 0) if self.planar else np.asarray(z[a:b, c:d, :3])
            out[a - y0:b - y0, c - x0:d - x0] = px
        return out


def to_he(matrix, x_um, y_um):
    p = matrix @ np.vstack([x_um, y_um, np.ones(len(x_um))])
    return p[0], p[1]


def estimate(slide, c, res=16.0, mpp_override=None):
    """For a sample without a published matrix: find the map from cell places (micrometres) to H&E pixels.
    Step 1: try the 8 flips and quarter turns, each with the shift that best lays the cell density over the dark pixels.
    Step 2: measure the leftover shift in 1 mm blocks and fit one affine map through them."""
    import re
    from scipy.signal import fftconvolve
    mpp = float(re.search(r'PhysicalSizeX="([0-9.]+)"', slide.tif.ome_metadata).group(1))
    mpp = mpp_override or mpp
    lv = min(range(len(slide.levels)), key=lambda k: abs(slide.w / slide.size(k)[1] * mpp - 4.0))
    lh, lw = slide.size(lv)
    px = slide.w / lw * mpp                               # micrometres per pixel at this level
    dark = 255.0 - slide.read(lv, 0, lh, 0, lw).mean(2)

    def binned(a, um):
        b = max(int(round(um / px)), 1)
        h, w = a.shape[0] // b, a.shape[1] // b
        return a[:h * b, :w * b].reshape(h, b, w, b).mean((1, 3)), b * px

    def density(x, y, shape, um):
        d = np.zeros(shape)
        i, j = (y / um).astype(int), (x / um).astype(int)
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        np.add.at(d, (i[ok], j[ok]), 1)
        return d

    d16, um = binned(dark, res)
    wx, wy = float(c["x"].max()), float(c["y"].max())
    turns = {"as is": lambda x, y: (x, y), "flip x": lambda x, y: (wx - x, y), "flip y": lambda x, y: (x, wy - y), "half turn": lambda x, y: (wx - x, wy - y),
             "swap": lambda x, y: (y, x), "swap, flip x": lambda x, y: (wy - y, x), "swap, flip y": lambda x, y: (y, wx - x), "swap, half turn": lambda x, y: (wy - y, wx - x)}
    best = None
    a0 = d16 - d16.mean()
    for name, fn in turns.items():
        x, y = fn(c["x"], c["y"])
        cd = density(x, y, (int(y.max() / um) + 1, int(x.max() / um) + 1), um)
        cc = fftconvolve(a0, (cd - cd.mean())[::-1, ::-1], mode="full")
        i, j = np.unravel_index(cc.argmax(), cc.shape)
        peak = cc.max() / (np.linalg.norm(a0) * np.linalg.norm(cd - cd.mean()))
        if best is None or peak > best[0]:
            best = (peak, name, fn, (j - (cd.shape[1] - 1)) * um, (i - (cd.shape[0] - 1)) * um)
    peak, name, fn, sx, sy = best
    print(f"estimate: orientation '{name}', shift ({sx:.0f}, {sy:.0f}) um, peak {peak:.3f}", flush=True)
    x, y = fn(c["x"], c["y"])
    x, y = x + sx, y + sy                                 # micrometres in the H&E frame, to within one 16-um bin
    d4, um4 = binned(dark, 4.0)
    cd4 = density(x, y, d4.shape, um4)
    src, dst = [], []
    step, pad = int(1000 / um4), int(80 / um4)
    for r in range(pad, d4.shape[0] - step - pad, step):
        for q in range(pad, d4.shape[1] - step - pad, step):
            blk = cd4[r:r + step, q:q + step]
            if blk.sum() < 300:
                continue
            ref = d4[r - pad:r + step + pad, q - pad:q + step + pad]
            cc = fftconvolve(ref - ref.mean(), (blk - blk.mean())[::-1, ::-1], mode="valid")
            i, j = np.unravel_index(cc.argmax(), cc.shape)
            src.append([(q + step / 2) * um4, (r + step / 2) * um4, 1.0])
            dst.append([(q + step / 2 + j - pad) * um4, (r + step / 2 + i - pad) * um4])
    src, dst = np.array(src), np.array(dst)
    keep = np.abs(dst - src[:, :2] - np.median(dst - src[:, :2], 0)).max(1) < 40       # drop blocks whose shift is far from the rest
    fit = np.linalg.lstsq(src[keep], dst[keep], rcond=None)[0].T                           # 2 x 3: frame -> refined frame
    print(f"estimate: {keep.sum()} of {len(src)} blocks used, median leftover shift {np.median(dst[keep] - src[keep, :2], 0).round(1)} um", flush=True)
    # compose: cell place -> turned and shifted frame -> refined frame -> H&E pixels
    probe = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    px_, py_ = fn(probe[:, 0], probe[:, 1])
    framed = np.column_stack([px_ + sx, py_ + sy, np.ones(3)])
    out = (fit @ framed.T).T / mpp                        # H&E pixels of the three probe points
    m = np.linalg.solve(np.column_stack([probe, np.ones(3)]), out).T
    return np.vstack([m, [0, 0, 1]])


def centre_darker(slide, m, x_um, y_um, rng, n=600):
    """Share of sampled cells for which the image is darker at the nucleus centre (2 um) than in a ring around it (8 to 12 um)."""
    x, y = to_he(m, x_um, y_um)
    ok = np.flatnonzero((x > 100) & (x < slide.w - 100) & (y > 100) & (y < slide.h - 100))
    if len(ok) < n:
        return 0.0
    mpp = 1.0 / np.sqrt(abs(np.linalg.det(m[:2, :2])))
    rad = int(round(12 / mpp))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    dist = np.hypot(yy, xx) * mpp
    hits = []
    for i in rng.choice(ok, n, replace=False):
        cx, cy = int(round(x[i])), int(round(y[i]))
        g = 255.0 - slide.read(0, cy - rad, cy + rad + 1, cx - rad, cx + rad + 1).mean(2)
        hits.append(g[dist <= 2].mean() > g[(dist >= 8) & (dist <= 12)].mean())
    return float(np.mean(hits))


def align(sample):
    """Candidate maps from cell places (micrometres) to H&E pixels: the published matrix read both ways, the image taken as
    already lying in the Xenium frame, and, without a published matrix, a map estimated from the image. The candidate for
    which the image is darkest at the nucleus centres is kept."""
    from PIL import Image, ImageDraw
    d, p = paths(sample)
    c = np.load(d / "cells.npz")
    slide = Slide(p["he"])
    scale = np.diag([1 / XEN_UM, 1 / XEN_UM, 1.0])  # micrometres -> Xenium pixels
    readings = {"image already in the Xenium frame": scale}
    if p["align"].exists():
        a = np.loadtxt(p["align"], delimiter=",")
        readings.update({"matrix maps H&E to Xenium": np.linalg.inv(a) @ scale, "matrix maps Xenium to H&E": a @ scale})
    else:
        readings["no published matrix; estimated from the image"] = estimate(slide, c)
    rng = np.random.default_rng(0)
    result = {"sample": sample, "image": [slide.w, slide.h], "readings": {}}
    best = None
    for name, m in readings.items():
        x, y = to_he(m, c["x"], c["y"])
        inside = float(((x >= 0) & (x < slide.w) & (y >= 0) & (y < slide.h)).mean())
        dk = centre_darker(slide, m, c["x"], c["y"], rng) if inside > 0.5 else 0.0
        result["readings"][name] = {"cells_inside_image": round(inside, 4), "nucleus_centre_darker_than_surround": round(dk, 3)}
        if best is None or dk > best[0]:
            best = (dk, name, m)
    dk, name, m = best
    he_mpp = 1.0 / np.sqrt(abs(np.linalg.det(m[:2, :2])))
    result.update({"chosen": name, "he_micrometres_per_pixel": round(float(he_mpp), 4), "nucleus_centre_darker_than_surround": round(dk, 3), "um_to_he": m.tolist()})
    (d / "align.json").write_text(json.dumps(result, indent=1))
    # one picture for a check by eye: a field with many cells, centres marked
    x, y = to_he(m, c["x"], c["y"])
    pick = rng.choice(np.flatnonzero((x > 400) & (x < slide.w - 400) & (y > 400) & (y < slide.h - 400)), 200, replace=False)
    i = pick[np.argmax([np.sum((np.abs(x - x[j]) < 150) & (np.abs(y - y[j]) < 150)) for j in pick])]
    cx, cy, s = int(x[i]), int(y[i]), int(round(100 / he_mpp))
    im = Image.fromarray(slide.read(0, cy - s, cy + s, cx - s, cx + s)).resize((800, 800), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    k = 800 / (2 * s)
    for j in np.flatnonzero((np.abs(x - cx) < s) & (np.abs(y - cy) < s)):
        u, v = (x[j] - cx + s) * k, (y[j] - cy + s) * k
        dr.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(0, 255, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f"align_{sample}.png")
    print(json.dumps({k: v for k, v in result.items() if k != "um_to_he"}, indent=1))


def shift(sample, n_blocks=600):
    """Ledger entry 51: how far are the cell places off the H&E image? Per block of 256 micrometres, the shift (0.5-um steps,
    up to 16 um) that best lays the cell centres over the dark pixels; then what one better affine map would leave over."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve
    d, p = paths(sample)
    c = np.load(d / "cells.npz")
    al = json.loads((d / "align.json").read_text())
    m, he_mpp = np.array(al["um_to_he"]), al["he_micrometres_per_pixel"]
    slide = Slide(p["he"])
    myo, lum = classes(c)
    duct = duct_cells(c, myo, lum)
    x, y = to_he(m, c["x"], c["y"])
    xu, yu = x * he_mpp, y * he_mpp                       # micrometres in the H&E image's frame
    B, PAD, UM = 256.0, 16.0, 0.5
    n, pad = int(B / UM), int(PAD / UM)
    on = (xu >= 0) & (yu >= 0) & (xu < slide.w * he_mpp) & (yu < slide.h * he_mpp)
    xu, yu, duct = xu[on], yu[on], duct[on]
    key = (yu // B).astype(int) * 100000 + (xu // B).astype(int)
    keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    dcnt = np.bincount(inv, weights=duct, minlength=len(keys))
    r, q = keys // 100000, keys % 100000
    ok = (cnt >= 150) & (r >= 1) & (q >= 1) & ((r + 2) * B < slide.h * he_mpp) & ((q + 2) * B < slide.w * he_mpp)
    rng = np.random.default_rng(0)
    with_ducts = np.flatnonzero(ok & (dcnt >= 20))
    others = np.flatnonzero(ok & (dcnt < 20))
    pick = np.concatenate([with_ducts, rng.choice(others, min(len(others), max(n_blocks - len(with_ducts), 0)), replace=False)])

    def measure(k, dx=0.0, dy=0.0):
        y0, x0 = (k // 100000) * B, (k % 100000) * B
        box = [int(round(v / he_mpp)) for v in (y0 - PAD, y0 + B + PAD, x0 - PAD, x0 + B + PAD)]
        img = Image.fromarray(slide.read(0, *box)).resize((n + 2 * pad, n + 2 * pad), Image.BOX)
        dark = 255.0 - np.asarray(img, np.float32).mean(2)
        sel = key == k
        i, j = ((yu[sel] + dy - y0) / UM).astype(int), ((xu[sel] + dx - x0) / UM).astype(int)
        keep = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        pts = np.zeros((n, n), np.float32)
        np.add.at(pts, (i[keep], j[keep]), 1)
        pts = gaussian_filter(pts, 2.0 / UM)
        cc = fftconvolve(dark - dark.mean(), (pts - pts.mean())[::-1, ::-1], mode="valid")  # (2 pad + 1) square; the middle is "no shift"
        i, j = np.unravel_index(cc.argmax(), cc.shape)
        return (j - pad) * UM, (i - pad) * UM, i in (0, 2 * pad) or j in (0, 2 * pad)

    got = np.array([measure(keys[b]) for b in pick])
    good = got[:, 2] == 0
    s = got[good, :2]
    pos = np.column_stack([(q[pick][good] + 0.5) * B, (r[pick][good] + 0.5) * B])
    has_ducts = (dcnt[pick] >= 20)[good]

    # check of the tool: a planted move of the cells must come back as the opposite change in the measured shift
    probe = pick[good][:30]
    moved = np.array([measure(keys[b], 3.0, -2.0)[:2] for b in probe])
    planted = np.median(moved - s[:30], 0)
    tool_ok = bool(np.abs(planted - [-3.0, 2.0]).max() <= 0.5)

    A = np.column_stack([pos, np.ones(len(pos))])
    left = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]   # what one better affine map would leave over

    def summary(w):
        ln, ll = np.hypot(s[w, 0], s[w, 1]), np.hypot(left[w, 0], left[w, 1])
        return {"blocks": int(w.sum()), "median_shift_um": round(float(np.median(ln)), 2), "p90_shift_um": round(float(np.percentile(ln, 90)), 2),
                "median_shift_vector_um": np.median(s[w], 0).round(2).tolist(),
                "after_one_better_map_median_um": round(float(np.median(ll)), 2), "after_one_better_map_p90_um": round(float(np.percentile(ll, 90)), 2)}

    result = {"sample": sample, "patient": SAMPLES[sample]["patient"], "he_micrometres_per_pixel": he_mpp, "blocks_tried": int(len(pick)),
              "blocks_on_window_edge": int((~good).sum()), "planted_shift_recovered_um": planted.round(2).tolist(), "tool_ok": tool_ok,
              "duct_blocks": summary(has_ducts) if has_ducts.sum() else None, "all_blocks": summary(np.ones(len(s), bool))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"shift_{sample}.json").write_text(json.dumps(result, indent=1))
    np.savez(OUT / f"shift_{sample}.npz", pos=pos, shift=s, has_ducts=has_ducts)
    print(json.dumps(result, indent=1))


# ---------- features: UNI patch tokens at each duct cell ----------

def features(sample):
    """For every duct cell and every myoepithelial cell: the UNI patch token under the nucleus centre (8 micrometres wide)
    and the mean of the 3 x 3 tokens around it (24 micrometres), read at the project's 20x resolution."""
    from PIL import Image
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
    import duct_tokens as dt
    d, p = paths(sample)
    c = np.load(d / "cells.npz")
    al = json.loads((d / "align.json").read_text())
    m, he_mpp = np.array(al["um_to_he"]), al["he_micrometres_per_pixel"]
    myo, lum = classes(c)
    keep = np.flatnonzero(duct_cells(c, myo, lum) | myo)
    x, y = to_he(m, c["x"][keep], c["y"][keep])
    k = he_mpp / MPP                          # H&E pixels -> 20x pixels
    x20, y20 = x * k, y * k
    tx, ty = (x20 // TILE).astype(int), (y20 // TILE).astype(int)
    px, py = ((x20 % TILE) // PATCH).astype(int), ((y20 % TILE) // PATCH).astype(int)
    slide = Slide(p["he"])
    src = int(round(TILE / k))                # side of one tile in H&E pixels
    tiles = sorted(set(zip(ty.tolist(), tx.tolist())))
    of_tile = {t: [] for t in tiles}
    for i, t in enumerate(zip(ty.tolist(), tx.tolist())):
        of_tile[t].append(i)
    tok = dt.Tokens()
    centre, around = np.zeros((len(keep), 1024), np.float16), np.zeros((len(keep), 1024), np.float16)
    tissue = np.zeros(len(keep), np.float16)
    for s in range(0, len(tiles), 64):
        batch = tiles[s:s + 64]
        imgs = np.stack([np.asarray(Image.fromarray(slide.read(0, int(round(r * TILE / k)), int(round(r * TILE / k)) + src,
                                                               int(round(q * TILE / k)), int(round(q * TILE / k)) + src)).resize((TILE, TILE), Image.BOX))
                         for r, q in batch])
        t = tok(imgs)                         # (n, 14, 14, 1024)
        for b, key in enumerate(batch):
            share = float((imgs[b].sum(2, dtype=np.uint16) < 660).mean())
            for i in of_tile[key]:
                r0, r1, c0, c1 = max(py[i] - 1, 0), min(py[i] + 2, GRID), max(px[i] - 1, 0), min(px[i] + 2, GRID)
                centre[i], around[i], tissue[i] = t[b, py[i], px[i]], t[b, r0:r1, c0:c1].mean((0, 1)), share
        if s % 640 == 0:
            print(f"{sample}: tiles {s}/{len(tiles)}", flush=True)
    np.savez(d / "features.npz", cell=keep, centre=centre, around=around, tissue=tissue)
    print(sample, "cells with features", len(keep), "tiles", len(tiles))


FIELD = 56.0      # micrometres: side of the picture cut around one cell; at 224 pixels that is 0.25 um per pixel (40x, as BRACS is scanned)


def cellfeatures(sample):
    """Ledger entry 52: one picture per labelled duct cell, FIELD micrometres wide with the cell in the middle, described by UNI:
    the mean of the 2 x 2 tokens at the centre (8 micrometres, the nucleus) and the mean of all tokens (the surroundings)."""
    from PIL import Image
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
    import duct_tokens as dt
    d, p = paths(sample)
    c = np.load(d / "cells.npz")
    al = json.loads((d / "align.json").read_text())
    m, he_mpp = np.array(al["um_to_he"]), al["he_micrometres_per_pixel"]
    myo, lum = classes(c)
    clear = duct_cells(c, myo, lum) & (c["nuc_luminal"] >= 4) & (c["nuc_muscle"] == 0)  # the labelled cells, as in load()
    pos = clear & (c["nuc_stain"] >= 2)
    keep = np.flatnonzero(pos | (clear & (c["nuc_stain"] == 0) & (c["whole_stain"] <= 1)))
    x, y = to_he(m, c["x"][keep], c["y"][keep])
    half = FIELD / he_mpp / 2
    slide = Slide(p["he"])
    tok = dt.Tokens()
    centre, field = np.zeros((len(keep), 1024), np.float16), np.zeros((len(keep), 1024), np.float16)
    order = np.random.default_rng(0).permutation(len(keep))
    for s in range(0, len(keep), 256):
        idx = order[s:s + 256]
        imgs = np.stack([np.asarray(Image.fromarray(slide.read(0, int(round(y[i] - half)), int(round(y[i] + half)), int(round(x[i] - half)),
                                                               int(round(x[i] + half)))).resize((TILE, TILE), Image.LANCZOS)) for i in idx])
        t = tok(imgs)                         # (n, 14, 14, 1024)
        centre[idx], field[idx] = t[:, 6:8, 6:8].mean((1, 2)), t.mean((1, 2))
        if s == 0:                            # a sheet for a check by eye: positives on top, negatives below
            a, b = idx[pos[keep[idx]]][:32], idx[~pos[keep[idx]]][:32]
            sheet = np.full((8 * TILE, 8 * TILE, 3), 255, np.uint8)
            for n, i in enumerate(list(a) + [None] * (32 - len(a)) + list(b)):
                if i is not None:
                    sheet[n // 8 * TILE:(n // 8 + 1) * TILE, n % 8 * TILE:(n % 8 + 1) * TILE] = imgs[list(idx).index(i)]
            OUT.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sheet).save(OUT / f"cellcrops_{sample}.png")
        print(f"{sample}: cells {min(s + 256, len(keep))}/{len(keep)}", flush=True)
    np.savez(d / "features_cell.npz", cell=keep, centre=centre, field=field)
    print(sample, "labelled cells with a picture", len(keep), "positive", int(pos[keep].sum()))


# ---------- test: train on some patients, test on another ----------

BLOCK = 500.0     # micrometres: cells are resampled in blocks of this size, because neighbours are alike
NEAR = 40.0       # micrometres: "the same neighbourhood" for the local test
MAX_TRAIN = 25000  # labelled duct cells used per training sample


def load(sample, cell=False):
    """Labelled duct cells of one sample: features, label (1 = carries the stain's genes), place, and geometry for the baseline.
    With cell=True also the description of each cell's own picture (entry 52), for the same cells."""
    from scipy.spatial import cKDTree
    d, _ = paths(sample)
    c, f = np.load(d / "cells.npz"), np.load(d / "features.npz")
    myo, lum = classes(c)
    duct = duct_cells(c, myo, lum)
    xy = np.column_stack([c["x"], c["y"]])
    clear = duct & (c["nuc_luminal"] >= 4) & (c["nuc_muscle"] == 0)  # clearly luminal, no muscle transcript in the nucleus
    pos = clear & (c["nuc_stain"] >= 2)
    neg = clear & (c["nuc_stain"] == 0) & (c["whole_stain"] <= 1)
    row = {cell: i for i, cell in enumerate(f["cell"])}
    sel = np.array([i for i in np.flatnonzero(pos | neg) if f["tissue"][row[i]] >= 0.25])
    rows = np.array([row[i] for i in sel])
    other = ~myo & ~lum
    tm, tl, to = cKDTree(xy[myo]), cKDTree(xy[lum]), cKDTree(xy[other])
    geo = np.column_stack([np.log1p(tm.query(xy[sel])[0]), tm.query_ball_point(xy[sel], 30.0, return_length=True),
                           tl.query_ball_point(xy[sel], 30.0, return_length=True), np.log1p(to.query(xy[sel])[0])]).astype(np.float32)
    # ducts: groups of duct cells that touch (centres within 15 micrometres)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dc = np.flatnonzero(duct)
    pairs = cKDTree(xy[dc]).query_pairs(15.0, output_type="ndarray")
    comp = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(dc), len(dc))), directed=False)[1]
    comp_of = dict(zip(dc.tolist(), comp.tolist()))
    out = {"x": np.hstack([f["centre"][rows], f["around"][rows]]).astype(np.float32), "y": pos[sel].astype(int), "xy": xy[sel], "geo": geo,
           "duct": np.array([comp_of[i] for i in sel]), "block": (xy[sel, 0] // BLOCK).astype(int) * 10000 + (xy[sel, 1] // BLOCK).astype(int),
           "counts": {"cells": int(len(xy)), "myoepithelial": int(myo.sum()), "luminal_type": int(lum.sum()), "duct_cells": int(duct.sum()),
                      "positive": int(pos[sel].sum()), "negative": int(neg[sel].sum())}}
    if cell:
        g = np.load(d / "features_cell.npz")
        row2 = {c2: i for i, c2 in enumerate(g["cell"])}
        rows2 = np.array([row2[i] for i in sel])
        out["centre"], out["field"] = g["centre"][rows2].astype(np.float32), g["field"][rows2].astype(np.float32)
    return out


WIDE = (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1)  # entry 52: entry 48's grid ended at 0.0001, and that edge was chosen every time


def fit_score(xa, ya, ga, xb, grid=(0.0001, 0.001, 0.01, 0.1)):
    """Class-weighted logistic regression; the strength C is chosen on blocks of the training cells."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from train_classifier import classifier
    best = None
    for cc in grid:
        p = np.zeros(len(ya))
        for i, j in GroupKFold(3).split(xa, ya, ga):
            p[j] = classifier(cc).fit(xa[i], ya[i]).decision_function(xa[j])
        s = roc_auc_score(ya, p)
        if best is None or s > best[0]:
            best = (s, cc)
    return classifier(best[1]).fit(xa, ya).decision_function(xb), best[1]


def local_auc(score, y, xy):
    """For each positive cell: the share of negative cells within NEAR micrometres that it outranks. One value per positive cell."""
    from scipy.spatial import cKDTree
    neg = np.flatnonzero(y == 0)
    tree = cKDTree(xy[neg])
    out = np.full(len(y), np.nan)
    for i in np.flatnonzero(y == 1):
        j = neg[tree.query_ball_point(xy[i], NEAR)]
        if len(j):
            out[i] = (score[i] > score[j]).mean() + 0.5 * (score[i] == score[j]).mean()
    return out


def block_draws(block, rng, n=1000):
    ids = np.unique(block)
    where = {b: np.flatnonzero(block == b) for b in ids}
    return [np.concatenate([where[b] for b in rng.choice(ids, len(ids))]) for _ in range(n)]


def test():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    names = [s for s in SAMPLES if (DATA / s / "features.npz").exists()]
    data = {s: load(s) for s in names}
    rng = np.random.default_rng(0)
    result = {"samples": {s: dict(patient=SAMPLES[s]["patient"], **data[s]["counts"]) for s in names}, "held_out": {}}
    r95 = lambda v: [round(float(q), 3) for q in np.percentile(v, [2.5, 97.5])]
    for held in names:
        b = data[held]
        train = [s for s in names if SAMPLES[s]["patient"] != SAMPLES[held]["patient"]]
        xs, ys, gs, geos = [], [], [], []
        for k, s in enumerate(train):
            a = data[s]
            pick = rng.choice(len(a["y"]), min(MAX_TRAIN, len(a["y"])), replace=False)
            xs.append(a["x"][pick]); ys.append(a["y"][pick]); gs.append(a["block"][pick] + k * 10 ** 9); geos.append(a["geo"][pick])
        xa, ya, ga, geo_a = np.vstack(xs), np.hstack(ys), np.hstack(gs), np.vstack(geos)
        score, c_model = fit_score(xa, ya, ga, b["x"])
        base, _ = fit_score(geo_a, ya, ga, b["geo"])
        y = b["y"]
        draws = [i for i in block_draws(b["block"], rng) if 0 < y[i].sum() < len(i)]
        loc_m, loc_b = local_auc(score, y, b["xy"]), local_auc(base, y, b["xy"])
        has = ~np.isnan(loc_m)
        ducts = [np.flatnonzero(b["duct"] == d) for d in np.unique(b["duct"])]
        ducts = [i for i in ducts if len(i) >= 30]
        share, pred = np.array([y[i].mean() for i in ducts]), np.array([score[i].mean() for i in ducts])
        pred_b = np.array([base[i].mean() for i in ducts])
        dd = [rng.choice(len(ducts), len(ducts)) for _ in range(1000)] if len(ducts) >= 8 else []
        result["held_out"][held] = {
            "patient": SAMPLES[held]["patient"], "trained_on": train, "C": c_model, "cells": int(len(y)), "positive": int(y.sum()),
            "auc": round(float(roc_auc_score(y, score)), 3), "auc_range95": r95([roc_auc_score(y[i], score[i]) for i in draws]),
            "baseline_auc": round(float(roc_auc_score(y, base)), 3),
            "auc_minus_baseline": round(float(roc_auc_score(y, score) - roc_auc_score(y, base)), 3),
            "difference_range95": r95([roc_auc_score(y[i], score[i]) - roc_auc_score(y[i], base[i]) for i in draws]),
            "local_auc": round(float(loc_m[has].mean()), 3), "local_auc_baseline": round(float(loc_b[has].mean()), 3), "positives_with_a_near_negative": int(has.sum()),
            "local_auc_range95": r95([np.nanmean(loc_m[i]) for i in draws]),
            "ducts_with_30_cells": len(ducts),
            "duct_rank_correlation": round(float(spearmanr(share, pred)[0]), 3) if len(ducts) >= 8 else None,
            "duct_rank_correlation_baseline": round(float(spearmanr(share, pred_b)[0]), 3) if len(ducts) >= 8 else None,
            "duct_rank_correlation_range95": r95([spearmanr(share[i], pred[i])[0] for i in dd]) if dd else None}
        np.savez(OUT / f"gate1_scores_{held}.npz", score=score, base=base, y=y, xy=b["xy"], duct=b["duct"])
        print(held, json.dumps(result["held_out"][held]), flush=True)
    (OUT / "gate1.json").write_text(json.dumps(result, indent=1))


def test_cell():
    """Ledger entry 52: gate 1 again with each labelled cell described by its own picture. Same cells, labels, baseline and rule
    as test(); next to the main model, three variants that show which change did what."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    names = [s for s in SAMPLES if (DATA / s / "features_cell.npz").exists()]
    data = {s: load(s, cell=True) for s in names}
    variants = {"picture: centre and field": lambda a: np.hstack([a["centre"], a["field"]]), "centre tokens alone": lambda a: a["centre"],
                "all-token mean alone": lambda a: a["field"], "entry 48's features, wider grid": lambda a: a["x"]}
    rng = np.random.default_rng(0)
    r95 = lambda v: [round(float(q), 3) for q in np.percentile(v, [2.5, 97.5])]
    result = {"samples": {s: dict(patient=SAMPLES[s]["patient"], **data[s]["counts"]) for s in names}, "held_out": {}}
    for held in names:
        b, train = data[held], [s for s in names if s != held]
        y = b["y"]
        ya = np.hstack([data[s]["y"] for s in train])
        ga = np.hstack([data[s]["block"] + k * 10 ** 9 for k, s in enumerate(train)])
        base, _ = fit_score(np.vstack([data[s]["geo"] for s in train]), ya, ga, b["geo"])
        draws = [i for i in block_draws(b["block"], rng) if 0 < y[i].sum() < len(i)]
        old = np.load(OUT / f"gate1_scores_{held}.npz")   # entry 48's scores, to compare on the same cells
        assert (old["y"] == y).all() and np.allclose(old["xy"], b["xy"])
        loc_old, loc_b = local_auc(old["score"], y, b["xy"]), local_auc(base, y, b["xy"])
        out = {"patient": SAMPLES[held]["patient"], "trained_on": train, "cells": int(len(y)), "positive": int(y.sum()),
               "baseline_auc": round(float(roc_auc_score(y, base)), 3), "local_auc_baseline": round(float(np.nanmean(loc_b)), 3),
               "entry48_auc": round(float(roc_auc_score(y, old["score"])), 3), "entry48_local_auc": round(float(np.nanmean(loc_old)), 3), "variants": {}}
        for name, fx in variants.items():
            score, c_model = fit_score(np.vstack([fx(data[s]) for s in train]), ya, ga, fx(b), WIDE)
            loc = local_auc(score, y, b["xy"])
            out["variants"][name] = {
                "C": c_model, "auc": round(float(roc_auc_score(y, score)), 3), "auc_range95": r95([roc_auc_score(y[i], score[i]) for i in draws]),
                "auc_minus_baseline": round(float(roc_auc_score(y, score) - roc_auc_score(y, base)), 3),
                "difference_range95": r95([roc_auc_score(y[i], score[i]) - roc_auc_score(y[i], base[i]) for i in draws]),
                "local_auc": round(float(np.nanmean(loc)), 3), "local_auc_range95": r95([np.nanmean(loc[i]) for i in draws]),
                "local_auc_minus_entry48": round(float(np.nanmean(loc) - np.nanmean(loc_old)), 3),
                "local_difference_range95": r95([np.nanmean(loc[i]) - np.nanmean(loc_old[i]) for i in draws])}
            if name == "picture: centre and field":
                ducts = [i for i in (np.flatnonzero(b["duct"] == d) for d in np.unique(b["duct"])) if len(i) >= 30]
                if len(ducts) >= 8:
                    out["variants"][name]["duct_rank_correlation"] = round(float(spearmanr([y[i].mean() for i in ducts], [score[i].mean() for i in ducts])[0]), 3)
                np.savez(OUT / f"gate1_cell_scores_{held}.npz", score=score, base=base, y=y, xy=b["xy"], duct=b["duct"])
            print(held, name, json.dumps(out["variants"][name]), flush=True)
        result["held_out"][held] = out
    (OUT / "gate1_cell.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"cells": cells, "align": align, "shift": shift, "features": features, "cellfeatures": cellfeatures,
     "test": lambda mode=None: test_cell() if mode == "cell" else test()}[sys.argv[1]](*sys.argv[2:])
