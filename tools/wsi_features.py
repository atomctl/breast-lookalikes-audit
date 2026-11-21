#!/usr/bin/env python3
"""Whole-slide features, one slide at a time: download, embed every tissue tile with UNI at 20x, save, delete the slide.

Usage (from the project folder):
  python -I tools/wsi_features.py <list.json>   # a JSON list of remote paths such as "BRACS_WSI/train/Group_AT/Type_ADH/BRACS_1499.svs"

Output: data/features/uni_wsi/<slide>.npz with feats (n, 1024) float16, coords (n, 2) in level-0 pixels, step (level-0 pixels per tile), mpp.
Resumable: slides that already have an .npz are skipped. The next slide downloads while the current one is read,
so at most two slide files are on disk at any time.
"""
import ftplib
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path

ROOT = pathlib.Path(__file__).resolve().parent.parent
TMP = ROOT / "data/bracs_wsi"
OUT = ROOT / "data/features/uni_wsi"
HOST, USER, PASS = "histoimage.na.icar.cnr.it", "utente_reader", "ICAR-CNR"  # public read-only login from the BRACS download page
TILE = 224


def download(remote, local):
    """Fetch one slide; after a network break, carry on from the bytes already on disk."""
    for attempt in range(40):
        try:
            ftp = ftplib.FTP(HOST, timeout=180)
            ftp.login(USER, PASS)
            ftp.voidcmd("TYPE I")
            size = ftp.size(remote)
            have = local.stat().st_size if local.exists() else 0
            if have > size:
                local.unlink()
                have = 0
            if have < size:
                with open(local, "ab") as f:
                    ftp.retrbinary(f"RETR {remote}", f.write, blocksize=1 << 20, rest=have or None)
            if local.stat().st_size == size:
                return size
        except Exception as e:  # network trouble: reconnect and resume
            print("download problem:", type(e).__name__, e, flush=True)
            time.sleep(20)
    raise RuntimeError(f"could not download {remote}")


def tissue_grid(slide):
    """Level-0 corners of the 20x tiles that are at least half tissue, judged on a small overview of the slide."""
    w, h = slide.dimensions
    mpp = float(slide.properties.get("tiffslide.mpp-x") or 0.25)
    step = round(TILE * 0.5 / mpp)  # level-0 pixels covered by one 224-px tile at 0.5 um per pixel
    level = slide.get_best_level_for_downsample(32)
    lw, lh = slide.level_dimensions[level]
    small = np.asarray(slide.read_region((0, 0), level, (lw, lh)).convert("RGB"), dtype=np.int16)
    total = small.sum(2)
    tissue = (total < 660) & (total > 90)  # the regions' rule (mean colour under 220), minus black background
    scale = w / lw
    keep = []
    for y in range(0, h - step + 1, step):
        y0, y1 = int(y / scale), max(int((y + step) / scale), int(y / scale) + 1)
        for x in range(0, w - step + 1, step):
            x0, x1 = int(x / scale), max(int((x + step) / scale), int(x / scale) + 1)
            if tissue[y0:y1, x0:x1].mean() >= 0.5:
                keep.append((x, y))
    return np.array(keep, np.int64).reshape(-1, 2), step, mpp


def main():
    import timm
    import tiffslide
    import torch
    from PIL import Image

    import extract_features as ef
    todo = json.loads(pathlib.Path(sys.argv[1]).read_text())
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model, norm, embed, _ = ef.load_model("uni")
    cfg = timm.data.resolve_model_data_config(model)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.eval().to(dev).half()
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    pool = ThreadPoolExecutor(8)
    todo = [r for r in todo if not (OUT / f"{pathlib.Path(r).stem}.npz").exists()]

    def fetch(remote):
        t = time.time()
        size = download(remote, TMP / f"{pathlib.Path(remote).stem}.svs")
        return size, time.time() - t

    fetcher = ThreadPoolExecutor(1)
    ahead = fetcher.submit(fetch, todo[0]) if todo else None
    for n, remote in enumerate(todo):
        name = pathlib.Path(remote).stem
        dest = OUT / f"{name}.npz"
        local = TMP / f"{name}.svs"
        size, took = ahead.result()
        ahead = fetcher.submit(fetch, todo[n + 1]) if n + 1 < len(todo) else None
        t1 = time.time()
        slide = tiffslide.TiffSlide(local)
        coords, step, mpp = tissue_grid(slide)

        def read(xy):
            im = slide.read_region((int(xy[0]), int(xy[1])), 0, (step, step)).convert("RGB")
            return np.asarray(im.resize((TILE, TILE), Image.BOX))

        feats = []
        for b in range(0, len(coords), 96):
            tiles = np.stack(list(pool.map(read, coords[b:b + 96])))
            with torch.inference_mode():
                x = torch.from_numpy(tiles).to(dev).permute(0, 3, 1, 2).float().div_(255)
                feats.append(model(((x - mean) / std).half()).float().cpu().numpy().astype(np.float16))
        feats = np.concatenate(feats) if feats else np.zeros((0, 1024), np.float16)
        assert np.isfinite(feats).all()
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, feats=feats, coords=coords, step=step, mpp=mpp, dimensions=np.array(slide.dimensions), remote=remote)
        tmp.rename(dest)
        slide.close()
        local.unlink()
        print(f"{n + 1}/{len(todo)} {name}: {size / 1e9:.2f} GB, {len(coords)} tissue tiles, download {took / 60:.1f} min, embed {(time.time() - t1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
