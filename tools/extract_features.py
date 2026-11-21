#!/usr/bin/env python3
"""Turn every BRACS region into tile features with one frozen model.

Usage:
  python -I extract_features.py <latest_version_dir> <out_dir> <model>
         [--limit N] [--check-fp32] [--precision half|autocast|fp32] [--workers 4] [--batch 64]

Models: resnet50, vit_l_imagenet, uni, virchow2, uni2_h, hoptimus0
(loading code and normalisation follow each model card).

Tiles:
- 20x zoom: each 40x region is halved (2x2 pixel average, PIL reduce(2)).
- 224-px tiles on a grid, plus one edge-aligned tile per row/column for the remainder,
  so every pixel is seen (edge tiles overlap their neighbour).
- Regions under 224 px are padded with white.
- Coordinates are saved, so training can later drop edge or blank tiles without re-extracting.

Output:
- One .npz per region, mirroring the input folders, holding:
  feats (n, D) float16, coords (n, 2) [x, y] at 20x, tissue (n,) share of non-white pixels,
  size [w, h] at 20x before padding, padded flag, nan_tiles count.
- Resumable: regions with a finished .npz are skipped; files are written to a temp name, then renamed.
- Progress: <out>/<model>/_extract.log, _status.json, _meta.json.
"""
import argparse
import json
import os
import pathlib
import struct
import sys
import time

import numpy as np
import timm
import torch
from PIL import Image
from timm.layers import SwiGLUPacked

TILE = 224
# Zoom: 2 = 20x (the default), 4 = 10x, 8 = 5x. An environment variable, so the loader's worker processes see it too.
# Other zooms are saved next to the 20x features, e.g. data/features/uni_10x.
REDUCE = int(os.environ.get("BRACS_REDUCE", "2"))
Image.MAX_IMAGE_PIXELS = 200_000_000  # largest BRACS region is ~173 MP


def load_model(key):
    """Return (model, (mean, std) or None to use the model's own config, embed_fn or None, source)."""
    if key == "resnet50":
        return timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0), None, None, \
            "timm resnet50.a1_in1k (ImageNet-1k)"
    if key == "vit_l_imagenet":
        return timm.create_model("vit_large_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0), \
            None, None, "timm vit_large_patch16_224.augreg_in21k_ft_in1k (ImageNet-21k then 1k)"
    if key == "uni":
        m = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        return m, None, None, "hf-hub:MahmoodLab/uni"
    if key == "uni2_h":
        kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                  mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
                  act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
        return timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw), None, None, \
            "hf-hub:MahmoodLab/UNI2-h"
    if key == "virchow2":
        m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked,
                              act_layer=torch.nn.SiLU)

        def embed(out):  # model card: class token + mean of patch tokens; tokens 1-4 are registers
            return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)
        return m, None, embed, "hf-hub:paige-ai/Virchow2"
    if key == "hoptimus0":
        m = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                              dynamic_img_size=False)
        return m, ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)), None, \
            "hf-hub:bioptimus/H-optimus-0"
    if key == "midnight":
        from transformers import AutoModel

        class Tokens(torch.nn.Module):  # the model card's usage: all output tokens
            def __init__(self):
                super().__init__()
                self.m = AutoModel.from_pretrained("kaiko-ai/midnight")

            def forward(self, x):
                return self.m(x).last_hidden_state

        def embed(out):  # model card: class token + mean of patch tokens
            return torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
        return Tokens(), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), embed, "kaiko-ai/midnight (Midnight-12k, MIT)"
    raise SystemExit(f"unknown model {key}")


def starts(length):
    """Tile start positions along one side: a 224-px grid plus one edge-aligned tile if needed."""
    if length <= TILE:
        return [0]
    s = list(range(0, length - TILE + 1, TILE))
    if s[-1] + TILE < length:
        s.append(length - TILE)
    return s


def expected_tiles(path):
    with open(path, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    w2, h2 = -(-w // REDUCE), -(-h // REDUCE)  # PIL reduce rounds odd sizes up
    return len(starts(max(w2, TILE))) * len(starts(max(h2, TILE)))


def region_tiles(path):
    a = np.asarray(Image.open(path).convert("RGB").reduce(REDUCE))
    h, w = a.shape[:2]
    padded = h < TILE or w < TILE
    if padded:
        b = np.full((max(h, TILE), max(w, TILE), 3), 255, np.uint8)
        b[:h, :w] = a
        a = b
    coords = [(x, y) for y in starts(a.shape[0]) for x in starts(a.shape[1])]
    tiles = np.stack([a[y:y + TILE, x:x + TILE] for x, y in coords])
    tissue = (tiles.sum(axis=3, dtype=np.uint16) < 660).mean(axis=(1, 2)).astype(np.float16)  # mean RGB < 220
    return tiles, np.array(coords, np.int32), tissue, np.array([w, h], np.int32), padded


class Regions(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        tiles, coords, tissue, size, padded = region_tiles(self.paths[i])
        return i, torch.from_numpy(tiles), coords, tissue, size, padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-fp32", action="store_true")
    ap.add_argument("--precision", default="half", choices=["half", "autocast", "fp32"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    src = pathlib.Path(args.src).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve() / (args.model if REDUCE == 2 else f"{args.model}_{40 // REDUCE}x")
    out.mkdir(parents=True, exist_ok=True)
    log_path, status_path = out / "_extract.log", out / "_status.json"

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + f"[{args.model}] " + msg
        with open(log_path, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    all_paths = sorted(src.rglob("*.png"))
    if args.limit:
        all_paths = all_paths[:args.limit]

    def dest(p):
        return out / p.relative_to(src).with_suffix(".npz")

    todo = [p for p in all_paths if not dest(p).exists()]
    tiles_todo = sum(expected_tiles(p) for p in todo)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, norm, embed, source = load_model(args.model)
    cfg = timm.data.resolve_model_data_config(model)
    mean, std = norm if norm else (cfg["mean"], cfg["std"])
    model = model.eval().to(dev)
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    def run(x_uint8, precision):
        x = x_uint8.to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean_t) / std_t
        with torch.inference_mode():
            if precision == "half":
                o = model(x.half())
            elif precision == "autocast":
                with torch.autocast(device_type=dev.type, dtype=torch.float16):
                    o = model(x)
            else:
                o = model(x)
            if embed:
                o = embed(o)
            return o.float().cpu()

    if args.check_fp32 and all_paths:
        t, *_ = region_tiles(all_paths[0])
        x = torch.from_numpy(t[:16])
        ref = run(x, "fp32")
        if args.precision == "half":
            model.half()
        test = run(x, args.precision)
        cos = torch.nn.functional.cosine_similarity(ref, test, dim=1)
        log(f"check {args.precision} vs fp32 on {len(x)} tiles: cosine min {cos.min():.5f}, mean {cos.mean():.5f}; "
            f"nan {int(torch.isnan(test).any())}")
    elif args.precision == "half":
        model.half()

    meta = dict(model=args.model, source=source, mean=list(mean), std=list(std), precision=args.precision,
                zoom=f"{40 // REDUCE}x (40x reduced with PIL reduce({REDUCE}))", tile=TILE,
                tiling="224-px grid + one edge-aligned tile per row/column; regions under 224 px padded white",
                torch=torch.__version__, timm=timm.__version__, device=str(dev))
    (out / "_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{source}: {len(all_paths)} regions, {len(all_paths) - len(todo)} already done, "
        f"{len(todo)} to do = {tiles_todo:,} tiles; precision {args.precision}; mean {mean} std {std}")
    if not todo:
        log("nothing to do")

    loader = torch.utils.data.DataLoader(Regions(todo), batch_size=None, shuffle=False,
                                         num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    pending, queue = {}, []  # queue holds (region index, tile tensor chunk) in arrival order
    state = dict(tiles=0, regions=0, nan_regions=0, dim=None)
    t0 = last = time.time()

    def save(i):
        r = pending.pop(i)
        feats = torch.cat(r["feats"]).numpy()
        nan_tiles = int(np.isnan(feats).any(axis=1).sum())
        state["nan_regions"] += nan_tiles > 0
        state["dim"] = feats.shape[1]
        d = dest(todo[i])
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.stem + ".tmp.npz")
        np.savez(tmp, feats=feats.astype(np.float16), coords=r["coords"], tissue=r["tissue"],
                 size=r["size"], padded=np.array(r["padded"]), nan_tiles=np.array(nan_tiles))
        os.replace(tmp, d)
        state["regions"] += 1

    def process(n):
        taken, owners = [], []
        while n > 0:
            i, chunk = queue.pop(0)
            if len(chunk) > n:
                queue.insert(0, (i, chunk[n:]))
                chunk = chunk[:n]
            taken.append(chunk)
            owners.append((i, len(chunk)))
            n -= len(chunk)
        feats = run(torch.cat(taken), args.precision)
        k = 0
        for i, m in owners:
            pending[i]["feats"].append(feats[k:k + m])
            pending[i]["got"] += m
            k += m
            if pending[i]["got"] == pending[i]["need"]:
                save(i)
        state["tiles"] += len(feats)

    queued = 0
    for i, tiles, coords, tissue, size, padded in loader:
        i = int(i)
        pending[i] = dict(feats=[], got=0, need=len(tiles), coords=np.asarray(coords), tissue=np.asarray(tissue),
                          size=np.asarray(size), padded=bool(padded))
        queue.append((i, tiles))
        queued += len(tiles)
        while queued >= args.batch:
            process(args.batch)
            queued -= args.batch
        now = time.time()
        if now - last >= 60:
            last = now
            speed = state["tiles"] / (now - t0)
            eta = (tiles_todo - state["tiles"]) / speed / 3600 if speed else -1
            log(f"{state['regions']}/{len(todo)} regions, {state['tiles']:,}/{tiles_todo:,} tiles, "
                f"{speed:.1f} tiles/s, ETA {eta:.2f} h, regions with NaN {state['nan_regions']}")
            status_path.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **state,
                                                   regions_todo=len(todo), tiles_todo=tiles_todo,
                                                   tiles_per_s=round(speed, 2), eta_hours=round(eta, 2))))
    if queued:
        process(queued)

    done = sum(1 for p in all_paths if dest(p).exists())
    took = (time.time() - t0) / 3600
    log(f"DONE: {done}/{len(all_paths)} regions have features ({state['dim']} numbers per tile); "
        f"{state['tiles']:,} tiles in {took:.2f} h; regions with NaN {state['nan_regions']}")
    sys.exit(0 if done == len(all_paths) and not pending else 1)


if __name__ == "__main__":
    main()
