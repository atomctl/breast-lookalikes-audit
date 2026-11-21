#!/usr/bin/env python3
"""Speed test: how fast can this Mac turn 224-px tiles into features?

The models are same-size stand-ins with random weights: speed depends on the
architecture's size, not on what the weights learned. Architectures follow the
public model cards (UNI2-h, Virchow2, H-optimus-0).

Usage: python -I speed_test.py <BRACS_RoI/latest_version dir>
"""
import gc
import os
import pathlib
import random
import struct
import sys
import time

import numpy as np
import timm
import torch
from PIL import Image
from timm.layers import SwiGLUPacked

ROOT = pathlib.Path(sys.argv[1])
Image.MAX_IMAGE_PIXELS = 200_000_000  # largest BRACS region is ~173 MP
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sync = torch.mps.synchronize if dev.type == "mps" else (lambda: None)
print(f"torch {torch.__version__}, timm {timm.__version__}, device {dev}, cpu cores {os.cpu_count()}", flush=True)

# exact tile counts from the PNG headers
files = sorted(ROOT.rglob("*.png"))
t20 = t40 = 0
for p in files:
    with open(p, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    t20 += (w // 2 // 224) * (h // 2 // 224)
    t40 += (w // 224) * (h // 224)
print(f"{len(files)} regions -> {t20:,} tiles at 20x, {t40:,} tiles at 40x (full 224-px tiles)", flush=True)

MODELS = [
    ("ResNet-50, ImageNet baseline", "resnet50", {}),
    ("ViT-L/16, size of UNI v1 / ImageNet ViT-L", "vit_large_patch16_224", {}),
    ("UNI2-h size (ViT-H/14, SwiGLU, 8 registers)", "vit_giant_patch14_224",
     dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
          mlp_ratio=2.66667 * 2, no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
          reg_tokens=8, dynamic_img_size=True)),
    ("Virchow2 size (ViT-H/14, SwiGLU, 4 registers)", "vit_huge_patch14_224",
     dict(img_size=224, init_values=1e-5, mlp_ratio=5.3375, reg_tokens=4, mlp_layer=SwiGLUPacked,
          act_layer=torch.nn.SiLU, dynamic_img_size=True)),
    ("H-optimus-0 size (ViT-g/14, 4 registers)", "vit_giant_patch14_reg4_dinov2",
     dict(img_size=224, init_values=1e-5, dynamic_img_size=False)),
]

BATCH, N_BATCHES = 40, 5  # 200 timed tiles per model
results = []
for label, arch, kw in MODELS:
    m = timm.create_model(arch, pretrained=False, num_classes=0, **kw)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    m = m.half().eval().to(dev)
    x = torch.randn(BATCH, 3, 224, 224, dtype=torch.float16, device=dev)
    with torch.inference_mode():
        for _ in range(2):  # warm-up
            out = m(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(N_BATCHES):
            out = m(x)
        sync()
        dt = time.perf_counter() - t0
    tps = BATCH * N_BATCHES / dt
    mem = torch.mps.driver_allocated_memory() / 1e9 if dev.type == "mps" else float("nan")
    feat = out.shape[-1]
    print(f"{label}: {params:,.0f}M params, {feat} features/tile, {tps:,.1f} tiles/s, "
          f"20x: {t20 / tps / 60:,.0f} min, 40x: {t40 / tps / 60:,.0f} min, GPU memory {mem:.1f} GB", flush=True)
    results.append((label, tps))
    del m, x, out
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()

# CPU side: decode real PNGs, halve to 20x, cut 224-px tiles (one core)
random.seed(0)
sample = random.sample([p for p in files if "/train/" in str(p)], 24)
n_tiles = 0
t0 = time.perf_counter()
for p in sample:
    im = Image.open(p).convert("RGB").reduce(2)
    a = np.asarray(im)
    th, tw = a.shape[0] // 224, a.shape[1] // 224
    if th and tw:
        tiles = a[: th * 224, : tw * 224].reshape(th, 224, tw, 224, 3).swapaxes(1, 2)
        n_tiles += th * tw
dt = time.perf_counter() - t0
print(f"CPU prep (decode PNG, halve, tile) on one core: {n_tiles} tiles from {len(sample)} real images "
      f"in {dt:.1f} s = {n_tiles / dt:,.0f} tiles/s", flush=True)
