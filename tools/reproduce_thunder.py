#!/usr/bin/env python3
"""M1 reproduction: rerun the THUNDER benchmark's linear-probing protocol on BRACS with our own code.

THUNDER (github.com/MICS-Lab/thunder, read at commit 38905bc, arXiv 2507.07860) publishes linear-probing
F1 on BRACS: UNI 59.4, H-optimus-0 61.1. Its protocol, as read from its source:
  - the WHOLE region image goes through the model's own timm transform (shorter side resized to 224, centre crop);
    there is no tiling (`div_patches: False` for bracs);
  - the model's pooled output is the embedding;
  - a linear layer is trained with Adam for 200 epochs, batch 64, plain cross-entropy,
    for every pair of lr in (1e-3, 1e-4, 1e-5) and weight decay in (0, 1e-3, 1e-4);
  - the best epoch and pair are chosen by macro F1 on validation, and macro F1 is reported on the test set.

Usage (from the project folder):
  python -I tools/reproduce_thunder.py embed <model>         # uni or hoptimus0 -> outputs/m1/thunder/<model>_<split>.npy
  python -I tools/reproduce_thunder.py probe <model>         # trains, prints validation macro F1 only
  python -I tools/reproduce_thunder.py probe <model> --final-test   # also scores the test set: only in the final test pass
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/bracs/BRACS_RoI/latest_version"
OUT = ROOT / "outputs/m1/thunder"
PUBLISHED = {"uni": 59.4, "hoptimus0": 61.1}


def split_rows(split):
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def embed(key):
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from extract_features import load_model
    Image.MAX_IMAGE_PIXELS = 200_000_000
    model, _, embed_fn, source = load_model(key)
    assert embed_fn is None  # both models here use the plain pooled output
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print(source, transform, flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.eval().to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:  # embeddings only; nothing is scored here
        dest = OUT / f"{key}_{split}.npy"
        if dest.exists():
            continue
        rows, x = split_rows(split), []
        for n in range(0, len(rows), 32):
            batch = torch.stack([transform(Image.open(IMAGES / r["file"]).convert("RGB")) for r in rows[n:n + 32]])
            with torch.no_grad():
                x.append(model(batch.to(device)).float().cpu().numpy())
            print(split, n + len(batch), "/", len(rows), flush=True)
        x = np.concatenate(x)
        assert np.isfinite(x).all() and len(x) == len(rows)
        np.save(dest, x)


def probe(key, final_test):
    from sklearn.metrics import f1_score
    torch.manual_seed(0)
    load = lambda s: (torch.from_numpy(np.load(OUT / f"{key}_{s}.npy")), torch.tensor([int(r["label"]) for r in split_rows(s)]))
    xt, yt = load("train")
    xv, yv = load("val")
    grid = [(lr, wd) for lr in (1e-3, 1e-4, 1e-5) for wd in (0.0, 1e-3, 1e-4)]
    heads = [torch.nn.Linear(xt.shape[1], 7) for _ in grid]
    opts = [torch.optim.Adam(h.parameters(), lr=lr, weight_decay=wd) for h, (lr, wd) in zip(heads, grid)]
    loss_fn = torch.nn.CrossEntropyLoss()
    best = (-1.0, None, None, None)  # validation macro F1, pair, epoch, weights
    for epoch in range(200):
        order = torch.randperm(len(xt))
        for n in range(0, len(xt), 64):
            i = order[n:n + 64]
            for h, o in zip(heads, opts):
                o.zero_grad()
                loss_fn(h(xt[i]), yt[i]).backward()
                o.step()
        with torch.no_grad():
            for h, pair in zip(heads, grid):
                f1 = f1_score(yv, h(xv).argmax(1), average="macro")
                if f1 > best[0]:
                    best = (f1, pair, epoch, {k: v.clone() for k, v in h.state_dict().items()})
    result = {"model": key, "published_test_macro_f1": PUBLISHED[key], "val_macro_f1": round(100 * best[0], 1),
              "lr": best[1][0], "weight_decay": best[1][1], "epoch": best[2]}
    if final_test:
        xs, ys = load("test")
        head = torch.nn.Linear(xt.shape[1], 7)
        head.load_state_dict(best[3])
        with torch.no_grad():
            pred = head(xs).argmax(1)
        result["our_test_macro_f1"] = round(100 * f1_score(ys, pred, average="macro"), 1)
        result["our_test_weighted_f1"] = round(100 * f1_score(ys, pred, average="weighted"), 1)
    name = f"result_{key}_test.json" if final_test else f"result_{key}_val.json"
    (OUT / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2])
    else:
        probe(sys.argv[2], "--final-test" in sys.argv)
