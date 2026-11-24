#!/usr/bin/env python3
"""Ledger entry 39: train UNI itself on the labelled tiles (LoRA: small trainable additions inside the frozen model), one patient fold held out.

Usage (from the project folder):
  python -I tools/grayzone_finetune.py <fold 0-4> [--epochs 4] [--smoke]   # -> outputs/grayzone/finetune_fold<k>.json

The held-out fold's patients are never seen by the training. Two read-outs on the held-out regions, each against plain UNI:
(1) the project's classifier on the tuned model's tile averages; (2) the tuned model's own answer, averaged over a region's tiles.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import extract_features as ef
import folds as foldlib
import grayzone_features as gf
import short_lists as sl
from train_classifier import STRENGTHS, classifier, patient_draws, pooled, range95, rows, weighted_f1

SRC = sl.ROOT / "data/bracs/BRACS_RoI/latest_version"
RANK, ALPHA, FIRST_BLOCK, PER_REGION = 8, 16, 12, 8


class LoRA(nn.Module):
    """A frozen linear layer plus a small trainable low-rank addition."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, RANK, bias=False)
        self.up = nn.Linear(RANK, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * (ALPHA / RANK)


class Tiles(torch.utils.data.Dataset):
    """One item = up to PER_REGION random tissue tiles of one region, with flips and quarter turns."""

    def __init__(self, files, labels, seed):
        self.files, self.labels, self.seed = files, labels, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i, int(torch.randint(0, 2 ** 31, (1,)))])
        tiles, _, tissue, _, _ = ef.region_tiles(SRC / self.files[i])
        keep = np.flatnonzero(tissue >= 0.5)
        keep = keep if len(keep) else np.arange(len(tiles))
        t = tiles[rng.choice(keep, min(PER_REGION, len(keep)), replace=False)]
        t = np.stack([np.rot90(x, rng.integers(4)) if rng.random() < 0.75 else x for x in t])
        if rng.random() < 0.5:
            t = t[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(t)), torch.full((len(t),), self.labels[i])


def collate(items):
    return torch.cat([a for a, _ in items]), torch.cat([b for _, b in items]), [len(a) for a, _ in items]


def embed_all(model, head, files, dev, mean, std, log, region=False):
    """Per region: the tile average of the features, and the average of the head's log-probabilities
    (with region=True: the head's log-probabilities for the tile average, as the head was trained in entry 57)."""
    feats, logp = [], []
    loader = torch.utils.data.DataLoader(ef.Regions([SRC / f for f in files]), batch_size=None, shuffle=False, num_workers=4)
    model.eval()
    with torch.inference_mode():
        for i, tiles, coords, tissue, size, padded in loader:
            keep = np.asarray(tissue) >= 0.5
            keep = keep if keep.any() else np.ones(len(keep), bool)
            out = []
            for s in range(0, len(tiles), 64):
                x = tiles[s:s + 64].to(dev).permute(0, 3, 1, 2).float().div_(255)
                out.append(model((x - mean) / std))
            f = torch.cat(out)[torch.from_numpy(keep).to(dev)]
            feats.append(f.mean(0).cpu().numpy())
            logp.append((torch.log_softmax(head(f.mean(0, keepdim=True)), 1)[0] if region else torch.log_softmax(head(f), 1).mean(0)).cpu().numpy())
            if log and i % 500 == 0:
                print("  embedded", i, "/", len(files), flush=True)
    return np.array(feats), np.array(logp)


def best_c(x, y, patient):
    inner = list(foldlib.split(4, y, patient))
    scores = {}
    for c in STRENGTHS:
        p = np.zeros((len(y), 7))
        for a, b in inner:
            p[b] = classifier(c).fit(x[a], y[a]).predict_proba(x[b])
        scores[c] = gf.gray_f1(p, y)[0]
    return max(scores, key=scores.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--region-loss", action="store_true", help="entry 57: one loss per region on the average of its tiles, not one per tile")
    args = ap.parse_args()
    tag = "finetune_region" if args.region_loss else "finetune"
    train = rows("train")
    files = [r["file"] for r in train]
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    a, b = folds[args.fold]
    if args.smoke:
        a, b = a[:24], b[:12]
    dev = torch.device("mps" if torch.backends.mps.is_available() and not args.smoke else "cpu")
    torch.manual_seed(0)
    model, _, _, _ = ef.load_model("uni")
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks[FIRST_BLOCK:]:
        blk.attn.qkv = LoRA(blk.attn.qkv)
    head = nn.Linear(1024, 7)
    model, head = model.to(dev), head.to(dev)
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    loader = torch.utils.data.DataLoader(Tiles([files[i] for i in a], y[a], 0), batch_size=3, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
    weight = torch.tensor(len(a) / (7 * np.bincount(y[a], minlength=7).clip(1)), dtype=torch.float32, device=dev)
    print(f"fold {args.fold}: {len(a)} training regions, {len(b)} held-out regions, {sum(p.numel() for p in lora):,} trainable numbers, {steps} steps on {dev}", flush=True)
    t0, n, run = time.time(), 0, 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, lab, sizes in loader:
            x = x.to(dev).permute(0, 3, 1, 2).float().div_(255)
            f = model((x - mean) / std)
            if args.region_loss:   # the average of a region's tiles carries the region's label; a single tile carries none
                f, lab = torch.stack([g.mean(0) for g in f.split(sizes)]), torch.stack([g[0] for g in lab.split(sizes)])
            loss = nn.functional.cross_entropy(head(f), lab.to(dev), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            n += 1; run = 0.98 * run + 0.02 * float(loss) if n > 1 else float(loss)
            if n % 100 == 0:
                print(f"  step {n}/{steps} loss {run:.3f} {(time.time() - t0) / 60:.0f} min", flush=True)
            if args.smoke and n >= 2:
                break
        if args.smoke:
            break
    xa, la = embed_all(model, head, [files[i] for i in a], dev, mean, std, True, args.region_loss)
    xb, lb = embed_all(model, head, [files[i] for i in b], dev, mean, std, True, args.region_loss)
    if args.smoke:  # the smoke run only checks that training and embedding work
        print("smoke ok:", xa.shape, la.shape, xb.shape, bool(np.isfinite(xb).all()))
        return
    np.savez(sl.ROOT / f"outputs/grayzone/{tag}_fold{args.fold}_raw.npz", a=a, b=b, xa=xa, la=la, xb=xb, lb=lb)   # the run is kept even if the scoring below fails
    base = pooled("uni", "train", 0.5)
    result = {"fold": args.fold, "training_regions": int(len(a)), "held_out_regions": int(len(b)), "held_out_patients": int(len(np.unique(patient[b]))),
              "epochs": args.epochs, "minutes": round((time.time() - t0) / 60)}
    probs = {}
    for name, tr, te in (("plain UNI", base[a], base[b]), ("tuned UNI, project classifier", xa, xb)):
        c = best_c(tr, y[a], patient[a]) if not args.smoke else 0.001
        probs[name] = classifier(c).fit(tr, y[a]).predict_proba(te)
    probs["tuned UNI, own answer"] = np.exp(lb)
    probs["adopted classifier"] = sl.load()[0][b]   # entry 57: the bar to beat (tile average next to the whole-region view), out of fold
    from grayzone_slide_origin import SEVERITY, slide_labels
    lab = slide_labels()
    grp_a = np.array([SEVERITY[lab[train[i]["slide"]]] == 4 for i in b]) & (y[b] == 4)   # ADH on a slide without carcinoma
    named = lambda p, m: round(float((np.array(gf.GRAY)[p[m][:, gf.GRAY].argmax(1)] == 4).mean()), 4) if m.any() else None
    result["held_out_adh"] = {"group_A_regions": int(grp_a.sum()), "group_C_regions": int(((y[b] == 4) & ~grp_a).sum())}
    draws = patient_draws(patient[b], np.random.default_rng(0))
    for name, p in probs.items():
        macro, adh = gf.gray_f1(p, y[b])
        result[name] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "weighted_f1_7_types": round(weighted_f1(y[b], p.argmax(1)), 4),
                        "group_A_named_ADH": named(p, grp_a), "group_C_named_ADH": named(p, (y[b] == 4) & ~grp_a)}
        for ref in ("plain UNI", "adopted classifier"):
            if name not in ("plain UNI", "adopted classifier"):
                dm = [gf.gray_f1(p, y[b], i)[0] - gf.gray_f1(probs[ref], y[b], i)[0] for i in draws]
                da = [gf.gray_f1(p, y[b], i)[1] - gf.gray_f1(probs[ref], y[b], i)[1] for i in draws]
                result[name][f"minus {ref}"] = {"gray_macro_f1": round(macro - gf.gray_f1(probs[ref], y[b])[0], 4), "gray_range95": range95(dm),
                                                 "adh_f1": round(adh - gf.gray_f1(probs[ref], y[b])[1], 4), "adh_range95": range95(da)}
    if not args.smoke:
        np.savez(sl.ROOT / f"outputs/grayzone/{tag}_fold{args.fold}.npz", held=b, **{k.replace(" ", "_").replace(",", ""): v for k, v in probs.items()})
        (sl.ROOT / f"outputs/grayzone/{tag}_fold{args.fold}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
