#!/usr/bin/env python3
"""Ledger entry 38 (a): keep the arrangement of a region's tiles: attention pooling and a small transformer with tile positions.

Usage (from the project folder):
  python -I tools/grayzone_arrangement.py   # -> outputs/grayzone/arrangement.json   (CPU, UNI tile features)
"""
import json
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import grayzone_features as gf
import short_lists as sl
from train_classifier import FEATS, patient_draws, pooled, range95, rows, weighted_f1

MAX_TILES, WIDTH, EPOCHS, STARTS = 256, 256, 40, 3
torch.set_num_threads(6)


def load_regions():
    out = []
    for r in rows("train"):
        z = np.load(FEATS / "uni" / (r["file"][:-4] + ".npz"))
        keep = z["tissue"] >= 0.5
        keep = keep if keep.any() else np.ones(len(keep), bool)
        out.append((z["feats"][keep].astype(np.float32), (z["coords"][keep] / 224).astype(np.float32)))
    return out


def batch(regions, idx, mean, std, rng=None):
    n = min(max(len(regions[i][0]) for i in idx), MAX_TILES)
    x, pos, mask = np.zeros((len(idx), n, 1024), np.float32), np.zeros((len(idx), n, 2), np.float32), np.zeros((len(idx), n), bool)
    for j, i in enumerate(idx):
        f, c = regions[i]
        if len(f) > n:
            pick = np.sort(rng.choice(len(f), n, replace=False)) if rng is not None else np.linspace(0, len(f) - 1, n).astype(int)
            f, c = f[pick], c[pick]
        x[j, :len(f)], pos[j, :len(f)], mask[j, :len(f)] = (f - mean) / std, c - c.min(0), True
    return torch.from_numpy(x), torch.from_numpy(pos), torch.from_numpy(mask)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(1024, WIDTH), nn.ReLU(), nn.Dropout(0.25))
        self.a, self.b, self.w = nn.Linear(WIDTH, 128), nn.Linear(WIDTH, 128), nn.Linear(128, 1)
        self.out = nn.Linear(WIDTH, 7)

    def forward(self, x, pos, mask):
        h = self.inp(x)
        score = self.w(torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).squeeze(-1).masked_fill(~mask, -1e9)
        return self.out((torch.softmax(score, 1).unsqueeze(-1) * h).sum(1))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(1024, WIDTH), nn.ReLU(), nn.Dropout(0.25))
        self.cls = nn.Parameter(torch.zeros(1, 1, WIDTH))
        layer = nn.TransformerEncoderLayer(WIDTH, 4, WIDTH * 2, dropout=0.25, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(WIDTH, 7)
        self.register_buffer("freq", torch.exp(torch.arange(0, WIDTH // 4) * (-np.log(1000.0) / (WIDTH // 4))))

    def forward(self, x, pos, mask):
        a = pos.unsqueeze(-1) * self.freq  # (regions, tiles, 2, WIDTH/4): each tile's column and row as waves
        h = self.inp(x) + torch.cat([a.sin(), a.cos()], -1).flatten(2)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        pad = torch.cat([torch.zeros(len(h), 1, dtype=torch.bool), ~mask], 1)
        return self.out(self.enc(h, src_key_padding_mask=pad)[:, 0])


def predict(model, regions, idx, mean, std):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            out.append(torch.softmax(model(*batch(regions, idx[s:s + 64], mean, std)), 1).numpy())
    return np.concatenate(out)


def fit(kind, regions, y, patient, a, seed):
    """Train on fold `a`; the pass is chosen on 20% of its patients; returns the model at its best pass."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ids = rng.permutation(np.unique(patient[a]))
    held = np.isin(patient[a], ids[:max(len(ids) // 5, 1)])
    tr, va = a[~held], a[held]
    flat = np.concatenate([regions[i][0] for i in tr])
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    weight = torch.tensor(len(tr) / (7 * np.bincount(y[tr], minlength=7).clip(1)), dtype=torch.float32)
    model = kind()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    best, state = -1, None
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), 32):
            i = order[s:s + 32]
            loss = nn.functional.cross_entropy(model(*batch(regions, i, mean, std, rng)), torch.from_numpy(y[i]), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        score = gf.gray_f1(predict(model, regions, va, mean, std), y[va])[0]
        if score > best:
            best, state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, mean, std


def main():
    train = rows("train")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    _, _, _, folds, *_ = sl.load()
    regions = load_regions()
    draws = patient_draws(patient, np.random.default_rng(0))
    base, _ = gf.out_of_fold(pooled("uni", "train", 0.5), y, patient, folds)
    result = {"A uni tile average": {"gray_macro_f1": round(gf.gray_f1(base, y)[0], 4), "adh_f1": round(gf.gray_f1(base, y)[1], 4),
                                     "weighted_f1_7_types": round(weighted_f1(y, base.argmax(1)), 4)}}
    print(result, flush=True)
    for name, kind in (("attention pooling", Attention), ("transformer with tile positions", Transformer)):
        prob = np.zeros((len(y), 7))
        for k, (a, b) in enumerate(folds):
            for seed in range(STARTS):
                model, mean, std = fit(kind, regions, y, patient, a, seed)
                prob[b] += predict(model, regions, b, mean, std) / STARTS
            print(name, "fold", k, "done", flush=True)
        macro, adh = gf.gray_f1(prob, y)
        d = [gf.gray_f1(prob, y, i)[0] - gf.gray_f1(base, y, i)[0] for i in draws]
        e = [gf.gray_f1(prob, y, i)[1] - gf.gray_f1(base, y, i)[1] for i in draws]
        result[name] = {"gray_macro_f1": round(macro, 4), "adh_f1": round(adh, 4), "weighted_f1_7_types": round(weighted_f1(y, prob.argmax(1)), 4),
                        "gray_macro_f1_minus_A": round(macro - result["A uni tile average"]["gray_macro_f1"], 4), "range95_of_difference": range95(d),
                        "adh_f1_minus_A": round(adh - result["A uni tile average"]["adh_f1"], 4), "adh_range95_of_difference": range95(e),
                        "adopted": bool(np.percentile(d, 2.5) > 0)}
        np.save(sl.ROOT / f"outputs/grayzone/oof_{name.replace(' ', '_')}.npy", prob)
        print(name, result[name], flush=True)
    np.save(sl.ROOT / "outputs/grayzone/oof_A_uni.npy", base)
    (sl.ROOT / "outputs/grayzone/arrangement.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
