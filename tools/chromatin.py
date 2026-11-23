"""Ledger entry 74: chromatin texture of saved nuclei, gates 1 and 2 before any UDH-ADH test.

    python -I tools/chromatin.py gates
"""
import csv
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data/bracs/BRACS_RoI/latest_version"
CELLS = ROOT / "data/features/duct_cells"
OUT = ROOT / "outputs/chromatin"
NAMES = ["h_mean", "h_std", "h_p90_p10", "h_entropy", "h_laplace_var"]
Image.MAX_IMAGE_PIXELS = 400_000_000
# Ruifrok & Johnston stain vectors (as in skimage.color.rgb2hed)
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def haematoxylin(rgb):
    od = -np.log(np.maximum(rgb.astype(float), 1) / 255.0)
    return (od @ HED_FROM_RGB)[..., 0]


def key(f):
    return f[:-4].replace("/", "__")


def textures(file, max_nuclei=None, rng=None):
    n = np.load(CELLS / (key(file) + ".npz"))["nuclei"]
    if max_nuclei and len(n) > max_nuclei:
        n = n[rng.choice(len(n), max_nuclei, replace=False)]
    img = Image.open(SRC / file).convert("RGB")
    out = []
    for x, y, area, *_ in n:
        r = max(int(np.ceil(np.sqrt(area / np.pi))), 3)
        x0, y0 = int(round(x)) - r - 1, int(round(y)) - r - 1
        if x0 < 1 or y0 < 1 or x0 + 2 * r + 3 >= img.width or y0 + 2 * r + 3 >= img.height:
            continue
        h = haematoxylin(np.asarray(img.crop((x0, y0, x0 + 2 * r + 3, y0 + 2 * r + 3))))
        yy, xx = np.mgrid[:h.shape[0], :h.shape[1]]
        disc = (yy - r - 1) ** 2 + (xx - r - 1) ** 2 <= r * r
        v = h[disc]
        lap = (h[1:-1, 1:-1] * 4 - h[:-2, 1:-1] - h[2:, 1:-1] - h[1:-1, :-2] - h[1:-1, 2:])[disc[1:-1, 1:-1]]
        p, _ = np.histogram(v, bins=16, range=(0, 1.5))
        p = p[p > 0] / p.sum()
        out.append([np.log(area), v.mean(), v.std(), np.percentile(v, 90) - np.percentile(v, 10), -(p * np.log2(p)).sum(), lap.var()])
    return np.array(out)


def gates():
    t0 = time.time()
    with open(ROOT / "data/regions.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train" and (CELLS / (key(r["file"]) + ".npz")).exists()]
    rng = np.random.default_rng(0)
    # gate 1: 200 nuclei from the first two slides (by name) that carry both UDH and ADH nuclei
    by = {}
    for r in rows:
        by.setdefault(r["slide"], []).append(r)
    two = [s for s in sorted(by) if {"UDH", "ADH"} <= {r["label_name"] for r in by[s]}][:2]
    nuc = np.vstack([textures(r["file"], 25, rng) for s in two for r in by[s] if r["label_name"] in ("UDH", "ADH")])
    nuc = nuc[rng.choice(len(nuc), min(200, len(nuc)), replace=False)]
    g1 = {}
    for j, name in enumerate(NAMES, start=1):
        v, la = nuc[:, j], nuc[:, 0]
        cv = float(v.std() / abs(v.mean())) if v.mean() else float("inf")
        r2 = float(np.corrcoef(v, la)[0, 1] ** 2)
        g1[name] = {"cv": round(cv, 4), "r2_from_log_area": round(r2, 4), "ok": cv > 0.05 and r2 < 0.9}
    g1_pass = sum(v["ok"] for v in g1.values()) >= 3
    result = {"entry": 74, "gate1": {"slides": two, "nuclei": len(nuc), "features": g1, "pass": g1_pass}}
    if not g1_pass:
        result["stop"] = "gate 1 failed"
    else:
        # gate 2: UDH against DCIS regions, region medians, patient folds
        pick = []
        for lab in ("UDH", "DCIS"):
            cand = [r for r in rows if r["label_name"] == lab]
            pick += [cand[i] for i in rng.choice(len(cand), min(150, len(cand)), replace=False)]
        X, y, g = [], [], []
        for r in pick:
            t = textures(r["file"], 150, rng)
            if len(t) < 10:
                continue
            X.append(np.median(t[:, 1:], 0)); y.append(r["label_name"] == "DCIS"); g.append(int(r["patient"]))
        X, y, g = np.array(X), np.array(y), np.array(g)
        prob = np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, y, g):
            prob[b] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        auc = float(roc_auc_score(y, prob))
        result["gate2"] = {"regions": len(y), "DCIS": int(y.sum()), "UDH": int((~y).sum()), "patients": len(np.unique(g)), "auc": round(auc, 4), "pass": auc >= 0.75}
    result["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def pairs():
    """Ledger entry 75: chromatin score on UDH-ADH close and far pairs."""
    sys.path.insert(0, str(ROOT / "tools"))
    from within_slide import gap_mm, ADH, UDH, RESAMPLES, SEED
    t0 = time.time()
    with open(ROOT / "data/regions.csv") as f:
        allrows = list(csv.DictReader(f))
    tr = [r for r in allrows if r["split"] == "train"]; va = [r for r in allrows if r["split"] == "val"]
    rows = [r for r in tr if r["label_name"] in ("UDH", "ADH") and (CELLS / (key(r["file"]) + ".npz")).exists()]
    rng = np.random.default_rng(0)
    X, y, g, files = [], [], [], []
    for r in rows:
        t = textures(r["file"], 150, rng)
        if len(t) < 10:
            continue
        X.append(np.median(t[:, 1:], 0)); y.append(r["label_name"] == "ADH"); g.append(int(r["patient"])); files.append(r["file"])
    X, y, g = np.array(X), np.array(y), np.array(g)
    pats = np.unique(g); order = np.random.default_rng(SEED).permutation(pats); grp = {p: k % 5 for k, p in enumerate(order)}
    fold = np.array([grp[p] for p in g]); score = np.zeros(len(y))
    for k in range(5):
        a, b = fold != k, fold == k
        score[b] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    chrom = dict(zip(files, score))
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    eng = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(tr + va)}
    pt = {r["slide"]: int(r["patient"]) for r in allrows}
    T = []
    for s_, regs in json.loads((ROOT / "outputs/within_slide/positions.json").read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                if a["file"] not in chrom or u["file"] not in chrom:
                    continue
                gap = gap_mm(a["box"], u["box"]); band = 0 if gap <= 1 else 1 if gap > 3 else -1
                if band < 0:
                    continue
                dc = chrom[a["file"]] - chrom[u["file"]]; de = eng[a["file"]] - eng[u["file"]]
                T.append((pt[s_], band, 1.0 if dc > 0 else 0.5 if dc == 0 else 0.0, 1.0 if de > 0 else 0.5 if de == 0 else 0.0))
    T = np.array(T)
    def rep(mask):
        pp = np.unique(T[mask, 0]); idx = {p: np.where(mask & (T[:, 0] == p))[0] for p in pp}
        per = np.array([T[idx[p], 2].mean() - T[idx[p], 3].mean() for p in pp])
        r2 = np.random.default_rng(SEED); draws = []
        for _ in range(RESAMPLES):
            ch = r2.choice(len(pp), len(pp)); ii = np.concatenate([idx[pp[c]] for c in ch])
            draws.append((T[ii, 2].mean(), T[ii, 3].mean(), per[ch].mean()))
        q = lambda j: [round(float(np.percentile([d[j] for d in draws], v)), 4) for v in (2.5, 97.5)]
        return {"pairs": int(mask.sum()), "patients": len(pp), "chromatin": round(float(T[mask, 2].mean()), 4), "chromatin_range95": q(0),
                "engine": round(float(T[mask, 3].mean()), 4), "engine_range95": q(1), "paired_chromatin_minus_engine": round(float(per.mean()), 4), "paired_range95": q(2)}
    result = {"entry": 75, "region_auc_oof": round(float(roc_auc_score(y, score)), 4), "regions": len(y),
              "positive_control_far": rep(T[:, 1] == 1), "primary_close": rep(T[:, 1] == 0)}
    flo = result["positive_control_far"]["chromatin_range95"][0]; lo, hi = result["primary_close"]["chromatin_range95"]
    result["reading"] = ("positive control failed (far pairs not above 0.5); close not read" if flo <= 0.5 else
                         "chromatin orients touching pairs" if lo > 0.5 else "range contains 0.5: does not orient touching pairs at this sample")
    result["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "pairs.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    pairs() if "pairs" in sys.argv else gates()
