"""Ledger entry 61: UDH against ADH with the slide held fixed.

Does the frozen engine (entry 10) rank an ADH region above a UDH region better inside one slide
than across patients? Per patient p: W_p (within-slide pairs) minus B_p (between-patient pairs
scored by the same classifier). Primary: mean of W_p - B_p over patients, range from resampling patients.

    python tools/within_slide.py          # tool check on toy data, then the real run
"""
import csv
import json
import pathlib
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/within_slide"
UDH, ADH = 2, 4  # index in TYPES = N, PB, UDH, FEA, ADH, DCIS, IC
RESAMPLES, SEED = 2000, 0


def share_right(adh, udh):
    """Share of (ADH, UDH) pairs where the ADH score is higher; a tie counts half."""
    if len(adh) == 0 or len(udh) == 0:
        return np.nan, 0
    d = adh[:, None] - udh[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean()), d.size


def per_patient(score, is_adh, slide, patient, group):
    """W_p and B_p for every patient, and the pair counts behind them."""
    out = {}
    for p in np.unique(patient):
        mine = patient == p
        # within: pairs on the same slide, pooled over p's slides
        right, n = 0.0, 0
        for s in np.unique(slide[mine]):
            m = mine & (slide == s)
            w, k = share_right(score[m & is_adh], score[m & ~is_adh])
            if k:
                right, n = right + w * k, n + k
        # between: one region of p, one of another patient, same classifier group
        g = group[mine][0]
        other = (patient != p) & (group == g)
        b1, k1 = share_right(score[mine & is_adh], score[other & ~is_adh])
        b2, k2 = share_right(score[other & is_adh], score[mine & ~is_adh])
        kb = k1 + k2
        bright = (0 if not k1 else b1 * k1) + (0 if not k2 else b2 * k2)
        out[int(p)] = {"W": right / n if n else np.nan, "B": bright / kb if kb else np.nan,
                       "within_pairs": n, "between_pairs": kb, "W_right": right, "B_right": bright, "group": str(g)}
    return out


def summarise(pp, rng):
    keys = [p for p in pp if not (np.isnan(pp[p]["W"]) or np.isnan(pp[p]["B"]))]
    W = np.array([pp[p]["W"] for p in keys]); B = np.array([pp[p]["B"] for p in keys])
    def stats(idx):
        sub = [pp[keys[i]] for i in idx]
        return {"delta": float((W[idx] - B[idx]).mean()), "W": float(W[idx].mean()), "B": float(B[idx].mean()),
                "W_pooled": sum(s["W_right"] for s in sub) / sum(s["within_pairs"] for s in sub),
                "B_pooled": sum(s["B_right"] for s in sub) / sum(s["between_pairs"] for s in sub)}
    point = stats(np.arange(len(keys)))
    draws = [stats(rng.integers(0, len(keys), len(keys))) for _ in range(RESAMPLES)]
    ranges = {k: [float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5))] for k in point}
    above = int((W > 0.5).sum()); below = int((W < 0.5).sum())
    sign_p = binomtest(above, above + below, 0.5, alternative="greater").pvalue if above + below else np.nan
    return {"patients": len(keys), **{k: round(v, 4) for k, v in point.items()},
            "range95": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ranges.items()},
            "sign_test": {"W_above_half": above, "W_below_half": below, "ties": len(keys) - above - below, "p_one_sided": float(sign_p)}}


def reading(lo, hi):
    if lo > 0:
        return "range above 0: holding the slide fixed helps; slide-level differences hide part of the signal"
    if hi < 0:
        return "range below 0: inside one slide UDH and ADH look more alike than across patients (disputed border); 'design artefact' rejected"
    return "range contains 0: no sign that the cross-slide design hides signal; the wall is in the features or the labels"


def tool_check():
    """Toy slides: perfect ranking inside each slide, hidden by a random offset per slide."""
    rng = np.random.default_rng(1)
    rows = []
    for p in range(30):
        off = rng.normal(0, 3)
        for _ in range(6):
            rows.append((p, p, p % 3, 1, off + 1 + rng.normal(0, 0.1)))
            rows.append((p, p, p % 3, 0, off + rng.normal(0, 0.1)))
    patient, slide, group, adh, s = (np.array(c) for c in zip(*rows))
    good = summarise(per_patient(s, adh.astype(bool), slide, patient, group), np.random.default_rng(0))
    swapped = summarise(per_patient(s, ~adh.astype(bool), slide, patient, group), np.random.default_rng(0))
    assert good["range95"]["delta"][0] > 0.2, good
    assert swapped["range95"]["delta"][1] < -0.2, swapped
    return {"toy_good_delta": good["delta"], "toy_swapped_delta": swapped["delta"]}


def load():
    with open(ROOT / "data/regions.csv") as f:
        allrows = list(csv.DictReader(f))
    train = [r for r in allrows if r["split"] == "train"]
    val = [r for r in allrows if r["split"] == "val"]
    y = np.array([int(r["label"]) for r in train]); pt = np.array([int(r["patient"]) for r in train])
    fold = np.empty(len(train), object)
    for i, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, pt)):  # the M1 folds
        fold[b] = f"fold{i}"
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    rows = train + val
    assert len(prob) == len(rows)
    group = np.concatenate([fold, np.array(["val"] * len(val), object)])
    label = np.array([int(r["label"]) for r in rows])
    slide = np.array([r["slide"] for r in rows]); patient = np.array([int(r["patient"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    keep = np.isin(label, [UDH, ADH])
    both = {s for s in np.unique(slide[keep]) if {UDH, ADH} <= set(label[keep & (slide == s)])}
    keep &= np.isin(slide, list(both))
    score = prob[:, ADH] / (prob[:, ADH] + prob[:, UDH])
    return score[keep], label[keep] == ADH, slide[keep], patient[keep], group[keep], split[keep]


def main():
    result = {"entry": 61, "tool_check": tool_check()}
    score, is_adh, slide, patient, group, split = load()
    assert len(np.unique(slide)) == 40 and len(np.unique(patient)) == 23, (len(np.unique(slide)), len(np.unique(patient)))
    # the same patient never sits in two classifier groups
    assert all(len(set(group[patient == p])) == 1 for p in np.unique(patient))
    pp = per_patient(score, is_adh, slide, patient, group)
    result["primary"] = summarise(pp, np.random.default_rng(SEED))
    lo, hi = result["primary"]["range95"]["delta"]
    result["reading"] = reading(lo, hi)
    for sp in ("train", "val"):
        ids = set(np.unique(patient[split == sp]).tolist())
        result[sp] = summarise({p: v for p, v in pp.items() if p in ids}, np.random.default_rng(SEED))
    result["per_patient"] = {p: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for p, d in pp.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_patient"}, indent=2))


def gap_mm(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2])); dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy)) * 0.25 / 1000


def distance():
    """Ledger entry 62: near (<= 1 mm) against far (> 3 mm) UDH-ADH pairs on the same slide."""
    with open(ROOT / "data/regions.csv") as f:
        allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    allrows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
    score = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(allrows)}
    patient = {r["slide"]: int(r["patient"]) for r in allrows}
    pos = json.loads((OUT / "positions.json").read_text())
    per = {}
    for slide, regs in pos.items():
        A = [x for x in regs if x["label"] == "ADH"]; U = [x for x in regs if x["label"] == "UDH"]
        for a in A:
            for u in U:
                g = gap_mm(a["box"], u["box"])
                kind = "near" if g <= 1 else "far" if g > 3 else None
                if kind:
                    d = score[a["file"]] - score[u["file"]]
                    per.setdefault(patient[slide], {"near": [], "far": []})[kind].append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
    keys = [p for p, v in per.items() if v["near"] and v["far"]]
    N = np.array([np.mean(per[p]["near"]) for p in keys]); F = np.array([np.mean(per[p]["far"]) for p in keys])
    rng = np.random.default_rng(SEED)
    draws = [(lambda i: (N[i] - F[i]).mean())(rng.integers(0, len(keys), len(keys))) for _ in range(RESAMPLES)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    near_all = [x for p in per.values() for x in p["near"]]; far_all = [x for p in per.values() for x in p["far"]]
    result = {"entry": 62, "patients": len(keys), "near_pairs": len(near_all), "far_pairs": len(far_all),
              "delta": round(float((N - F).mean()), 4), "range95": [round(float(lo), 4), round(float(hi), 4)],
              "mean_N": round(float(N.mean()), 4), "mean_F": round(float(F.mean()), 4),
              "pooled_near": round(float(np.mean(near_all)), 4), "pooled_far": round(float(np.mean(far_all)), 4),
              "reading": ("range below 0: close pairs are harder; supports the disputed-border reading" if hi < 0 else
                          "range above 0: close pairs are easier; the border reading is against the data" if lo > 0 else
                          "range contains 0: no support from distance")}
    (OUT / "distance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def test_look():
    """Ledger entry 63: entries 61 and 62 repeated on the test patients (second look, not clean)."""
    with open(ROOT / "outputs/final/regions_test.csv") as f:
        rows = list(csv.DictReader(f))
    with open(ROOT / "data/regions.csv") as f:
        slide_of = {r["file"]: r["slide"] for r in csv.DictReader(f)}
    rows = [r for r in rows if r["label"] in ("UDH", "ADH")]
    score = np.array([float(r["p_ADH"]) / (float(r["p_ADH"]) + float(r["p_UDH"])) for r in rows])
    is_adh = np.array([r["label"] == "ADH" for r in rows]); slide = np.array([slide_of[r["file"]] for r in rows])
    patient = np.array([int(r["patient"]) for r in rows])
    both = [s for s in np.unique(slide) if is_adh[slide == s].any() and (~is_adh[slide == s]).any()]
    k = np.isin(slide, both)
    pp = per_patient(score[k], is_adh[k], slide[k], patient[k], np.array(["test"] * k.sum(), object))
    result = {"entry": 63, "within_vs_between": summarise(pp, np.random.default_rng(SEED))}
    sc = {r["file"]: score[i] for i, r in enumerate(rows)}; pt = {slide_of[r["file"]]: int(r["patient"]) for r in rows}
    per = {}
    for s, regs in json.loads((OUT / "positions_test.json").read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                g = gap_mm(a["box"], u["box"]); kind = "near" if g <= 1 else "far" if g > 3 else None
                if kind:
                    d = sc[a["file"]] - sc[u["file"]]
                    per.setdefault(pt[s], {"near": [], "far": []})[kind].append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
    keys = [p for p, v in per.items() if v["near"] and v["far"]]
    N = np.array([np.mean(per[p]["near"]) for p in keys]); F = np.array([np.mean(per[p]["far"]) for p in keys])
    rng = np.random.default_rng(SEED)
    draws = [(lambda i: (N[i] - F[i]).mean())(rng.integers(0, len(keys), len(keys))) for _ in range(RESAMPLES)]
    na = [x for v in per.values() for x in v["near"]]; fa = [x for v in per.values() for x in v["far"]]
    result["distance"] = {"patients": len(keys), "delta": round(float((N - F).mean()), 4),
                          "range95": [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])],
                          "pooled_near": round(float(np.mean(na)), 4), "pooled_far": round(float(np.mean(fa)), 4),
                          "near_pairs": len(na), "far_pairs": len(fa)}
    (OUT / "test_look.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


BIN_EDGES = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]


def pair_table(split):
    """Every within-slide ADH-UDH pair with a position: (patient, gap_mm, right)."""
    if split == "test":
        with open(ROOT / "outputs/final/regions_test.csv") as f:
            rows = list(csv.DictReader(f))
        sc = {r["file"]: float(r["p_ADH"]) / (float(r["p_ADH"]) + float(r["p_UDH"])) for r in rows if r["label"] in ("UDH", "ADH")}
        pos = json.loads((OUT / "positions_test.json").read_text())
    else:
        with open(ROOT / "data/regions.csv") as f:
            allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
        allrows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
        prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
        sc = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(allrows)}
        pos = json.loads((OUT / "positions.json").read_text())
    with open(ROOT / "data/regions.csv") as f:
        pt = {r["slide"]: int(r["patient"]) for r in csv.DictReader(f)}
    out = []
    for s, regs in pos.items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                d = sc[a["file"]] - sc[u["file"]]
                out.append((pt[s], gap_mm(a["box"], u["box"]), 1.0 if d > 0 else 0.5 if d == 0 else 0.0))
    return np.array(out)


def slope(t, equal_patients=False):
    x, y = np.log2(t[:, 1] + 0.125), t[:, 2]
    w = np.ones(len(t))
    if equal_patients:
        _, inv, cnt = np.unique(t[:, 0], return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def bins_report(t, rng):
    pats = np.unique(t[:, 0]); idx = {p: np.where(t[:, 0] == p)[0] for p in pats}
    def stats(tt):
        b = np.digitize(tt[:, 1], BIN_EDGES[1:-1], right=True)
        acc = [float(tt[b == k, 2].mean()) if (b == k).any() else np.nan for k in range(len(BIN_EDGES) - 1)]
        return acc, slope(tt), slope(tt, True)
    acc, sl, sle = stats(t)
    draws = [stats(t[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]) for _ in range(RESAMPLES)]
    r = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    b = np.digitize(t[:, 1], BIN_EDGES[1:-1], right=True)
    labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4-8", "8+"]
    return {"pairs": len(t), "patients": len(pats),
            "bins": {labels[k]: {"pairs": int((b == k).sum()), "patients": int(len(np.unique(t[b == k, 0]))), "share_right": round(acc[k], 4),
                                 "range95": r([d[0][k] for d in draws])} for k in range(len(labels))},
            "slope": round(sl, 4), "slope_range95": r([d[1] for d in draws]),
            "slope_equal_patients": round(sle, 4), "slope_equal_patients_range95": r([d[2] for d in draws])}


def bins():
    """Ledger entry 64: accuracy against distance, every pair, patient resampling."""
    dev = pair_table("dev")
    result = {"entry": 64, "development": bins_report(dev, np.random.default_rng(SEED))}
    lo, hi = result["development"]["slope_range95"]
    result["reading"] = ("slope range above 0: accuracy rises steadily with distance" if lo > 0 else
                         "slope range below 0: the lead is against the data" if hi < 0 else "slope range contains 0: no steady trend")
    drop = dev[~np.isin(dev[:, 0], [49, 150])]
    result["without_49_150"] = {"slope": round(slope(drop), 4), "slope_range95": bins_report(drop, np.random.default_rng(SEED))["slope_range95"]}
    result["test_second_look"] = bins_report(pair_table("test"), np.random.default_rng(SEED))
    (OUT / "bins.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def all_pairs(split):
    """Every within-slide UDH/ADH pair with positions: (patient, gap_mm, |s_a - s_b|, different_label)."""
    if split == "test":
        with open(ROOT / "outputs/final/regions_test.csv") as f:
            rows = list(csv.DictReader(f))
        sc = {r["file"]: float(r["p_ADH"]) / (float(r["p_ADH"]) + float(r["p_UDH"])) for r in rows if r["label"] in ("UDH", "ADH")}
        pos = json.loads((OUT / "positions_test.json").read_text())
    else:
        with open(ROOT / "data/regions.csv") as f:
            allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
        allrows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
        prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
        sc = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(allrows)}
        pos = json.loads((OUT / "positions.json").read_text())
    with open(ROOT / "data/regions.csv") as f:
        pt = {r["slide"]: int(r["patient"]) for r in csv.DictReader(f)}
    out = []
    for s, regs in pos.items():
        regs = [x for x in regs if x["label"] in ("UDH", "ADH")]
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                out.append((pt[s], gap_mm(a["box"], b["box"]), abs(sc[a["file"]] - sc[b["file"]]), float(a["label"] != b["label"])))
    return np.array(out)


def nearness_fit(t):
    x = np.log2(t[:, 1] + 0.125); d = t[:, 3]
    X = np.column_stack([np.ones(len(t)), x, d, x * d])
    beta = np.linalg.lstsq(X, t[:, 2], rcond=None)[0]
    return {"same_slope": float(beta[1]), "diff_slope": float(beta[1] + beta[3]), "product": float(beta[3])}


def nearness_report(t, rng):
    pats = np.unique(t[:, 0]); idx = {p: np.where(t[:, 0] == p)[0] for p in pats}
    point = nearness_fit(t)
    draws = [nearness_fit(t[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]) for _ in range(RESAMPLES)]
    return {"pairs_different": int(t[:, 3].sum()), "pairs_same": int((1 - t[:, 3]).sum()), "patients": len(pats),
            **{k: round(v, 4) for k, v in point.items()},
            **{k + "_range95": [round(float(np.percentile([d[k] for d in draws], q)), 4) for q in (2.5, 97.5)] for k in point}}


def nearness():
    """Ledger entry 65: does UDH-ADH separation grow with distance faster than same-label spread?"""
    dev = all_pairs("dev")
    assert int(dev[:, 3].sum()) == 2022 and int((1 - dev[:, 3]).sum()) == 2555
    result = {"entry": 65, "development": nearness_report(dev, np.random.default_rng(SEED))}
    lo, hi = result["development"]["product_range95"]
    result["reading"] = ("range above 0: separation grows beyond nearness; the curve is about the UDH-ADH border" if lo > 0 else
                         "range below 0: the curve is nearness; the border reading weakens" if hi < 0 else
                         "range contains 0: entry 64 may be nearness alone")
    result["test_second_look"] = nearness_report(all_pairs("test"), np.random.default_rng(SEED))
    (OUT / "nearness.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def size_pairs(split):
    """ADH-UDH pairs: (patient, gap_mm, model picks the larger box, model picks ADH)."""
    if split == "test":
        with open(ROOT / "outputs/final/regions_test.csv") as f:
            rows = list(csv.DictReader(f))
        sc = {r["file"]: float(r["p_ADH"]) / (float(r["p_ADH"]) + float(r["p_UDH"])) for r in rows if r["label"] in ("UDH", "ADH")}
        pos = json.loads((OUT / "positions_test.json").read_text())
    else:
        with open(ROOT / "data/regions.csv") as f:
            allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
        allrows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
        prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
        sc = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(allrows)}
        pos = json.loads((OUT / "positions.json").read_text())
    with open(ROOT / "data/regions.csv") as f:
        pt = {r["slide"]: int(r["patient"]) for r in csv.DictReader(f)}
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    out = []
    for s, regs in pos.items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                d = sc[a["file"]] - sc[u["file"]]
                pick_adh = 1.0 if d > 0 else 0.5 if d == 0 else 0.0
                pick_big = pick_adh if area(a["box"]) > area(u["box"]) else 1 - pick_adh
                out.append((pt[s], gap_mm(a["box"], u["box"]), pick_big, pick_adh))
    return np.array(out)


def size_report(t, rng):
    pats = np.unique(t[:, 0]); idx = {p: np.where(t[:, 0] == p)[0] for p in pats}
    f = lambda tt: (tt[:, 2].mean(), tt[:, 2].mean() - tt[:, 3].mean())
    L, LA = f(t)
    draws = [f(t[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]) for _ in range(RESAMPLES)]
    r = lambda k: [round(float(np.percentile([d[k] for d in draws], q)), 4) for q in (2.5, 97.5)]
    return {"pairs": len(t), "patients": len(pats), "L": round(float(L), 4), "L_range95": r(0), "L_minus_A": round(float(LA), 4), "L_minus_A_range95": r(1)}


def size():
    """Ledger entry 66: at close range, does the engine pick the bigger region?"""
    dev = size_pairs("dev")
    near, far = dev[dev[:, 1] <= 1], dev[dev[:, 1] > 3]
    assert len(near) == 283
    result = {"entry": 66, "close": size_report(near, np.random.default_rng(SEED)), "far": size_report(far, np.random.default_rng(SEED))}
    lo, hi = result["close"]["L_range95"]
    result["reading"] = ("range above 0.5: leans to the bigger region at close range" if lo > 0.5 else
                         "range below 0.5: leans to the smaller region" if hi < 0.5 else "range contains 0.5: size is not the cause")
    test = size_pairs("test")
    result["test_close_second_look"] = size_report(test[test[:, 1] <= 1], np.random.default_rng(SEED))
    (OUT / "size.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def relative():
    """Ledger entry 67: a pair head trained on far pairs, scored on close pairs of held-out patients."""
    sys.path.insert(0, str(ROOT / "tools"))
    from short_lists import engine_features
    from train_classifier import classifier
    from sklearn.model_selection import GroupKFold
    with open(ROOT / "data/regions.csv") as f:
        allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    tr = [r for r in allrows if r["split"] == "train"]; va = [r for r in allrows if r["split"] == "val"]
    rows = tr + va
    feats = np.vstack([engine_features("train"), engine_features("val")])
    assert len(feats) == len(rows)
    fi = {r["file"]: i for i, r in enumerate(rows)}
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    eng = prob[:, ADH] / (prob[:, ADH] + prob[:, UDH])
    pt = {r["slide"]: int(r["patient"]) for r in rows}
    pairs = []  # (patient, gap, adh index, udh index)
    for s, regs in json.loads((OUT / "positions.json").read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                pairs.append((pt[s], gap_mm(a["box"], u["box"]), fi[a["file"]], fi[u["file"]]))
    P = np.array(pairs)
    pat, gap, ia, iu = P[:, 0].astype(int), P[:, 1], P[:, 2].astype(int), P[:, 3].astype(int)
    pats = np.unique(pat)
    rng = np.random.default_rng(SEED); order = rng.permutation(pats)
    group = {p: k % 5 for k, p in enumerate(order)}
    g = np.array([group[p] for p in pat])
    right = np.full(len(P), np.nan)
    for k in range(5):
        trn = (g != k) & (gap > 3)
        X = np.vstack([feats[ia[trn]] - feats[iu[trn]], feats[iu[trn]] - feats[ia[trn]]])
        y = np.concatenate([np.ones(trn.sum()), np.zeros(trn.sum())])
        head = classifier(0.001).fit(X, y)
        te = g == k
        o1 = head.decision_function(feats[ia[te]] - feats[iu[te]]); o2 = head.decision_function(feats[iu[te]] - feats[ia[te]])
        right[te] = np.where(o1 > o2, 1.0, np.where(o1 == o2, 0.5, 0.0))
    d = eng[ia] - eng[iu]; eright = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
    def rep(mask):
        pp = np.unique(pat[mask]); idx = {p: np.where(mask & (pat == p))[0] for p in pp}
        r2 = np.random.default_rng(SEED)
        draws = [(lambda ii: (right[ii].mean(), eright[ii].mean()))(np.concatenate([idx[p] for p in r2.choice(pp, len(pp))])) for _ in range(RESAMPLES)]
        q = lambda j: [round(float(np.percentile([x[j] for x in draws], v)), 4) for v in (2.5, 97.5)]
        return {"pairs": int(mask.sum()), "patients": len(pp), "head": round(float(right[mask].mean()), 4), "head_range95": q(0),
                "engine": round(float(eright[mask].mean()), 4), "engine_range95": q(1)}
    result = {"entry": 67, "control_far": rep(gap > 3), "primary_close_0_5": rep(gap <= 0.5), "close_1": rep(gap <= 1)}
    c = result["control_far"]["head"]; m = result["primary_close_0_5"]
    if c < 0.75:
        result["reading"] = "control failed: held-out far pairs under 0.75; nothing else is read"
    elif m["head"] >= 0.65 and m["head_range95"][0] > 0.5:
        result["reading"] = "pass: the absolute head was the problem; a relative head fixes close pairs"
    elif m["head"] < 0.55:
        result["reading"] = "fail: direction at touching range is unstable; that is the finding"
    else:
        result["reading"] = "between 0.55 and 0.65 (or range not above 0.5): not decided"
    (OUT / "relative.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def close_head():
    """Ledger entry 69: a pair head trained on close pairs, scored on held-out close pairs."""
    sys.path.insert(0, str(ROOT / "tools"))
    from short_lists import engine_features
    from train_classifier import classifier
    with open(ROOT / "data/regions.csv") as f:
        allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    rows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
    feats = np.vstack([engine_features("train"), engine_features("val")])
    fi = {r["file"]: i for i, r in enumerate(rows)}
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    eng = prob[:, ADH] / (prob[:, ADH] + prob[:, UDH])
    pt = {r["slide"]: int(r["patient"]) for r in rows}
    pairs = []
    for s, regs in json.loads((OUT / "positions.json").read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                pairs.append((pt[s], gap_mm(a["box"], u["box"]), fi[a["file"]], fi[u["file"]]))
    P = np.array(pairs); pat, gap, ia, iu = P[:, 0].astype(int), P[:, 1], P[:, 2].astype(int), P[:, 3].astype(int)
    order = np.random.default_rng(SEED).permutation(np.unique(pat)); group = {p: k % 5 for k, p in enumerate(order)}
    g = np.array([group[p] for p in pat])
    def run(sel, with_score):
        def inp(i, j):
            x = feats[i] - feats[j]
            return np.column_stack([x, eng[i] - eng[j]]) if with_score else x
        right = np.full(len(P), np.nan)
        for k in range(5):
            trn = sel & (g != k); te = sel & (g == k)
            if not trn.any() or not te.any():
                continue
            X = np.vstack([inp(ia[trn], iu[trn]), inp(iu[trn], ia[trn])]); y = np.r_[np.ones(trn.sum()), np.zeros(trn.sum())]
            h = classifier(0.001).fit(X, y)
            o1, o2 = h.decision_function(inp(ia[te], iu[te])), h.decision_function(inp(iu[te], ia[te]))
            right[te] = np.where(o1 > o2, 1.0, np.where(o1 == o2, 0.5, 0.0))
        return right
    d = eng[ia] - eng[iu]; eright = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
    def rep(right, mask):
        mask = mask & ~np.isnan(right)
        pp = np.unique(pat[mask]); idx = {p: np.where(mask & (pat == p))[0] for p in pp}
        per = np.array([right[idx[p]].mean() - eright[idx[p]].mean() for p in pp])
        r2 = np.random.default_rng(SEED); draws = []
        for _ in range(RESAMPLES):
            ch = r2.choice(len(pp), len(pp)); ii = np.concatenate([idx[pp[c]] for c in ch])
            draws.append((right[ii].mean(), eright[ii].mean(), per[ch].mean()))
        q = lambda j: [round(float(np.percentile([x[j] for x in draws], v)), 4) for v in (2.5, 97.5)]
        return {"pairs": int(mask.sum()), "patients": len(pp), "head": round(float(right[mask].mean()), 4), "head_range95": q(0),
                "engine": round(float(eright[mask].mean()), 4), "engine_range95": q(1),
                "paired_head_minus_engine": round(float(per.mean()), 4), "paired_range95": q(2)}
    if "allpairs" in sys.argv:  # ledger entry 70: train on every pair, read close at 1 mm
        allp, close1, close05, far = np.ones(len(P), bool), gap <= 1, gap <= 0.5, gap > 3
        r = run(allp, False)
        result = {"entry": 70, "control_far": rep(r, far), "primary_close_1": rep(r, close1), "close_0_5": rep(r, close05)}
        c = result["control_far"]["head"]; m = result["primary_close_1"]; lo, hi = m["head_range95"]
        result["reading"] = ("control failed (far < 0.70); nothing else is read" if c < 0.70 else
                             "above chance and at least the engine" if lo > 0.5 and m["head"] >= m["engine"] else
                             "range contains 0.5: not shown above chance at this sample" if lo <= 0.5 else
                             "above chance but below the engine")
        (OUT / "allpairs.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    close, far = gap <= 0.5, gap > 3
    result = {"entry": 69, "control_far": rep(run(far, False), far), "primary_close": rep(run(close, False), close),
              "arm_with_engine_score": rep(run(close, True), close)}
    c = result["control_far"]["head"]; lo, hi = result["primary_close"]["head_range95"]
    result["reading"] = ("control failed (far < 0.70); nothing else is read" if c < 0.70 else
                         "learnable: close-pair direction is in the features; the heads were the limit" if lo > 0.65 else
                         "range contains 0.5: the features do not carry the direction at this scale" if lo <= 0.5 <= hi else "weak, not decided")
    (OUT / "close_head.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def shapes_test():
    """Ledger entry 71: the far-close drop for polygon-polygon against rectangle-rectangle pairs."""
    kind = json.loads((OUT / "shape_kind.json").read_text())
    with open(ROOT / "data/regions.csv") as f:
        allrows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    allrows = [r for r in allrows if r["split"] == "train"] + [r for r in allrows if r["split"] == "val"]
    prob = np.vstack([np.load(ROOT / "outputs/m2/oof_prob_engine.npy"), np.load(ROOT / "outputs/m2/val_prob_engine.npy")])
    sc = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(allrows)}
    pt = {r["slide"]: int(r["patient"]) for r in allrows}
    rows = []
    for s, regs in json.loads((OUT / "positions.json").read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                g = gap_mm(a["box"], u["box"]); band = 0 if g <= 1 else 1 if g > 3 else -1
                if band < 0:
                    continue
                k = "-".join(sorted([kind[a["file"]], kind[u["file"]]]))
                d = sc[a["file"]] - sc[u["file"]]
                rows.append((pt[s], band, {"polygon-polygon": 0, "polygon-rectangle": 1, "rectangle-rectangle": 2}[k], 1.0 if d > 0 else 0.5 if d == 0 else 0.0))
    t = np.array(rows); pats = np.unique(t[:, 0]); idx = {p: np.where(t[:, 0] == p)[0] for p in pats}
    def stats(tt):
        acc = lambda k, b: tt[(tt[:, 2] == k) & (tt[:, 1] == b), 3].mean() if ((tt[:, 2] == k) & (tt[:, 1] == b)).any() else np.nan
        dpp = acc(0, 1) - acc(0, 0); drr = acc(2, 1) - acc(2, 0)
        return {"D_pp": dpp, "D_rr": drr, "D_rr_minus_D_pp": drr - dpp, **{f"{n}_{bn}": acc(k, b) for k, n in enumerate(["pp", "mixed", "rr"]) for b, bn in ((0, "close"), (1, "far"))}}
    point = stats(t); rng = np.random.default_rng(SEED)
    draws = [stats(t[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]) for _ in range(RESAMPLES)]
    res = {k: {"value": round(float(v), 4), "range95": [round(float(np.nanpercentile([d[k] for d in draws], q)), 4) for q in (2.5, 97.5)]} for k, v in point.items()}
    pc = lambda k, b: int(len(np.unique(t[(t[:, 2] == k) & (t[:, 1] == b), 0])))
    res["patients"] = {f"{n}_{bn}": pc(k, b) for k, n in enumerate(["pp", "mixed", "rr"]) for b, bn in ((0, "close"), (1, "far"))}
    lo, hi = res["D_pp"]["range95"]
    res["reading"] = ("range above 0: the drop holds with traced outlines; box overlap does not explain it" if lo > 0 else
                      "range below 0: close traced pairs are easier" if hi < 0 else "range contains 0: not shown with traced outlines")
    (OUT / "shapes.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    shapes_test() if "shapes" in sys.argv else close_head() if ("close" in sys.argv or "allpairs" in sys.argv) else relative() if "relative" in sys.argv else size() if "size" in sys.argv else nearness() if "nearness" in sys.argv else bins() if "bins" in sys.argv else test_look() if "test" in sys.argv else distance() if "distance" in sys.argv else main()
