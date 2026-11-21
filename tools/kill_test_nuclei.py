#!/usr/bin/env python3
"""Kill test for the "Two ways of seeing" demo idea. Validation patients only; the test set is not read.

Question: do tiles that look alike to UNI also look alike by their nuclei measurements?

Usage (from the project folder, with the .venv-nuclei environment):
  python -I tools/kill_test_nuclei.py measure   # StarDist on the 40x pixels of each tile -> outputs/kill_test/nuclei.csv
  python -I tools/kill_test_nuclei.py analyse   # neighbour overlap + verdict -> outputs/kill_test/result.json
  python -I tools/kill_test_nuclei.py models    # UNI vs. H-optimus-0 variant -> outputs/kill_test/result_models.json

Kill rule (fixed before the first result, 2026-11-17):
  noise    : overlap(nuclei, UNI) < 2 x chance, or nuclei neighbours share a tissue type no more than chance
  no story : overlap(nuclei, UNI) >= overlap(UNI, H-optimus-0)
  pass     : anything in between

Neighbours are the 10 nearest tiles from OTHER patients, so a patient's own look does not count as agreement.
"""
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/bracs/BRACS_RoI/latest_version"
FEATS = ROOT / "data/features"
OUT = ROOT / "outputs/kill_test"
PER_REGION = 16   # tiles sampled per region
MIN_TISSUE = 0.5  # share of non-white pixels a tile needs
MIN_NUCLEI = 5    # nuclei a tile needs for its statistics to mean anything
K = 10
STATS = ["log_count", "area_share", "log_area_mean", "area_cv", "ecc_mean", "ecc_sd", "solidity_mean",
         "stain_mean", "stain_sd", "log_gap_mean", "gap_cv"]


def val_rows():
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "val"]


def tile_stats(labels, rgb):
    """Nuclei statistics for one 448-px tile at 40x, or None if it holds too few nuclei."""
    from scipy.spatial import cKDTree
    from skimage.color import rgb2hed
    from skimage.measure import regionprops
    props = regionprops(labels, intensity_image=rgb2hed(rgb)[..., 0])
    if len(props) < MIN_NUCLEI:
        return None
    area = np.array([p.area for p in props], float)
    ecc = np.array([p.eccentricity for p in props])
    stain = np.array([p.intensity_mean for p in props])
    gap = cKDTree([p.centroid for p in props]).query([p.centroid for p in props], k=2)[0][:, 1]
    return [np.log(len(props)), area.sum() / labels.size, np.log(area.mean()), area.std() / area.mean(),
            ecc.mean(), ecc.std(), np.mean([p.solidity for p in props]),
            stain.mean(), stain.std(), np.log(gap.mean()), gap.std() / gap.mean()]


def measure():
    from csbdeep.utils import normalize
    from PIL import Image
    from stardist.models import StarDist2D
    Image.MAX_IMAGE_PIXELS = 200_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "nuclei.csv"
    done = set()
    if dest.exists():
        with open(dest) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    model = StarDist2D.from_pretrained("2D_versatile_he")
    rng = np.random.default_rng(0)
    rows = val_rows()
    shown = set()
    with open(dest, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["file", "tile", "label_name", "patient", *STATS])
        for n, r in enumerate(rows):
            z = np.load(FEATS / "uni" / (r["file"][:-4] + ".npz"))
            pick = np.flatnonzero(z["tissue"] >= MIN_TISSUE)
            pick = np.sort(rng.choice(pick, min(PER_REGION, len(pick)), replace=False))  # drawn even when skipped: same sample on resume
            if r["file"] in done:
                continue
            img = Image.open(IMAGES / r["file"]).convert("RGB")
            kept = 0
            for t in pick:
                x, y = (int(v) * 2 for v in z["coords"][t])
                x, y = min(x, img.width - 448), min(y, img.height - 448)
                if x < 0 or y < 0:  # region smaller than one tile: skipped, counted in analyse
                    continue
                rgb = np.asarray(img.crop((x, y, x + 448, y + 448)))
                labels, _ = model.predict_instances(normalize(rgb, 1, 99.8, axis=(0, 1, 2)))
                s = tile_stats(labels, rgb)
                if s is None:
                    continue
                w.writerow([r["file"], t, r["label_name"], r["patient"], *(f"{v:.5f}" for v in s)])
                kept += 1
                if r["label_name"] not in shown:  # one outline picture per tissue type, for a check by eye
                    from skimage.segmentation import mark_boundaries
                    shown.add(r["label_name"])
                    Image.fromarray((mark_boundaries(rgb, labels, color=(0, 1, 0)) * 255).astype(np.uint8)).save(
                        OUT / f"outline_{r['label_name']}.png")
            f.flush()
            print(f"{n + 1}/{len(rows)} {r['file']} tiles kept {kept}/{len(pick)}", flush=True)


def neighbours(x, patient, k=K):
    """Index of the k nearest tiles from other patients, for every tile (Euclidean distance)."""
    sq = (x * x).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * x @ x.T
    d[patient[:, None] == patient[None, :]] = np.inf
    return np.argpartition(d, k, axis=1)[:, :k]


def overlap(a, b):
    """Per tile: the share of its neighbours that two spaces have in common."""
    return np.array([len(set(p) & set(q)) / len(p) for p, q in zip(a, b)])


def by_patient_range(values, patient, rng):
    """95% range of the mean, resampling whole patients."""
    ids = np.unique(patient)
    sums = np.array([values[patient == p].sum() for p in ids])
    counts = np.array([(patient == p).sum() for p in ids])
    draws = rng.integers(0, len(ids), (2000, len(ids)))
    return [round(float(v), 4) for v in np.percentile(sums[draws].sum(1) / counts[draws].sum(1), [2.5, 97.5])]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)  # unit length: Euclidean order = cosine order


def model_spaces(rows):
    """Each model's features for the measured tiles."""
    spaces, cache = {}, {}
    for m in ["uni", "hoptimus0", "resnet50"]:
        x = []
        for r in rows:
            key = (m, r["file"])
            if key not in cache:
                cache[key] = np.load(FEATS / m / (r["file"][:-4] + ".npz"))["feats"].astype(np.float32)
            x.append(cache[key][int(r["tile"])])
        spaces[m] = unit(np.array(x))
    return spaces


def models():
    """Second kill test: is the UNI vs. H-optimus-0 disagreement real, and does it sit on the ambiguous tiles?

    Rule (fixed before the first result, 2026-11-18):
      real difference : overlap(UNI, H-optimus-0) clearly below the ceiling (one half of UNI's numbers vs. the other half)
      a story         : per tile, agreement between the two models rises with the share of same-type neighbours,
                        and the by-patient 95% range of that rank correlation excludes zero
      otherwise kill.
    """
    from scipy.stats import spearmanr
    with open(OUT / "nuclei.csv") as f:
        rows = list(csv.DictReader(f))
    patient = np.array([int(r["patient"]) for r in rows])
    label = np.array([r["label_name"] for r in rows])
    rng = np.random.default_rng(0)
    sp = model_spaces(rows)
    half = rng.permutation(sp["uni"].shape[1])
    a, b = half[:len(half) // 2], half[len(half) // 2:]
    by_k = {}
    for k in [10, 50, 200]:
        nu, nh = neighbours(sp["uni"], patient, k), neighbours(sp["hoptimus0"], patient, k)
        by_k[k] = {"uni vs hoptimus0": round(float(overlap(nu, nh).mean()), 4),
                   "ceiling (uni half vs half)": round(float(overlap(
                       neighbours(unit(sp["uni"][:, a]), patient, k), neighbours(unit(sp["uni"][:, b]), patient, k)).mean()), 4),
                   "uni vs resnet50": round(float(overlap(nu, neighbours(sp["resnet50"], patient, k)).mean()), 4),
                   "chance": round(k / len(rows), 4)}
    nu, nh = neighbours(sp["uni"], patient, 50), neighbours(sp["hoptimus0"], patient, 50)
    agree = overlap(nu, nh)
    same = ((label[nu] == label[:, None]).mean(1) + (label[nh] == label[:, None]).mean(1)) / 2
    rho = float(spearmanr(agree, same)[0])
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    draws = []
    for _ in range(1000):
        i = np.concatenate([where[p] for p in rng.choice(ids, len(ids))])
        draws.append(spearmanr(agree[i], same[i])[0])
    lo, hi = (round(float(v), 4) for v in np.percentile(draws, [2.5, 97.5]))
    types = sorted(set(label))
    result = {"tiles": len(rows), "patients": int(len(ids)), "by_k": by_k,
              "agreement_by_type_k50": {t: round(float(agree[label == t].mean()), 4) for t in types},
              "same_type_share_by_type_k50": {t: round(float(same[label == t].mean()), 4) for t in types},
              "rank_correlation_agreement_vs_same_type": round(rho, 4), "range95_by_patient": [lo, hi],
              "real_difference": by_k[50]["uni vs hoptimus0"] < 0.8 * by_k[50]["ceiling (uni half vs half)"],
              "story": lo > 0}
    result["verdict"] = "PASS" if result["real_difference"] and result["story"] else "KILL"
    (OUT / "result_models.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def analyse():
    with open(OUT / "nuclei.csv") as f:
        rows = list(csv.DictReader(f))
    patient = np.array([int(r["patient"]) for r in rows])
    label = np.array([r["label_name"] for r in rows])
    rng = np.random.default_rng(0)

    spaces = {"nuclei": np.array([[float(r[s]) for s in STATS] for r in rows])}
    spaces["nuclei"] = (spaces["nuclei"] - spaces["nuclei"].mean(0)) / spaces["nuclei"].std(0)
    spaces.update(model_spaces(rows))
    nn = {k: neighbours(v, patient) for k, v in spaces.items()}
    assert overlap(nn["uni"], nn["uni"]).min() == 1.0  # self-check: a space agrees fully with itself

    def chance(fn):  # the same measure after shuffling which tile owns which nuclei statistics
        return float(np.mean([fn(neighbours(spaces["nuclei"][rng.permutation(len(rows))], patient)) for _ in range(5)]))

    pairs = {}
    for a, b in [("nuclei", "uni"), ("nuclei", "hoptimus0"), ("nuclei", "resnet50"), ("uni", "hoptimus0"), ("uni", "resnet50")]:
        o = overlap(nn[a], nn[b])
        pairs[f"{a} vs {b}"] = {"overlap": round(float(o.mean()), 4), "range95": by_patient_range(o, patient, rng)}
    same_type = {k: round(float((label[v] == label[:, None]).mean()), 4) for k, v in nn.items()}
    chance_overlap = chance(lambda s: overlap(s, nn["uni"]).mean())
    chance_type = chance(lambda s: (label[s] == label[:, None]).mean())
    o = overlap(nn["nuclei"], nn["uni"])
    per_type = {t: round(float(o[label == t].mean()), 4) for t in sorted(set(label))}

    nu, ceiling = pairs["nuclei vs uni"]["overlap"], pairs["uni vs hoptimus0"]["overlap"]
    if nu < 2 * chance_overlap or same_type["nuclei"] <= chance_type:
        verdict = "KILL: noise"
    elif nu >= ceiling:
        verdict = "KILL: no story"
    else:
        verdict = "PASS"
    result = {"tiles": len(rows), "regions": len({r["file"] for r in rows}), "regions_in_val": len(val_rows()),
              "patients": int(len(np.unique(patient))), "k": K, "pairs": pairs,
              "chance_overlap": round(chance_overlap, 4), "same_type_share": same_type,
              "chance_same_type_share": round(chance_type, 4), "nuclei_vs_uni_by_type": per_type,
              "tiles_by_type": {t: int((label == t).sum()) for t in sorted(set(label))}, "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"measure": measure, "analyse": analyse, "models": models}[sys.argv[1]]()
