#!/usr/bin/env python3
"""Build the case viewer: real regions the system never trained on, with its decision and the three kinds of evidence.

Usage (from the project folder):
  python -I tools/build_viewer.py   # -> outputs/viewer/index.html, data.js, img/

Cases are regions of the 30 test patients (used once in ledger entry 9; here they are only displayed).
They are drawn at random with a fixed seed, 8 per type, so failures are shown as often as they happen.
Images stay on this machine. Before anything is published: credit BRACS (CC BY-NC 4.0) and check its download terms.
"""
import csv
import json
import pathlib
import shutil
import sys

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import evidence_tiles as et
import evidence_why as why
import patient_promise as pp
import short_lists as sl
from train_classifier import rows

ROOT = sl.ROOT
OUT = ROOT / "outputs/viewer"
IMAGES = ROOT / "data/bracs/BRACS_RoI/latest_version"
PER_TYPE, MAX_TILES, MAX_SIDE, TILE = 8, 150, 1200, 224
NAMES = ["Normal", "Pathological benign", "Usual ductal hyperplasia", "Flat epithelial atypia", "Atypical ductal hyperplasia",
         "Ductal carcinoma in situ", "Invasive carcinoma"]
Image.MAX_IMAGE_PIXELS = 200_000_000


def main():
    with open(ROOT / "data/regions.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    with open(ROOT / "outputs/final/regions_test.csv") as f:
        saved = list(csv.DictReader(f))
    assert [r["file"] for r in test] == [r["file"] for r in saved]
    prob = np.array([[float(r[f"p_{t}"]) for t in sl.TYPES] for r in saved])
    y = np.array([int(r["label"]) for r in test])
    patient = np.array([int(r["patient"]) for r in test])
    act, lists = pp.apply_system(prob, patient)
    top = prob.max(1)
    context = (top + pp.patient_mean(top, patient)) / 2
    system = json.loads(next(p for p in (pp.OUT / "system_v5.json", pp.OUT / "system_v4.json", pp.OUT / "system_v3.json", pp.OUT / "system_v2.json") if p.exists()).read_text())

    final = joblib.load(ROOT / "outputs/m2/engine.joblib")
    view = np.load(ROOT / "outputs/m1/thunder/uni_test.npy")
    train = rows("train")
    y_t = np.array([int(r["label"]) for r in train])
    lib = final.decision_function(sl.engine_features("train"))

    pk = ROOT / "outputs/evidence/patterns_keep.json"  # ledger entry 22: the model whose patterns passed
    patterns = json.loads(pk.read_text()) if pk.exists() else None
    usable = [k for k, v in patterns["patterns"].items() if v["usable"]] if patterns and patterns["in_viewer"] else []

    tiles_json = json.loads((ROOT / "outputs/evidence/tiles.json").read_text()) if (ROOT / "outputs/evidence/tiles.json").exists() else None
    tile_lib = et.library() if tiles_json and tiles_json["shown"] else None
    crops = {}

    def crop(region, xy):
        """A 224-px tile of a training region at 20x, saved once."""
        key = (int(region), int(xy[0]), int(xy[1]))
        if key not in crops:
            im = Image.open(IMAGES / train[key[0]]["file"]).convert("RGB").reduce(2)
            name = f"img/t{key[0]}_{key[1]}_{key[2]}.jpg"
            im.crop((key[1], key[2], key[1] + TILE, key[2] + TILE)).resize((150, 150), Image.LANCZOS).save(OUT / name, quality=82)
            crops[key] = {"img": name, "label": int(y_t[key[0]]), "id": pathlib.Path(train[key[0]]["file"]).stem.replace("BRACS_", "")}
        return crops[key]

    rng = np.random.default_rng(0)
    n_tiles = np.array([len(why.tissue_tiles(r["file"])[0]) for r in test])
    chosen = []
    for k in range(7):
        pool = np.flatnonzero((y == k) & (n_tiles >= 2) & (n_tiles <= MAX_TILES))
        chosen += sorted(rng.choice(pool, min(PER_TYPE, len(pool)), replace=False).tolist())

    if usable:
        import torch

        import evidence_patterns as ep
        import evidence_patterns_keep as kp
        text = np.load(ROOT / "outputs/evidence/keep/text.npy")[[list(ep.PATTERNS).index(k) for k in usable]]
        ref = np.load(ROOT / "outputs/evidence/keep/region_mean.npy") @ text.T
        keep_model, _ = kp.load_keep()
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        emb = kp.embed_regions([IMAGES / test[i]["file"] for i in chosen], keep_model.to(dev), dev)
        q = np.zeros((len(test), len(usable)))
        q[chosen] = emb @ text.T
        del keep_model

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)
    thumbs = {}

    def thumb(i):
        if i not in thumbs:
            im = Image.open(IMAGES / train[i]["file"]).convert("RGB").reduce(2)
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2)).resize((220, 220), Image.LANCZOS)
            name = f"img/n{i}.jpg"
            im.save(OUT / name, quality=80)
            thumbs[i] = {"img": name, "id": pathlib.Path(train[i]["file"]).stem.replace("BRACS_", "")}
        return thumbs[i]

    cases = []
    for i in chosen:
        r = test[i]
        feats, coords = why.tissue_tiles(r["file"])
        tiles, whole, base = why.split_scores(final, feats, view[i])
        x = np.hstack([feats.mean(0), view[i]])[None]
        assert np.abs(tiles.sum(0) + whole + base - final.decision_function(x)[0]).max() < 1e-3  # the receipt adds up
        im = Image.open(IMAGES / r["file"]).convert("RGB").reduce(2)
        f = min(1.0, MAX_SIDE / max(im.size))
        small = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS) if f < 1 else im
        stem = pathlib.Path(r["file"]).stem
        small.save(OUT / "img" / f"{stem}.jpg", quality=84)
        d = np.sqrt(((lib - final.decision_function(x)) ** 2).sum(1))
        show = np.flatnonzero(lists[i]) if not act[i] and lists[i].any() else np.argsort(-prob[i])[:2]
        look = {int(k): [thumb(int(j)) for j in np.flatnonzero(y_t == k)[np.argsort(d[y_t == k])[:3]]] for k in show[:4]}
        parts = stem.split("_")
        case = {"id": stem, "slide": "_".join(parts[:2]), "region": parts[-1], "label": int(y[i]), "decision": "act" if act[i] else "ask",
                "top": int(prob[i].argmax()), "prob": [round(float(v), 4) for v in prob[i]], "list": [int(k) for k in np.flatnonzero(lists[i])],
                "context": round(float(context[i]), 4), "img": f"img/{stem}.jpg", "w": small.width, "h": small.height, "um": round(0.5 / f, 4),
                "tile": round(TILE * f, 2), "tiles": [[round(float(cx * f), 1), round(float(cy * f), 1), [round(float(v), 4) for v in t]] for (cx, cy), t in zip(coords, tiles)],
                "whole": [round(float(v), 4) for v in whole], "base": [round(float(v), 4) for v in base], "look": look}
        if tile_lib is not None:
            unit, c2 = et.tiles_of(r["file"])
            if len(unit) == len(tiles):  # same tissue tiles as the evidence map
                order = np.argsort(-prob[i])
                on = [k for k in order if lists[i][k]]
                a, b = (on[0], on[1]) if (not act[i] and len(on) >= 2) else (order[0], order[1])
                v = tiles[:, a] - tiles[:, b]
                picks = [(int(v.argmax()), int(a)), (int(v.argmin()), int(b))]
                near = et.nearest(unit[[p for p, _ in picks]], tile_lib[0], k=4)
                case["struct"] = []
                for (t, k), nn in zip(picks, near):
                    x0, y0 = int(coords[t][0]), int(coords[t][1])
                    name = f"img/q{stem}_{t}.jpg"
                    im.crop((x0, y0, x0 + TILE, y0 + TILE)).resize((150, 150), Image.LANCZOS).save(OUT / name, quality=82)
                    case["struct"].append({"tile": t, "for": k, "img": name, "near": [crop(tile_lib[1][j], tile_lib[2][j]) for j in nn]})
        if usable:
            pc = [(float((ref[:, j] < q[i, j]).mean()), name) for j, name in enumerate(usable)]
            case["patterns"] = [{"name": name, "pct": round(100 * p)} for p, name in sorted(pc, reverse=True)[:3] if p >= 0.80]
        cases.append(case)
        print(len(cases), stem, case["decision"], flush=True)

    look_json = json.loads((ROOT / "outputs/evidence/lookalikes.json").read_text())
    why_json = json.loads((ROOT / "outputs/evidence/why.json").read_text())
    strict = json.loads((pp.OUT / "result_strict.json").read_text()) if (pp.OUT / "result_strict.json").exists() else None
    best = look_json["spaces"][look_json["chosen_space"]]
    meta = {"types": sl.TYPES, "names": NAMES, "cutOff": system["cut_off"],
            "pairAuc": {k: v["tile_auc"] for k, v in why_json["pairs"].items()},
            "record": (f"On 300 batches of 30 new patients, at most 10% of the answers given alone were wrong in {round(100 * strict['budget_kept_in_share_of_splits'])} of 100 batches."
                       if strict else ""),
            "lookNote": (f"The nearest confirmed cases from other patients, as the classifier scores them. They show how the model reads this case and are not an independent second opinion. "
                         f"Among asked cases, {round(100 * best['precision_at_5_among_asked'])}% of the five nearest cases share the agreed type "
                         f"(a random case: {round(100 * look_json['random_share_among_asked'])}%), and using them did not make the list's order more accurate."),
            "tileNote": (f"The most similar tiles from confirmed cases of other patients, compared in the tissue model's own features, which the classifier has not shaped. "
                         f"Each neighbour carries its region's agreed type. On unseen patients, {round(100 * tiles_json['share_same_type_at_5'])}% of a tile's five nearest neighbours come from regions of the same type "
                         f"(a random tile: {round(100 * tiles_json['random_share'])}%). For UDH and ADH they are barely better than chance.") if tiles_json and tiles_json["shown"] else "",
            "patternNote": ("Scored from the pattern's name by a text-and-image model (KEEP), on up to 12 tiles of the region. Only the 8 patterns that passed their test can appear, "
                            "and only when the region scores higher than 80% of regions. The patterns that separate ADH from DCIS (cribriform, micropapillary) did not pass and are never shown." if usable else ""),
            "credits": "Tissue images: BRACS (Brancati et al. 2022), CC BY-NC 4.0, cropped and resized. Features: UNI (non-commercial; not redistributed). "
                       "Cases: 8 per type drawn at random from the 30 test patients. Research demo, not for diagnosis."}
    (OUT / "data.js").write_text("window.META = " + json.dumps(meta) + ";\nwindow.CASES = " + json.dumps(cases, separators=(",", ":")) + ";\n")
    shutil.copy(pathlib.Path(__file__).with_name("viewer_template.html"), OUT / "index.html")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{len(cases)} cases, {len(thumbs)} look-alike thumbnails, {size:.1f} MB -> {OUT / 'index.html'}")
    print("asked", sum(c["decision"] == "ask" for c in cases), "answered", sum(c["decision"] == "act" for c in cases))


if __name__ == "__main__":
    main()
