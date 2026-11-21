#!/usr/bin/env python3
"""Ledger entry 55, milestone 3: the gate. The slide-level step must work on BRACS slides before any HISTAI slide is scored.

20 BRACS slides of training patients are downloaded again, one at a time (8 benign, 6 ADH, 6 DCIS; seed 0).
The classifier has seen these patients' regions, so this is a test of the plumbing, not of the classifier.

Usage (from the project folder):
  python -I tools/slide_gate.py plan    # choose the slides -> outputs/slide_step/gate_slides.json
  python -I tools/slide_gate.py run     # download, run the slide-level step, delete the slide -> data/features/slide_step/bracs/
  python -I tools/slide_gate.py test    # the three checks in numbers -> outputs/slide_step/gate.json
  python -I tools/slide_gate.py look    # the check by eye: downloads 4 slides again -> outputs/slide_step/gate_top_doubt_<n>.png
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/slide_step"
FEATS = ROOT / "data/features/slide_step/bracs"
TMP = ROOT / "data/bracs_wsi"
GROUP = {"N": "BT", "PB": "BT", "UDH": "BT", "FEA": "AT", "ADH": "AT", "DCIS": "MT", "IC": "MT"}
GRAY = ("UDH", "FEA", "ADH", "DCIS")
NAMES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]


def plan():
    import pandas as pd
    d = pd.read_excel(ROOT / "data/bracs/BRACS.xlsx")
    d = d[d["Set"] == "Training"]
    marked = json.loads((ROOT / "outputs/grayzone/wsi_regions.json").read_text())
    rng = np.random.default_rng(0)

    def pick(labels, n, need_marks):
        names = sorted(x for x in d[d["WSI label"].isin(labels)]["WSI Filename"] if not need_marks or (x in marked and marked[x]["matched"]))
        return [str(x) for x in rng.choice(names, n, replace=False)]

    chosen = pick(["UDH"], 4, False) + pick(["PB"], 2, False) + pick(["N"], 2, False) + pick(["ADH"], 6, True) + pick(["DCIS"], 6, True)
    label = dict(zip(d["WSI Filename"], d["WSI label"]))
    rows = [{"slide": s, "label": label[s], "remote": f"BRACS_WSI/train/Group_{GROUP[label[s]]}/Type_{label[s]}/{s}.svs"} for s in chosen]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_slides.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


def run():
    import tiffslide
    import slide_step as ss
    import wsi_features as wf
    marked = json.loads((ROOT / "outputs/grayzone/wsi_regions.json").read_text())
    FEATS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    uni = ss.Uni()
    for r in json.loads((OUT / "gate_slides.json").read_text()):
        dest = FEATS / f"{r['slide']}.npz"
        if dest.exists():
            continue
        local = TMP / f"{r['slide']}.svs"
        size = wf.download(r["remote"], local)
        s = tiffslide.TiffSlide(local)
        mpp = float(s.properties.get("tiffslide.mpp-x") or 0.25)
        s.close()
        boxes = [m for m in marked.get(r["slide"], {}).get("matched", []) if m["label"] in GRAY]
        out = ss.process(local, mpp, uni, [m["box"] for m in boxes])
        np.savez(dest, **out, box_file=np.array([m["file"] for m in boxes]), box=np.array([m["box"] for m in boxes], np.float64).reshape(-1, 4))
        local.unlink()
        print(r["slide"], r["label"], f"{size / 1e9:.2f} GB | windows", len(out["windows"]), "| marked gray-zone boxes", len(boxes), flush=True)


def doubt(prob):
    top = prob.max(1)
    return 1 - (top + top.mean()) / 2   # 1 minus the patient-context signal, the slide standing for the patient


def test():
    import joblib
    import tiffslide  # noqa: F401  (only to fail early if the environment is wrong)
    from sklearn.metrics import roc_auc_score
    import short_lists as sl
    from train_classifier import rows
    engine = joblib.load(ROOT / "outputs/m2/engine.joblib")
    train = rows("train")
    where = {r["file"]: i for i, r in enumerate(train)}
    region_prob = engine.predict_proba(sl.engine_features("train"))
    slides = json.loads((OUT / "gate_slides.json").read_text())
    found, same, n_box, score, per = 0, 0, 0, {}, []
    for r in slides:
        z = np.load(FEATS / f"{r['slide']}.npz", allow_pickle=False)
        w, step = z["windows"], int(z["step"])
        side = 3 * step
        cx, cy = w[:, 0] + side / 2, w[:, 1] + side / 2
        f = 0
        for (x0, y0, x1, y1), bp, name in zip(z["box"], z["box_prob"] if "box_prob" in z.files else [], z["box_file"]):
            inside = ((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)).any()
            bx, by = (x0 + x1) / 2, (y0 + y1) / 2
            covers = ((w[:, 0] <= bx) & (bx < w[:, 0] + side) & (w[:, 1] <= by) & (by < w[:, 1] + side)).any()
            f += bool(inside or covers)
            if str(name) in where and np.isfinite(bp).all():
                same += int(bp.argmax() == region_prob[where[str(name)]].argmax())
                n_box += 1
        found += f
        score[r["slide"]] = float((z["prob"][:, 5] + z["prob"][:, 6]).max()) if len(w) else 0.0
        per.append({"slide": r["slide"], "label": r["label"], "tissue_tiles": int(z["tissue_tiles"]), "windows": int(len(w)), "marked_boxes": int(len(z["box"])), "boxes_found": f,
                    "malignancy_score": round(score[r["slide"]], 3)})
    boxes = sum(p["marked_boxes"] for p in per)
    y = [1 if r["label"] == "DCIS" else 0 for r in slides if r["label"] != "ADH"]
    s = [score[r["slide"]] for r in slides if r["label"] != "ADH"]
    result = {"slides": len(slides), "marked_gray_zone_boxes": boxes, "boxes_found_share": round(found / boxes, 3),
              "boxes_scored_both_ways": n_box, "same_top_type_share": round(same / max(n_box, 1), 3),
              "dcis_against_benign_slides_auc": round(float(roc_auc_score(y, s)), 3),
              "median_malignancy_score": {g: round(float(np.median([score[r["slide"]] for r in slides if r["label"] in ls])), 3) for g, ls in (("benign", ("N", "PB", "UDH")), ("ADH", ("ADH",)), ("DCIS", ("DCIS",)))},
              "per_slide": per}
    result["pass_without_the_check_by_eye"] = bool(result["boxes_found_share"] >= 0.85 and result["same_top_type_share"] >= 0.85 and result["dcis_against_benign_slides_auc"] >= 0.80)
    (OUT / "gate.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def look():
    """The check by eye: the 50 proposals with the most doubt, cut from their slides again -> gate_top_doubt_<n>.png."""
    import tiffslide
    from PIL import Image, ImageDraw
    import wsi_features as wf
    slides = {r["slide"]: r for r in json.loads((OUT / "gate_slides.json").read_text())}
    rows = []
    for name in slides:
        z = np.load(FEATS / f"{name}.npz", allow_pickle=False)
        if len(z["windows"]):
            rows += [(float(d), name, int(w[0]), int(w[1]), 3 * int(z["step"]), NAMES[p.argmax()]) for d, w, p in zip(doubt(z["prob"]), z["windows"], z["prob"])]
    top = sorted(rows, reverse=True)[:50]
    TMP.mkdir(parents=True, exist_ok=True)
    cut = {}
    for name in sorted({t[1] for t in top}):
        local = TMP / f"{name}.svs"
        wf.download(slides[name]["remote"], local)
        s = tiffslide.TiffSlide(local)
        for i, t in enumerate(top):
            if t[1] == name:
                cut[i] = s.read_region((t[2], t[3]), 0, (t[4], t[4])).convert("RGB").resize((400, 400), Image.LANCZOS)
        s.close()
        local.unlink()
        print(name, "cut", flush=True)
    for n in range(2):
        sheet = Image.new("RGB", (2000, 2000), "white")
        for k in range(25):
            i = n * 25 + k
            sheet.paste(cut[i], (k % 5 * 400, k // 5 * 400))
            ImageDraw.Draw(sheet).text((k % 5 * 400 + 4, k // 5 * 400 + 4), f"{i + 1} {top[i][1][6:]} {slides[top[i][1]]['label']} top={top[i][5]} d={top[i][0]:.2f}", fill="black", stroke_width=2, stroke_fill="white")
        sheet.save(OUT / f"gate_top_doubt_{n + 1}.png")
    (OUT / "gate_top_doubt.json").write_text(json.dumps([dict(zip(("doubt", "slide", "x", "y", "side", "top_type"), t)) for t in top], indent=1))


if __name__ == "__main__":
    {"plan": plan, "run": run, "test": test, "look": look}[sys.argv[1]]()
