#!/usr/bin/env python3
"""Check the three headline numbers from the shipped artifacts alone.

    python reproduce.py                      # from the repo root
    python reproduce.py <regions.csv> <oof_prob_engine.npy> <val_prob_engine.npy> <positions.json>

Needs numpy only. No model, no dataset, no venv of the original project. The two pair shares are
recomputed here; the third number is a literature value and is printed as the constant it is.

What the pair shares mean: inside one slide, every ADH region is paired with every UDH region and
the engine score (P(ADH) / (P(ADH) + P(UDH))) is compared. A pair counts as right when the ADH
region scores higher, a tie counts half. Pairs are bucketed by the free-space distance between
their two boxes (0.25 um/px, so 4 px = 1 um). "Far" is above 3 mm, "close" is 0.5 mm or less.
The out-of-fold probabilities come from a model that never saw the patient of the region it scores.
"""
import csv
import json
import pathlib
import sys

import numpy as np

ADH, UDH = 4, 2  # label numbers in the region table
FAR_MM, CLOSE_MM = 3.0, 0.5
UM_PER_PX = 0.25
CLAIM_FAR, CLAIM_CLOSE, CLAIM_F1 = 0.78, 0.57, 0.38  # what README.md says
F1_SOURCE = "Pati et al., HACT-Net (arXiv 2102.11057), table VIII: pathologists re-reading BRACS reproduced ADH at F1 ~0.38"


def gap_mm(a, b):
    """Free space between two boxes in mm; 0 when they touch or overlap."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy)) * UM_PER_PX / 1000


def main(argv):
    here = pathlib.Path(__file__).resolve().parent
    regions_p, oof_p, val_p, pos_p = (pathlib.Path(v) for v in (argv or [
        here / "data/regions.csv", here / "outputs/m2/oof_prob_engine.npy",
        here / "outputs/m2/val_prob_engine.npy", here / "outputs/within_slide/positions.json"]))
    for p in (regions_p, oof_p, val_p, pos_p):
        print(f"input {p}  {p.stat().st_size:,} B")

    with open(regions_p) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    rows = [r for r in rows if r["split"] == "train"] + [r for r in rows if r["split"] == "val"]  # the order the probabilities are in
    prob = np.vstack([np.load(oof_p), np.load(val_p)])
    if len(prob) != len(rows):
        sys.exit(f"probability rows {len(prob)} do not match region rows {len(rows)}")
    score = {r["file"]: float(prob[i, ADH] / (prob[i, ADH] + prob[i, UDH])) for i, r in enumerate(rows)}
    patient = {r["slide"]: int(r["patient"]) for r in rows}

    pairs = []  # (patient, gap_mm, right) - right is 1, 0.5 or 0
    for slide, regs in json.loads(pos_p.read_text()).items():
        for a in [x for x in regs if x["label"] == "ADH"]:
            for u in [x for x in regs if x["label"] == "UDH"]:
                d = score[a["file"]] - score[u["file"]]
                pairs.append((patient[slide], gap_mm(a["box"], u["box"]), 1.0 if d > 0 else 0.5 if d == 0 else 0.0))
    p = np.array(pairs)
    if not len(p):
        sys.exit("no ADH-UDH pair on the same slide in positions.json")

    def share(mask):
        return float(p[mask, 2].mean()), int(mask.sum()), len(np.unique(p[mask, 0]))

    ok = True
    print(f"\n{len(p)} within-slide ADH-UDH pairs on {len(np.unique(p[:, 0]))} patients\n")
    for name, mask, claim in (("far pairs, gap > 3 mm", p[:, 1] > FAR_MM, CLAIM_FAR),
                              ("close pairs, gap <= 0.5 mm", p[:, 1] <= CLOSE_MM, CLAIM_CLOSE)):
        value, n, pats = share(mask)
        hit = round(value, 2) == claim or abs(value - claim) < 0.005
        ok &= hit
        print(f"{name:26s} {value:6.1%}  ({n:,} pairs, {pats} patients)  README says {claim:.0%}  {'match' if hit else 'MISMATCH'}")
    print(f"\n{'third number':26s} ADH F1 {CLAIM_F1}  (literature constant, not recomputed here)\n{'':26s} {F1_SOURCE}")
    print("\nthe claim is that the far share is well above the close share, and that the 0.38 is other people's number, not ours")
    print("\nRESULT:", "all three numbers reproduce" if ok else "at least one number does not match README")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
