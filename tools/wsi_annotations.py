#!/usr/bin/env python3
"""Where the labelled regions sit on their whole slide.

BRACS gives the pathologists' outlines as QuPath .qpdata files (Java serialisation). This reads only the shapes
(polygons and rectangles, level-0 pixels) and matches each shape to a labelled region image by its width and height,
so the label comes from the region's file name and no QuPath is needed.

Usage (from the project folder):
  python -I tools/wsi_annotations.py   # -> outputs/grayzone/wsi_regions.json for the slides in data/features/uni_wsi
"""
import csv
import json
import pathlib
import re
import struct
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ZIPS = [ROOT / "data/bracs_wsi_annotations/train.zip", ROOT / "data/bracs_wsi_annotations/val.zip"]
ROI = ROOT / "data/bracs/BRACS_RoI/latest_version"


def shapes(raw):
    """Bounding boxes (x0, y0, x1, y1) of every polygon and rectangle in one .qpdata file."""
    arrays = []
    for m in re.finditer(rb"u(?:r\x00\x02\[F.{8}\x02\x00\x00xp|q\x00\x7e..)", raw, re.S):
        p = m.end()
        n = struct.unpack(">i", raw[p:p + 4])[0]
        if 2 < n < 100000 and p + 4 + 4 * n <= len(raw):
            a = np.frombuffer(raw[p + 4:p + 4 + 4 * n], ">f4")
            if np.isfinite(a).all() and a.min() >= 0 and a.max() < 3e5:
                arrays.append((m.start(), p + 4 + 4 * n, a))
    boxes = []
    for (s0, e0, x), (s1, e1, y) in zip(arrays, arrays[1:]):
        if len(x) == len(y) and s1 - e0 < 16:  # the x array, then the y array right after it
            boxes.append((float(x.min()), float(y.min()), float(x.max()), float(y.max())))
    # rectangles: ints c, t (-1, 0), then doubles x, x2, y, y2
    for m in re.finditer(rb"\xff\xff\xff\xff\x00\x00\x00\x00(.{32})\x00\x00\x00\x00", raw, re.S):
        x, x2, y, y2 = struct.unpack(">4d", m.group(1))
        if all(np.isfinite([x, x2, y, y2])) and 0 <= x < x2 < 3e5 and 0 <= y < y2 < 3e5:
            boxes.append((x, y, x2, y2))
    return boxes


def match(boxes, regions, tol=3):
    """Give each region image the one shape with the same width and height (within tol pixels)."""
    out, used = [], set()
    for name, label, w, h in regions:
        best = [(abs((b[2] - b[0]) - w) + abs((b[3] - b[1]) - h), i) for i, b in enumerate(boxes) if i not in used]
        best = [c for c in best if c[0] <= 2 * tol]
        if len(best) >= 1 and (len(best) == 1 or sorted(best)[1][0] > sorted(best)[0][0]):
            i = min(best)[1]
            used.add(i)
            out.append({"file": name, "label": label, "box": [round(v) for v in boxes[i]]})
    return out


def main():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    slides = {f.stem for f in (ROOT / "data/features/uni_wsi").glob("*.npz")}
    by_slide = {}
    with open(ROOT / "data/regions.csv") as f:
        for r in csv.DictReader(f):
            by_slide.setdefault(r["slide"], []).append(r)
    files = {}
    for path in ZIPS:
        z = zipfile.ZipFile(path)
        files.update({pathlib.Path(n).stem: (z, n) for n in z.namelist() if n.endswith(".qpdata")})
    result, total, found = {}, 0, 0
    for s in sorted(slides):
        regs = []
        for r in by_slide.get(s, []):
            with Image.open(ROI / r["file"]) as im:
                regs.append((r["file"], r["label_name"], im.width, im.height))
        boxes = shapes(files[s][0].read(files[s][1])) if s in files else []
        got = match(boxes, regs)
        result[s] = {"shapes": len(boxes), "regions": len(regs), "matched": got}
        total += len(regs)
        found += len(got)
        print(f"{s}: {len(boxes)} shapes, {len(regs)} region images, {len(got)} matched")
    (ROOT / "outputs/grayzone/wsi_regions.json").write_text(json.dumps(result))
    print(f"matched {found} of {total} region images on {len(slides)} slides")


if __name__ == "__main__":
    main()
