"""Step 5 (M0): check that the official BRACS split keeps every patient in one group,
and write the region table that every later step uses.

Usage:
    python -I tools/patient_split.py <latest_version_dir> <BRACS.xlsx> <out_csv>

Writes one row per region: file, split, label, label_name, slide, patient.
Exits with code 1 if any check fails.
"""
import os
import pathlib
import sys

import pandas as pd

CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]  # labels 0..6, from benign to malignant
SETS = {"Training": "train", "Validation": "val", "Testing": "test"}  # BRACS.xlsx name -> folder name


def slide_of(filename):
    """'BRACS_1325_PB_11.png' -> 'BRACS_1325', the slide the region was cut from.
    Split from the right, because the slide name itself contains an underscore."""
    return filename.rsplit("_", 2)[0]


def list_regions(root):
    """One row per region PNG. The split and the type come from the folder names."""
    rows = []
    for p in sorted(root.glob("*/*/*.png")):  # <split>/<n_TYPE>/<file>.png
        split, folder = p.parent.parent.name, p.parent.name  # e.g. 'train', '1_PB'
        number, name = folder.split("_", 1)
        rows.append({
            "file": p.relative_to(root).as_posix(),
            "split": split,
            "label": int(number),
            "label_name": name,
            "slide": slide_of(p.name),
            "name_in_file": p.stem.rsplit("_", 2)[1],  # the type written in the file name, for a check
        })
    return pd.DataFrame(rows)


def read_slides(xlsx):
    """Slide -> patient and official set, from the sheet 'WSI_Information' (one row per slide).
    The sheet has stray spaces (the column is called 'RoI '), so names and values are stripped."""
    w = pd.read_excel(xlsx, sheet_name="WSI_Information")
    w.columns = [c.strip() for c in w.columns]
    w = w.rename(columns={"WSI Filename": "slide", "Patient Id": "patient", "Set": "official_set"})
    w["official_set"] = w["official_set"].str.strip().map(SETS)
    return w[["slide", "patient", "official_set"]]


def leaky_patients(table):
    """Patients whose regions sit in more than one split. Must be an empty list."""
    splits_per_patient = table.groupby("patient")["split"].nunique()
    return sorted(splits_per_patient[splits_per_patient > 1].index.tolist())


def main(root, xlsx, out):
    regions = list_regions(pathlib.Path(root))
    slides = read_slides(xlsx)
    # The join: every region gets its slide's patient. many_to_one stops if a slide appears twice in the sheet.
    t = regions.merge(slides, on="slide", how="left", validate="many_to_one")

    checks = {
        "every region's slide is in BRACS.xlsx": bool(t["patient"].notna().all()),
        "type folder matches the type in the file name": bool((t["label_name"] == t["name_in_file"]).all()),
        "type number matches the type name": bool((t["label_name"] == t["label"].map(dict(enumerate(CLASSES)))).all()),
        "split folder matches the slide's official set": bool((t["split"] == t["official_set"]).all()),
        "no patient in more than one split": not leaky_patients(t),
    }

    t = t.drop(columns=["name_in_file", "official_set"])
    t["patient"] = t["patient"].astype("Int64")
    tmp = out + ".tmp"
    t.to_csv(tmp, index=False)
    os.replace(tmp, out)  # a half-written table never exists

    order = ["train", "val", "test"]
    print(f"{len(t)} regions from {t['slide'].nunique()} slides of {t['patient'].nunique()} patients -> {out}\n")
    print(t.groupby("split").agg(regions=("file", "size"), slides=("slide", "nunique"),
                                 patients=("patient", "nunique")).loc[order].to_string(), "\n")
    print("Different patients per type and split:")
    print(t.groupby(["label_name", "split"])["patient"].nunique().unstack().loc[CLASSES, order].to_string(), "\n")
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    if not checks["no patient in more than one split"]:
        print("leaky patients:", leaky_patients(t))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
