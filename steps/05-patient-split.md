# Step 5 · One patient, one group (the patient split)

**Code:** [`tools/patient_split.py`](../tools/patient_split.py) · **Ran:** 2026-11-14 13:20 · **Milestone:** M0

## In one sentence

We checked whether the official BRACS split keeps all of a patient's regions in one group, and saved one table, `data/regions.csv`, that every later step reads.

## A real example

`test/0_N/BRACS_1286_N_40.png`
1. The file name gives the slide: `BRACS_1286`.
2. BRACS.xlsx says slide `BRACS_1286` came from **patient 23**.
3. So this region belongs to patient 23.

That chain, **region → slide → patient**, is the whole join.

Patient 3 is the extreme case: 10 slides, 122 regions, 6 of the 7 types.

## Why it matters: data leakage

Regions from one patient share a lot that has nothing to do with the diagnosis:
- the same staining batch and scanner day;
- the same tissue thickness;
- the patient's own cell shapes.

If a patient has regions in training *and* in the test, the model can score well by **recognising the patient** instead of the type. The test score then promises more than the model can do for a new patient. This is **data leakage**, and it is one of the most common reasons medical-imaging results do not hold up.

BRACS.xlsx assigns the group (train, validation, test) **per slide**. A patient can have up to 10 slides, so even a split by slide could leak. Only the patient level is safe.

## Why this way

| Choice | Why | Other option |
|---|---|---|
| Check the official split instead of making our own | Published BRACS results use it, so our numbers can be compared (the M1 reproduction) | our own split grouped by patient: possible, but no longer comparable |
| Type from the folders; BRACS.xlsx only for slide → patient | the summary sheet is out of date for 2 slides (below) | trust the sheet's counts |
| Test the check on a **planted leak** too | a check that has never failed proves nothing; it could be broken and always say "no leak" | trust it |
| Save one table, `data/regions.csv` | training, error ranges and the tile map all need region → type, group, patient: one source of truth | recompute it in every script |

## The result (predict first, then open)

In the notebook, predict how many patients sit in more than one group before you open this.

<details><summary>Result</summary>

All 5 checks PASS. **0 of 151 patients** have regions in more than one group, so the official split is a patient split.

| Group | Regions | Slides | Patients |
|---|---|---|---|
| train | 3,657 | 281 | 106 |
| val | 312 | 37 | 15 |
| test | 570 | 69 | 30 |

For comparison (notebook part 2), with the same group sizes:
- shuffling **regions** at random puts about **110 of 151 patients** in two or three groups;
- shuffling whole **slides** still leaks about **51**.
</details>

## Found on the way (goes into the data card, step 6)

- **Few patients per type.** Validation has 4 to 10 patients per type (IC: 4), and the test has 11 to 19. Error ranges will be wide, and that is honest.
- **One patient can dominate a type.** In the test set, one patient gives 48% of the FEA regions. In training, one slide (`BRACS_1771`) gives 99 of the 624 FEA regions (16%). So error ranges must resample **patients**, not regions (M2).
- **The summary sheet is slightly out of date.** Slide `BRACS_1667` has 6 regions in the folders but 2 in the sheet; `BRACS_1771` has 101 vs. 105. We use the folders.
- **Stray spaces:** the column `"RoI "` and the value `"Validation "` (second sheet). The code strips them.
- Patient 67 has 3 slides in train and 2 in validation, but the 2 validation slides have no regions. That is no leak for regions; a whole-slide study would need care.
- 160 of the 547 slides have no regions (547 − 387).

## The code, part by part (`patient_split.py`)

1. **`slide_of(filename)`:** `rsplit("_", 2)` splits from the right, at most twice: `BRACS_1325_PB_11` → `BRACS_1325`, `PB`, `11`. Splitting from the left would cut the slide name, which contains an underscore itself.
2. **`list_regions(root)`:** walks `<group>/<n_TYPE>/<file>.png` and makes one row per region: file, split, label (0–6), label name, slide. It also keeps the type written in the file name, for a check.
3. **`read_slides(xlsx)`:** reads the sheet `WSI_Information` (one row per slide: slide, patient, official set), strips the stray spaces, and renames Training/Validation/Testing to train/val/test.
4. **The join:** `regions.merge(slides, on="slide", how="left", validate="many_to_one")`.
   - Many regions match one slide row.
   - `validate` stops with an error if a slide appears twice in the sheet.
   - `how="left"` keeps regions whose slide is missing, so the first check catches them instead of dropping them silently.
5. **`leaky_patients(table)`:** for each patient, count the different groups (`groupby("patient")["split"].nunique()`). Any patient with more than 1 is a leak.
6. **Five checks**, each printed as PASS or FAIL; the exit code is 1 if any fails.
7. **`regions.csv`** is written under a temporary name, then renamed, like the feature files.

Run it from the project folder:
`.venv/bin/python -I tools/patient_split.py data/bracs/BRACS_RoI/latest_version data/bracs/BRACS.xlsx data/regions.csv`

## Rebuild it yourself

In the notebook `learn/05-patient-split.ipynb`:
1. Rebuild `slide_of` and `leaky_patients`. The ✅ check runs them on the real table **and** on a planted leak.
2. Write your own split grouped by patient (`my_group_split`) and check that it never leaks.

## Check yourself

1. No region appears twice. Why can a random split of regions still give a test score that is too high?
2. BRACS.xlsx gives a group per slide. Why is a split by slide not enough?
3. Our check found no leak. Why also run it on a planted leak?
4. Why do we take each region's type from the folders and not from BRACS.xlsx?
5. One patient gives 48% of the FEA test regions. What does that mean for the FEA score's error range?

<details><summary>Answers</summary>

1. Regions of the same patient look alike in ways unrelated to the type. The model can recognise the patient, so the test measures memory, not diagnosis.
2. One patient can have several slides (up to 10 here), and a split by slide can put them in different groups. In the notebook, shuffling slides leaks about 51 patients.
3. A check is only proof if it can fail. With one region moved, it must report that patient; if it still says "no leak", the check is broken.
4. The summary sheet is out of date for 2 slides. The folders are the released data.
5. The FEA score depends heavily on one person. Resampling patients (the bootstrap in M2) shows that as a wide range; resampling regions would hide it.
</details>

## Words to know

- **Data leakage:** information from the test reaches training, so the test score is too optimistic.
- **Patient-level (grouped) split:** each patient's regions all stay in one group.
- **Slide (WSI, whole-slide image):** one scanned glass slide. **Region (RoI):** a part cut out of a slide.
- **Join (merge):** combine two tables on a shared column, here `slide`.
- **Many-to-one:** many rows on the left match exactly one row on the right.
- **Planted fault:** a deliberate error that proves a check can fail.
- **Source of truth:** the one place a fact is stored; everything else reads it from there.
