# Step 6 · The data card

**Written:** 2026-11-18 · **Milestone:** M0 (this closes it) · **Sources:** steps 1, 2 and 5, the BRACS site, `data/bracs/BRACS.xlsx`, the saved feature files

## In one sentence

One page that says what the data is, what we may do with it, and what it cannot tell us.

## What it is

- **Name:** BRACS, BReAst Carcinoma Subtyping (Brancati et al. 2022, arXiv 2111.04740).
- **Made by:** IRCCS Fondazione Pascale (Naples), ICAR-CNR and IBM Research Zurich.
- **Content:** regions of interest cut from H&E-stained breast tissue slides, scanned at 40× (0.25 µm per pixel).
- **Labels:** 7 types. Each label is the agreement of three pathologists (BRACS paper).
- **Our copy:** `data/bracs/BRACS_RoI/latest_version`, 4,539 regions from 387 slides of 151 patients. Download and checks: step 1.

## Licence: what we may do

- The BRACS site states **CC BY-NC 4.0** (https://www.bracs.icar.cnr.it/?p=3037).
- **Allowed:** non-commercial use, sharing and adapted versions, so image tiles in a public research demo are very likely fine.
- **Required credit on every page that shows tissue:** the authors, the paper, a link to the licence, and what we changed ("cropped", "resized").
- **Required citation** when results are published: the BRACS paper.
- **Not checked yet:** the download page is behind the login, and it may show extra terms. The agreement between the three institutes is not public. **The author checks the download page before the demo goes public (M3).**
- **The models have their own terms.** UNI and H-optimus-0 are non-commercial and may not be redistributed, so the demo shows precomputed results only.

## The numbers

| Group | Regions | Slides | Patients | Tiles at 20× |
|---|---|---|---|---|
| train | 3,657 | 281 | 106 | 99,368 |
| val | 312 | 37 | 15 | 9,237 |
| test | 570 | 69 | 30 | 12,023 |

Regions per type are in step 2. The split is the official one, and no patient is in two groups (step 5).

## Quirks that change how we work

| Quirk | What we do about it |
|---|---|
| Training is unbalanced (PB 714, Normal 357); the test set is balanced | class weights in training; report scores per type |
| Few patients: validation has 15 (4 to 10 per type), the test has 30 | error ranges resample **patients**, and they will be wide |
| One patient gives 48% of the FEA test regions; one slide gives 16% of the FEA training regions | same: resample patients, and read the FEA score with care |
| Region sizes run from 127 × 137 px to 17,611 × 9,842 px at 40× | tiles of 224 px at 20×; 103 regions are smaller than one tile and are padded with white |
| 7,917 of the 120,628 tiles are less than half tissue | the share of tissue is saved per tile, so training can drop them |
| One agreed label per region; **no vote per pathologist** in `BRACS.xlsx` | methods that need each expert's vote (Stutz et al. 2023) cannot be used as published |
| The summary sheet is out of date for 2 slides, and has stray spaces | types come from the folders; the code strips the spaces |
| One hospital, one scanner type | results say nothing about other hospitals until the second-hospital test |

## What this data cannot tell us

- **Nothing about real patients.** It is a research set, and the system is a research demo, not a diagnosis.
- **Nothing about whole slides.** A region was already chosen by a pathologist. Finding the region is a different problem.
- **The labels are not the truth.** They are what three pathologists agreed on. On atypia, pathologists agree only about half the time (Elmore et al. 2015), so some labels are disputed by nature.
- **How far a lesion extends.** ADH and DCIS differ partly by size, and a region or a tile does not always show that.

## The test set is locked

The 570 test regions are used **once**. That happened on 2026-11-18 (ledger entry 9), after the classifier and the system were frozen. Until then no script read them for training, tuning or picking a model: `tools/train_classifier.py` and `tools/short_lists.py` have no code path for the test set, and `tools/final_test.py` refuses to run a second time.

From now on the test numbers are history, not a target. Any later change to the classifier or the promises cannot be re-tested honestly on these 30 patients.

## Check yourself

1. Why may the demo show tissue tiles but not let anyone download UNI's features?
2. Validation has only 4 patients with invasive carcinoma. What does that do to the promise for that type?
3. Why is "three pathologists agreed" not the same as "true"?

<details><summary>Answers</summary>

1. BRACS is CC BY-NC 4.0, which allows sharing with credit. UNI's licence forbids redistribution.
2. The promise for that type rests on 4 people, so it is rough. The error range must show that.
3. Agreement measures consistency between people. On atypia, experts often disagree, so a different panel could have chosen another label.
</details>

## Words to know

- **Data card:** a short document that describes a dataset: source, licence, contents, limits.
- **CC BY-NC 4.0:** a licence that allows non-commercial sharing and adaptation with credit.
- **Locked test set:** data that is kept aside and used once, so the final score is honest.
