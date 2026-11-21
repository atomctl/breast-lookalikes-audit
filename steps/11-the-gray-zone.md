# Step 11 · The gray zone

**Code:** [`tools/grayzone_guard.py`](../tools/grayzone_guard.py), [`tools/wsi_features.py`](../tools/wsi_features.py), [`tools/wsi_lesion_extent.py`](../tools/wsi_lesion_extent.py), [`tools/grayzone_arrangement.py`](../tools/grayzone_arrangement.py), [`tools/duct_tokens.py`](../tools/duct_tokens.py), [`tools/duct_measure.py`](../tools/duct_measure.py), [`tools/build_reader_study.py`](../tools/build_reader_study.py) · **Ledger:** entries 23 to 42 · **Dates:** 18 and 19 Nov 2026

## In one sentence

The classifier could not be made better at naming the four hard types (UDH, FEA, ADH, DCIS), so the system was made safer on them instead, and a study with pathologists was prepared to show whether that helps.

## The four hard types

| Type | Cancer? | Told apart from its neighbour by |
|---|---|---|
| UDH | No | Cell type: mixed cells, where ADH has uniform ones |
| FEA | No, a warning sign | Arrangement: a flat lining, where ADH builds bridges |
| ADH | No, a raised risk | Quantity: under about 2 mm, where DCIS is larger |
| DCIS | Yes, not yet invasive | |

The yardstick for all of this step is the **gray-zone macro F1**: the average F1 over these four types, on regions that truly are one of them. The adopted classifier scores **0.705**, and **0.55** on ADH.

## What was tried, and what came of it

Every attempt was written into the ledger with its rule before it ran.

| Attempt | Entry | Result |
|---|---|---|
| Lesion size, slide context, a specialist model, stacking | 24, 25 | No gain |
| 40× magnification | 26 | Slightly worse |
| Extent of DCIS-like tissue across the whole slide | 27–31 | 0.86 on a 20-slide pilot, 0.58 on 48 new slides, 0.51 on untouched validation slides |
| Extent around each marked lesion | 33 | As good as the tile score (0.81 against 0.82), nothing on top |
| Find atypia, then let the measured extent decide | 34 | Worse (0.70 against 0.82) |
| A second tissue model joined with UNI | 38 | +0.001 |
| Sharper whole-region views (448 and 672 pixels) | 40 | +0.003 and −0.010 |
| Models that keep the arrangement of the tiles | 38 | The transformer with tile positions reaches 0.703 from tiles alone, level with the adopted 0.705 |
| Duct measurements from UNI's patch tokens | 41 | 0.60 alone, +0.0015 added to the classifier |
| Single nuclei inside the duct cells (StarDist), UDH against ADH | 46 | Failed all four tests: no textbook criterion confirmed, 0.62 alone against the classifier's 0.88, no help where the classifier is wrong. Seen afterwards: UDH nuclei are smaller and closer together (0.74) |

**ADH is two populations (entry 47).** 303 of the 387 training ADH regions come from slides that also hold DCIS or invasive carcinoma. In UNI's own numbers these lie toward the malignant side, and the ADH of ADH-only slides near the benign side (AUC 0.81). On the training patients the classifier names the first kind correctly in 66% of regions and the second kind in 25%. The validation patients (9 with ADH) do not confirm that gap, so it is a lead. A paper from Granada found the same split on whole slides (`papers/research-notes-2026-11-19-granada-vs-ledger.md`).

**The one pattern in all of this:** only arrangement moves the gray zone. The UNI tile average alone scores 0.654; adding the small whole-region view gives 0.705; a model that knows where each tile sits gives 0.703. Size, context, magnification and a second model give nothing.

## Why the score stopped

Three checks (entry 35):

1. **The labels cannot be checked for doubt.** BRACS published only regions the pathologists tagged "sure".
2. **More data helps slowly.** With 21, 42, 64 and 85 training patients the score is 0.61, 0.67, 0.70 and 0.705.
3. **The errors are near misses.** Of the wrong gray-zone answers only 17% were given with high confidence, and in 56% the agreed type was the runner-up.

In the best-known study of pathologists (Elmore et al., JAMA 2015), individual pathologists agreed with an expert consensus in 75% of readings overall and in 48% for atypia. Our classifier agrees with its consensus in 84% (training patients) and 80% (validation patients) on the same four groups. The cases are not the same, so this is not a fair race; it suggests the classifier is in the range where the label itself is the limit.

## What works: the guard

When the system answered alone on a region that was truly ADH, it was wrong one time in six. The guard (entry 36) asks the pathologist whenever a region's ADH probability reaches the value that 90% of truly ADH regions reach, about 5%.

| Over 300 patient splits | Without guard | With guard |
|---|---|---|
| ADH regions answered alone and wrongly | 4.0% | 0.8% |
| The 10% promise kept | 94.3% of splits | 97.3% of splits |
| Regions asked instead of answered | 66.6% | 75.6% |

It is frozen as `outputs/promise/system_v5.json`. The system never answers "ADH" alone.

**Limit:** the weak spot was found on the same 121 patients, so the size of the gain is optimistic.

## The duct instrument (prototype)

UNI gives one description per 16 pixels (8 micrometres). Grouped into 10 kinds without any labels, these descriptions separate connective tissue, the rim of a duct, open space and duct cells. Pictures: `outputs/ducts/`.

- The cell layer thickens as the textbook says: FEA 0.066 mm, UDH 0.080, ADH 0.091, DCIS 0.137 (ADH above FEA: AUC 0.79; DCIS above ADH: 0.74).
- Two measurements fail as built: fullness (open spaces cut by the edge of the region are not counted) and roundness of holes (too coarse at 8 micrometres).
- The tissue kinds were named by an AI from pictures. A pathologist has not checked them.

Its use is to explain and to measure, not to raise the score.

## The reader study (ready, not run)

The question no number above can answer: **do pathologists agree more with the consensus on gray-zone cases when they have the system than when they do not?** No published study has asked this for ADH.

- **Kit:** `outputs/reader_study/index.html`. 92 regions from the 30 test patients (20 each of UDH, FEA, ADH and DCIS, and 4 each of N, PB and IC), from 22 patients. Part 1 without help, part 2 with the system's answer or short list. Answers are saved to a file.
- **The key** (`key.json`) stays with the organiser; the page holds no labels.
- **Size:** with 3 readers and 80 gray-zone cases, a gain of 10 points is found about 8 times in 10 (`tools/reader_study_power.py`, a simulation that assumes 60% agreement without the system).
- **Analysis:** put the saved files in `outputs/reader_study/responses/` and run `tools/reader_study_analyse.py`. It reports agreement without and with the system, the system alone, agreement between readers, ADH on its own, and how often a reader moved from right to wrong after a wrong hint.
- **On these 92 cases the system** answers alone 7 times (1 wrong), asks 85 times (its list holds the agreed type 68 times), and its top answer is right 51 times.

**Limits:** both parts are read in one sitting, so a reader may remember a case from part 1; a gap of some weeks would be better. The images are real tissue under a non-commercial licence, so the kit stays on local machines.

## What is still open

- A stronger tissue model (Midnight was stopped at 13% of its extraction; UNI2-h needs the conditions accepted on Hugging Face).
- Training UNI itself on the labelled tiles (`tools/grayzone_finetune.py`, written and smoke-tested, not run).
- An outside test on HISTAI-breast (access granted; about 11 ADH and 54 DCIS cases by keyword).
- Single cells inside each duct for the instrument.
