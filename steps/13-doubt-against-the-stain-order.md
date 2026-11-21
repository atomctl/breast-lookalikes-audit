# Step 13 · Doubt against the stain order (running since 20 Nov under ledger entry 55)

**Date:** 19 Nov 2026 · **From:** a four-round debate between Claude and DeepSeek ([`papers/debate-2026-11-19-claude-deepseek.md`](../papers/debate-2026-11-19-claude-deepseek.md)); Claude had the last word · **Ledger:** entry 55 holds the frozen rule and the results. Where this page and the ledger differ, the ledger counts.

## Status (20 Nov, night)

The user's order: work as if nobody will ever answer, and decide without asking.

| Milestone | State |
|---|---|
| 0 Rule frozen | Done: ledger entry 55, before any label |
| 1 Labels | **Done.** 478 reports, two labellers (DeepSeek, Claude Sonnet), stain label agreed in 474 of 478. **Primary population: 121 UDH cases, 38 with a stain and 83 on H&E alone.** The power rule (30 and 30) is met so far. Real CK5/6 result stated: kept 18, lost 13 |
| 2 Slide-level step | Built: `tools/slide_step.py`. Its tile path reproduces the saved BRACS features exactly. Tried on three HISTAI slides of cancer cases outside the study |
| 3 Gate on 20 BRACS slides | **Not passed (20 Nov).** Numbers passed: regions found 93.9%, same top type 95.6%, DCIS against benign slides AUC 0.938. The check by eye failed: 15 of the 50 most doubtful windows show no duct (smooth material with knife marks, blood, foamy material). Sheets: `outputs/slide_step/gate_top_doubt_1.png`, `_2.png`. Details: ledger entry 55 |
| 4 HISTAI run | Extracted on Colab (Drive folder `histai_out`), **not opened**. The frozen rule says: delete unread, repair the step on BRACS first |
| 5 Analysis | Not started |

## The idea in three lines

1. Our problem: the answer key for the hard cases (UDH against ADH) is a disputed name.
2. A key that nobody disputes already exists in another hospital: **did the pathologist need an immunostain to sign out the case, or not?** That is an action, written in the report.
3. Our system was built to know when to ask. So the test is: **does the system's doubt match the pathologist's doubt, at a hospital it has never seen, without any training on that decision?**

## What is needed now, later, not needed

| Now | Later | Not needed |
|---|---|---|
| Hand labels of the HISTAI-breast reports (text on disk) | One other organ, if breast passes | A new classifier |
| A slide-level step: find duct regions, score each with the frozen system | The Magdeburg stain-and-outcome cohort | More readers on the 146 regions |
| The gate on BRACS slides | HISTAI-mixed (real stain slides) when access comes | A second foundation model |

## The plan

| # | Milestone | What is done | Check | Time |
|---|---|---|---|---|
| 0 | Rule frozen | Ledger entry with everything below; hash and time of the labelling rule | The entry exists before milestone 1 | 1 hour |
| 1 | Labels | The 478 HISTAI-breast reports whose conclusion does not affirm a malignant tumour (`tools/histai_reports.py select`), labelled by the written rule `data/histai_meta/stain-label-rule.md`, blind to any model score. **No person labels** (the working assumption from 19 Nov: nobody answers). Two labellers work apart: DeepSeek, and Claude Sonnet agents. Claude decides each disagreement by the rule and writes down why | Agreement of the two labellers per field (kappa), reported before any decision | 1 day |
| 2 | Slide-level step | Duct regions proposed from the token map of entry 41; each scored by the frozen system v5 | Unit test on one BRACS slide | 3 to 5 days |
| 3 | **Gate, on about 20 BRACS slides downloaded again** | Proposals must contain at least 85% of the marked gray-zone regions; the slide score must recover the BRACS slide groups; the 50 proposals with the most doubt are looked at by eye for scar and bleeding | **No pass, no HISTAI run.** The honest outcome is then "the region model was never slide-ready" | 2 to 3 days |
| 4 | HISTAI run | Slides streamed one at a time: download, extract, delete. Scores written without looking at labels | Positive control first (below) | About 8 hours of compute |
| 5 | Analysis and write-up | Primary endpoint, controls, strata | One command reproduces every number | 2 to 3 days |

## The rule (to be copied into the ledger before milestone 1)

- **Primary endpoint.** Population: HISTAI-breast cases of the label group `UDH` (usual ductal hyperplasia affirmed anywhere in the report; nothing more severe affirmed), with at least 8 proposed duct regions. The same test on the cases that name it in the conclusion itself is a sensitivity check. Positive: label STAIN (below). Negative: label HE_ALONE. Score: the mean doubt of the slide's 5 most doubtful duct regions (doubt = 1 minus the patient-context signal of system v5). Statistic: AUC with a case bootstrap.
- **PASS:** the lower 95% bound is above 0.60. **FAIL:** the AUC is under 0.60. **Between:** weak.
- **Power rule, read after labelling and before any slide is scored:** with fewer than 30 positives or fewer than 30 negatives the primary is declared too small, and the result is reported as a pilot.
- **Positive control, read before the primary:** the slide-level step must separate DCIS cases from benign cases at an AUC of at least 0.80, with the classifier's own malignancy score (the highest P(DCIS) + P(IC) over the slide's regions), not with doubt. If not, the pipeline failed and the primary says nothing about doubt.
- **Is it doubt, or only "there is a lesion to see"?** The same AUC after the log of the number of duct regions is taken out of the doubt. Reported next to the primary; if the AUC falls under 0.60 there, the pass is called "lesion size, not doubt".
- **Baseline:** plain top probability without the project's calibration. The system's signal must beat it for the "promise machinery" to count.
- **Lesions that draw a stain by habit** (papilloma, sclerosing adenosis, radial scar, columnar change) are flagged per case; the primary is also reported without them and for each alone.
- **Excluded and counted:** slides with fewer than 8 duct regions. If more than 15% of the population is excluded, the result is marked "selected cases".
- **Sensitivity checks, reported:** top-3 and the 90th percentile in place of top-5; cases whose stain was ordered to rule out invasion left out; cases with "no stain needed" in words left out; any-stain over all diagnoses as the positive class (expected to be easy and wrong, because DCIS gets routine receptor tests).
- **Secondary:** the same test in each diagnosis group (the 2 × 2 of stain against diagnosis); gray-zone entropy (UDH against ADH and DCIS) as a second score; the ADH cases on carcinoma-free slides (about 15): is the blind spot of entries 47 and 53 there again?
- **The stain's direction (DeepSeek's first choice when nobody answers; small and honest):** the label field `ck56` says what the real CK5/6 stain showed in the lesion: kept or lost. Do the duct-level stain score of entry 49 and the classifier's UDH-against-ADH score point the same way as the real stain? Reported as an AUC with its range; with fewer than 10 cases on a side it is a pilot without a verdict. A first reading by hand found about 16 such cases ([`data/histai_meta/ck5-cases.md`](../data/histai_meta/ck5-cases.md)).

## The labelling rule (draft by DeepSeek, corrected by Claude)

Read the whole report. Ask in this order and stop at the first match.

| # | Question | Label |
|---|---|---|
| 1 | Is the report empty, cut off, or does it only point to an outside study without its content? | UNCLEAR (left out, counted) |
| 2 | Does it name any immunostain as ordered, performed, reviewed, recommended or advised? For benign cases ER counts too, because there it is used to tell UDH from ADH. | STAIN |
| 3 | Does it say in words that no stain is needed? | HE_ALONE, flag `said_so` |
| 4 | Nothing of the above, and a full microscopic description is present | HE_ALONE |

Every case also gets: the diagnosis group from the conclusion (UDH only; UDH with ADH; ADH; FEA or columnar; DCIS; carcinoma; other), the habit-lesion flags, the number of slides, and for STAIN a reason (`diagnosis`, `rule out invasion`, `not said`) and whether the stain was done or only recommended.

Edge cases:
- Stain recommended, never done: STAIN, `recommended`.
- Stain done on an outside or consult block: STAIN, flag `other block`.
- "Myoepithelium is present" with no antibody and no word for immunostaining: HE_ALONE, flag `myoepithelium in words`. With such a word: STAIN.
- ER "mosaic" in a benign case: STAIN, reason `diagnosis`.
- ER, PR, HER2, Ki-67 on DCIS or carcinoma, alone: not a diagnostic stain (these cases are outside the primary).
- UDH and ADH in one case: group "UDH with ADH"; outside the primary, in the secondary strata.
- Several slides, stain on one: STAIN for the case; all slides are scored and the most doubtful slide stands for the case.

Corrected against DeepSeek's draft: a full report without any stain is HE_ALONE, not "unclear" (its rule would have removed almost all negatives); ER in a benign case is a diagnostic stain, not a receptor panel.

## Known beforehand

- **Already seen, before the rule:** the 59 reports that name CK5 were read on 19 Nov, and rough word counts were made (246 non-invasive duct-lesion cases, 114 with stain words, about 15 with ADH in the conclusion). No slide was opened and no score exists.
- **A stain order is not pure doubt.** Stains are also ordered to rule out invasion, by lab habit, and for consult cases. The within-diagnosis design and the strata reduce this; they do not remove it.
- **The earlier whole-slide attempts failed** (entries 27 to 34: 0.86 on a pilot, 0.58 on the real test; false alarms on scar and bleeding). The gate exists because of that.
- **Not new by itself:** the Granada paper already ran four tissue models on HISTAI. New here: the answer-or-ask behaviour, the report text, and the stain order as the reference.
- **To check before any claim of "first":** a prostate study of about 2021 that trained a model to predict stain orders (from Claude's memory, not opened); a search of this year's work for breast.
- **This does not replace the real key.** It gives an action key and a stain-informed second hospital. What makes a case ADH is still settled only by a real stain or by what the surgery found.

## Honest chances

- The gate (milestone 3): about 1 in 2.
- The primary endpoint, if the gate passes: about 1 in 3.
- Both: about 1 in 6. A fail still leaves three things that stand: the labelled report set, the slide-level step with its gate, and a clear statement of whether the system's doubt travels.

## The claim if it passes (DeepSeek's second, more careful sentence)

"In one unseen hospital, a frozen model never trained on ordering gives higher slide-level doubt to benign duct proliferations that a pathologist chose to resolve with an immunostain than to those signed out on H&E alone." The first version said "needed" and "matches"; only an order was observed, and an AUC is not a match.

## Second track, when HISTAI-mixed opens (DeepSeek's first choice when it was asked without our plan)

On 19 Nov at midnight the download was still refused (the file list is public, the files are not).

- **The idea:** build the answer key from the real stain, not from a name. Register each H&E slide to its CK5/6 slide, read the stained share per duct, and score the duct-level stain model of entry 49 against it. This is gate 2 of step 12.
- **First check, about 2 hours:** 6 cases, 30 ducts. Can the same duct be found on both slides at 20×?
- **Kill:** under 60% of the ducts can be matched.
- **Its honest chance, by DeepSeek:** 0.30. Limit known beforehand: 16 of the 22 cases are invasive cancers, so most ducts will be normal or DCIS, with few UDH or ADH.

## Stretch, in this order

1. **One other organ** (for example colon) with a generic classifier, only after the breast number exists. If the sign repeats, the test becomes a benchmark: "does a model's doubt match the pathologist's doubt?" HISTAI holds 46,128 cases with reports.
2. **Magdeburg, the real key:** the lab system can list the breast biopsies where CK5/6 was ordered for a duct lesion. For each: the H&E slide, the real stain result, the final diagnosis, and what the excision found. This needs consent, an ethics vote and scanning: months, the next project.

## This week, at no cost

- Email to the BRACS authors: the three pathologists' per-region answers, and whether stained slides exist for the gray-zone cases.
- First contact with Magdeburg pathology: 20 minutes, one question ("for which of these would you order CK5/6, and what do you look at?"), the reader kit as a second request.

## Where the two models differed

| Point | DeepSeek | Claude (decides) |
|---|---|---|
| Positive class | Any immunostain | Diagnostic stains only; receptor panels would fake a pass through DCIS |
| "Behaviour change" as the measure for a reader study | Proposed it | Refused: readers can agree more by following a wrong machine; it needs an arbiter |
| Gate | Duct recall with a false-duct limit | No false-duct limit: BRACS marks only representative regions |
| Magdeburg | "The next paper", no action now | Agreed on the order, but the two emails go out this week |
| DeepSeek's additions that were taken | | The positive control, the "no stain needed" flag, the minimum of 8 duct regions, the frozen and hashed rule, one organ only as the stretch |
| Second session, maximum thinking: hostile review | Five ways to a wrong conclusion | Taken: top-5 in place of a percentile, the lesion-size control, the habit-lesion flags, the exclusion limit, the clearer positive control, the more careful claim |
| Second session: the labelling rule | No stain mentioned = "unclear"; ER "mosaic" = receptor panel | Both overruled (see the rule) |
| Second session, asked without our plan | First choice: the real-stain key by registration. Blind spot: "the 146 regions were never verified; the model infers from slide composition" | The stain key is the second track. The second half is wrong as mechanics (the classifier never sees the slide) but right as a warning: the labels may follow the slide |
| A reader study scored against the assisted consensus | Proposed again | Refused again, for the same reason |
