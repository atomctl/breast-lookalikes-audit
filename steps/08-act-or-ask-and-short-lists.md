# Step 8 · Act or ask, and the short list

**Code:** [`tools/short_lists.py`](../tools/short_lists.py), [`tools/improve_classifier.py`](../tools/improve_classifier.py), [`tools/final_test.py`](../tools/final_test.py) · **Ran:** 2026-11-18 · **Milestones:** M1 (end) and M2 · **Ledger:** entries 5 to 10

## In one sentence

The classifier got one more view of each region, and around it we built the part that makes the project what it is: a rule that decides "answer alone or ask a pathologist", and a short list with a promise for the cases it asks about.

## A real example

`val/0_N/BRACS_1295_N_1.png` is labelled Normal. The classifier's top type is PB with probability 0.46.
- 0.46 is under the cut-off, so the system does **not** answer alone. It asks.
- Its short list is **N, PB, UDH**. The right type (N) is on it.

The classifier alone would have been wrong here. The system says "one of these three, please look".

## Part 1: a better classifier (entries 5 and 10)

Four ideas were tested by cross-validation on the training patients. A change was adopted only if its gain was clear of zero in the by-patient range.

| Idea | Gain in weighted F1 | Adopted? |
|---|---|---|
| Average the UNI and H-optimus-0 classifiers | +0.001 (−0.016 to 0.019) | no |
| Give UNI tiles at 10× and 5× too | +0.005 (−0.016 to 0.024) | no |
| Describe a region by the spread of its tiles too | −0.003 (−0.016 to 0.011) | no |
| Add UNI's view of the **whole region shrunk to 224 px** | **+0.023 (0.009 to 0.039)** | **yes** |

The last idea came from reading the THUNDER benchmark's code for the reproduction. A tile shows cells; the shrunk region shows the shape of the whole lesion. Each alone scores 0.700, and together 0.723.

The final classifier: one linear layer on UNI's tile average next to UNI's whole-region view.

## Part 2: why "a list of one type means confident" was dropped (entry 6)

The first plan had one mechanism: build a short list, and if it holds exactly one type, answer alone.

On the training patients that rule was the **worst** way to decide:

| Way to decide | Asked | Wrong among answered alone |
|---|---|---|
| Answer alone when the list has one type | 66% | 12.7% |
| Plain cut-off on the top probability, same share asked | 67% | 7.4% |

Why: with one threshold per type, a type with a loose threshold produces one-type lists that the classifier is not sure about.

So the system now has **two steps with two promises**.

## Part 3: the two steps (entries 7, 8 and 10)

**Step 1, act or ask.** Answer alone when the top probability is at least a cut-off.
- The cut-off is the lowest one for which the error among the answered calibration cases is under the **budget of 10%**, with 90% confidence.
- Cut-offs are tested from the highest down, with an exact binomial test, and the search stops at the first failure.

**Step 2, the short list.** For the asked cases only.
- Every type gets a score: 1 minus its probability.
- For each type, a threshold is set on calibration cases so that the right type stays under it 90% of the time. (Technical term: conformal prediction, the LAC score, one threshold per type.)
- A type is on the list if its score is under its threshold.
- The thresholds are set on the calibration cases **that would be asked**, because those are the cases the list is for.

**Where the calibration cases come from.** Every training region is scored by a classifier that never saw its patient (the 5 folds). That gives 3,657 honest scores from 106 patients, not 312 from 15.

## What went wrong, and the fix (entries 7 and 8)

The first version of step 2 missed its own bar: for DCIS, the right type was on the list in 79.95% of asked cases, under the 80% stop rule.

The cause was not spread evenly. **Two of 31 patients carried 70 of the 82 DCIS misses.** Regions of one patient are alike, so 49 regions from one person are not 49 independent cases.

The fix: when setting a type's threshold, count the **patients** with that type, not the regions. That makes the threshold stricter. It is a caution for clustered data, not a proof.

| | Lists before the fix | Lists after the fix |
|---|---|---|
| Right type on the list, among asked | 87.8% | 93.2% |
| Lowest type | 80.0% (DCIS) | 90.7% (DCIS) |
| Types on a list | 2.5 | 3.1 |

The price is longer lists.

## The checks before the test (final classifier, entry 10)

| | 106 training patients | 15 validation patients |
|---|---|---|
| Asked | 55.3% (48.8–62.4%) | 55.5% |
| Wrong among answered alone (budget 10%) | 8.7% (5.9–12.0%) | 8.6% |
| Right type on the list, among asked (promise 90%) | 93.3% (90.9–95.3%) | 98.3% |
| Types on a list | 3.2 | 3.3 |
| Wrong with the classifier alone | 27.8% | 30.8% |

## The test: 30 patients the system had never seen (entry 9)

The test set was opened once, on 2026-11-18, after everything above was frozen.

| | Aim | Test result |
|---|---|---|
| Asked | | 53.3% (45.5–59.7%) |
| Wrong among answered alone | at most 10% | **18.0% (10.3–25.2%)**, 48 of 266 |
| Right type on the list, among asked | at least 90% | **89.1% (84.5–94.1%)** |
| Types on a list | | 2.9 |
| Wrong with the classifier alone | | 34.9% |
| Missed with the system (wrong answer, or not on the list) | | 14.2% |

- **Promise 1 was not kept.** 18% of the answers given alone were wrong, and even the low end of the range is above the 10% budget.
- **Promise 2 was missed by one point.** 90% lies inside the range, so this is within what chance allows.
- **The system still helps:** it cuts missed cases from 34.9% to 14.2% while asking about half the time. But it does not do what it promised.

**Why promise 1 failed** (looked at after the test; nothing was re-tuned):
- The test patients were harder. The classifier was wrong on 34.9% of test regions, against 27.8% on the training patients, and it stayed just as confident.
- 3 of 24 patients carry 48% of the wrong answers.
- 15 of the 48 wrong answers are pathological benign called normal.

The cut-off was learned on 121 patients. It did not carry over to these 30. That is the honest result of this milestone, and the reason the next parts matter: a cut-off that adapts as pathologists' answers come back, and a test on a second hospital.

**Other test results:**
- The final classifier scores weighted F1 0.646 (0.569–0.722). The best verified published score on this split is 65.8 (FECT, a 2025 preprint), so ours is close to it and not above it.
- Tissue models beat general models on the test set too: UNI − ViT-L = 0.086 (0.028–0.145).
- The reproduction matched: with THUNDER's protocol we get macro F1 59.2 for UNI (published 59.4) and 61.1 for H-optimus-0 (published 61.1).

All numbers, type by type and patient by patient, are on the results page: `outputs/results/index.html`.

## What this does not promise

- **Not each patient.** The promises are averages over patients. One patient can be served much worse.
- **Not each type on its own with certainty.** With 11 to 19 test patients per type, per-type numbers are rough.
- **Not another hospital.** All of this is one centre's data.

## Check yourself

1. Why is "answer alone when the list has one type" a worse rule than a cut-off on the top probability?
2. Why are the list thresholds set only on calibration cases that would be asked?
3. Two patients caused most DCIS misses. Why does counting patients, not regions, make the threshold stricter?
4. Lowering the error budget from 10% to 5% does what to the share of cases asked?

<details><summary>Answers</summary>

1. A per-type threshold can be loose for one type, so a one-type list can come from a case the classifier is unsure about. The top probability measures that directly.
2. The list is only shown for asked cases, and those are harder than average. Thresholds from all cases would be too loose for them.
3. With fewer independent cases, the rule that sets the threshold has to leave more room, so it picks a higher threshold.
4. It raises it: on the training patients from 55% to 73%.
</details>

## Words to know

- **Conformal prediction:** a way to turn any classifier's scores into lists with a promised hit rate, using held-back cases.
- **Calibration cases:** held-back cases used to set the thresholds.
- **Out-of-fold score:** a score from a classifier that did not train on that case's patient.
- **Error budget:** the share of wrong answers we accept among the cases the system answers alone.
- **Clustered data:** cases that come in groups (here, regions of one patient) and are alike within a group.
