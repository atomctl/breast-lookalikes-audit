# Step 9 · Why the promise broke, and the outside check

**Code:** [`tools/loop_test.py`](../tools/loop_test.py), [`tools/patient_promise.py`](../tools/patient_promise.py), [`tools/second_look.py`](../tools/second_look.py), [`tools/second_hospital.py`](../tools/second_hospital.py) · **Ran:** 2026-11-18 · **Ledger:** entries 11 to 15

## In one sentence

After the test showed 18% wrong against a promise of 10%, we found the cause, rebuilt the "answer alone" step around patients, measured what it can honestly state, and checked the system on another hospital's images.

## A real example

Patient 29 has 49 DCIS regions in training, and the classifier misreads most of them in the same way. For the old cut-off test these were 49 separate pieces of evidence. In truth they are closer to one: one person, one staining, one tumour. That is the whole bug.

## The cause (entry 12)

The cut-off said: "with 90% confidence, at most 10% of my lone answers are wrong." We tested the sentence itself.

- Take the 121 training and validation patients.
- 300 times: set the cut-off on 91 of them, then check it on the other 30.
- A true "90% confidence" keeps the budget in about 90% of the 300.

**The old cut-off kept it in 62%.** Its confidence came from counting 1,600 regions as 1,600 independent cases.

## The rebuild (entries 12 and 13)

| Way to set the cut-off | Splits that kept the budget | Wrong on average | Asked |
|---|---|---|---|
| Counting regions (the tested system) | 61.7% | 9.2% | 55% |
| Resampling whole patients | 72.7% | 8.0% | 60% |
| Built for a batch of 30 patients | 84.7% | 6.5% | 67% |
| **Built for a batch of 30 patients, with patient context** | **85.3%** | **7.0%** | **59%** |

Two ideas are in the last row.

1. **Resample patients, in batches of 30.** To judge a cut-off, draw 30 patients at random, many times, and look at the error in each draw. A cut-off passes only if 90% of those draws stay under 10%. That counts two kinds of luck: which patients we learned from, and which 30 arrive next.
2. **Patient context.** A region's confidence is averaged with the mean confidence over the same patient's regions. It needs no labels. It works because some patients are simply harder: a patient's average confidence predicts that patient's error rate (rank correlation −0.54).

**The bar was 88%, so this still failed,** and it was declared the last attempt.

## One more design: the method audits itself (entry 16)

The author then asked to keep going, so that stop rule was lifted, openly, in the ledger.

The idea: a method that says "90%" and delivers 85% can measure that gap by itself and correct for it.
1. Inside the calibration patients, run a small audit: 100 times, set the cut-off on most of them and check it on 30 others, at several strictness levels.
2. Take the lowest strictness level that keeps the budget in 90% of those inner checks.
3. Set the final cut-off at that level.

The outer audit is unchanged, and its 30 patients never take part in the procedure they check.

| | Entry 13 | Entry 16, self-audited |
|---|---|---|
| Splits that kept the budget | 85.3% | **88.7%** |
| Wrong among answered alone, average | 7.0% | 5.8% |
| Asked | 59% | 63% |

It passed the 88% bar, but narrowly, and 88% was a bar set under the project's own 90% standard.

## The system now: a stricter self-audit (entry 17)

One change: the inner audit must keep the budget in 95 of 100 inner checks, not 90. The outer bar was raised to 90%.

| | Entry 16 | **Entry 17, the system now** |
|---|---|---|
| Splits that kept the budget | 88.7% | **93.7%** |
| Wrong among answered alone, average | 5.8% | **4.4%** |
| In 90% of splits | 1.7% to 11.7% | 0% to 10.4% |
| Asked | 63% | 68% |

**It passes clearly.** The price is that it asks a pathologist about two cases in three.

It also reacts to a harder batch without seeing a label. Across the 300 splits, batches that are harder for the classifier get asked about more: rank correlation 0.51 (0.41–0.59). Patient context is what does this.

On all 121 patients the system now uses cut-off 0.7624: it asks 66% and is wrong on 4.2% of what it answers alone.

Limits: this is the fifth design tried on the same 121 patients, and the worst of the 300 batches still had 17% wrong.

## What did not work

- **A cut-off that corrects itself (entry 11).** After each spot-checked answer the cut-off moves up or down. With 10% of answers checked, it got from 36% wrong to 25% when much harder patients arrived. The bar was 13%. It learns too slowly.
- **A second model's opinion, and the margin between the top two types.** Neither ranked the cases better than the top probability.

## The second look (entry 14): contaminated

On the 30 test patients the system of entry 17 is wrong on 9.9% (4.7–15.6%) of what it answers alone, 16 of 161. It was 18.0% with the tested system, 14.8% with entry 13's version and 12.2% with entry 16's. It asks about 72% of their cases, more than the 66% it asks at home, which is the reaction to a harder batch described above. These patients were already used, and knowing they were hard shaped the rebuild. So this is not a test. It is on the page because a reader will ask.

## The outside check (entry 15): another hospital

BACH is a set of 400 breast images from Porto, with four classes. How to read and score it was fixed before any image was opened, and the system was not changed.

| | BACH | Aim |
|---|---|---|
| Wrong among answered alone | **1.3%** (0–3.5%), 2 of 151 | at most 10% |
| Right class on the list, among asked | **91.2%** (87.7–94.6%) | at least 90% |
| Asked | 62.3% | |
| Classifier accuracy, 4 classes | 80.8% (76.8–84.5%) | |

Both aims were met. Three limits keep this honest:
- Four classes are an easier target than seven types: mixing up two benign types does not count as an error on BACH.
- 85 of the 151 lone answers are invasive carcinoma, the easiest class.
- These are the numbers with entry 17's system. With entry 13's system, which was the one fixed before BACH was opened, they were 1.8% (3 of 166) and 91.0%.
- It did not ask more often on the unfamiliar images. So it did not show that it "knows" they are unfamiliar.

Normal tissue is the weak spot there: 68% accuracy, and 8 normal images were called invasive.

## Where the project's question stands

"Does it know when it can't?" Partly.
- Its confidence carries real information: lone answers are far more accurate than the rest, at home and at another hospital.
- For a batch of 30 new patients it keeps "at most 10% wrong among lone answers" in about 94 of 100 batches, at the price of asking about two cases in three. That is a measured record on 121 patients, not a mathematical guarantee.
- It meets a harder batch with more questions, through patient context. It still cannot see a different hospital as "different": on BACH it asked no more than at home.

## Check yourself

1. Why were 49 regions of one patient not 49 pieces of evidence?
2. The method that resamples all calibration patients kept the budget in 73% of splits; the one that draws batches of 30 kept it in 85%. What extra kind of luck does the second one count?
3. Why does the second look at the test patients prove nothing by itself?
4. BACH's result is good. Name one reason not to read it as "the 10% promise holds for 7 types elsewhere".

<details><summary>Answers</summary>

1. They share the patient, the staining and the lesion, so they are right or wrong together.
2. The luck of which 30 patients arrive next. A small batch can be harder than average even when the long-run error is fine.
3. Those patients were already used once, and knowing they were hard shaped the rebuild. A fair test needs patients that played no part.
4. BACH has 4 coarse classes, so errors inside "benign" do not count; and most lone answers there are invasive carcinoma, the easiest class.
</details>

## Words to know

- **Clustered data:** cases that come in groups and are alike inside a group (regions of one patient).
- **Bootstrap:** drawing the data again at random, many times, to see how much a number could vary.
- **Risk-coverage curve, AURC:** answer the most confident cases first and track the error; the area under that curve scores a confidence signal (lower is better).
- **Distribution shift:** new cases differ from the ones the system learned from.
- **External validation:** a check on data from somewhere else, collected by other people.
