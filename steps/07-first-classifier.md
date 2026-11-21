# Step 7 · The first classifier

**Code:** [`tools/train_classifier.py`](../tools/train_classifier.py) · **Ran:** 2026-11-18 · **Milestone:** M1 · **Ledger:** entries 3 and 4

## In one sentence

We trained one small classifier per feature model and measured which model's numbers sort the 7 tissue types best.

## A real example

`val/4_ADH/…` regions, scored by the UNI classifier: of the 41 validation regions labelled ADH, the classifier called 9 ADH, 13 UDH, 7 DCIS, 5 FEA, 4 PB and 3 Normal. ADH sits between UDH and DCIS, and that is exactly where the classifier loses it.

## How it works

1. **One row per region.** A region has many tiles, each with about 1,000 numbers. We average them, so every region becomes one row of numbers.
2. **One linear layer.** The classifier multiplies those numbers by weights and adds them up, once per type. The highest sum wins. (Technical term: multinomial logistic regression, also called a linear probe.)
3. **Class weights.** Training has twice as many PB regions as Normal ones, so rare types count more.
4. **Two settings are chosen:** how strongly the weights are pulled towards zero (C), and whether tiles that are mostly white are dropped.

## How the settings were chosen without touching validation

**Cross-validation, grouped by patient.** The 106 training patients are split into 5 parts. Train on 4, score on the 5th, and rotate. A patient is never on both sides.

Why not choose on validation? Validation is needed later to set the promise of the short lists (M2). If it also chose the settings, that promise would be too optimistic.

## The result

| Model | Kind | Cross-validation (106 patients) | Validation (15 patients) |
|---|---|---|---|
| UNI | tissue | 0.700 (0.639–0.757) | 0.672 (0.568–0.805) |
| H-optimus-0 | tissue | 0.689 (0.631–0.746) | 0.665 (0.550–0.805) |
| ImageNet ViT-L | general | 0.621 (0.557–0.683) | 0.629 (0.513–0.713) |
| ResNet-50 | general | 0.568 (0.498–0.635) | 0.527 (0.444–0.608) |

The measure is weighted F1. The ranges come from resampling whole patients.

- **Tissue models win.** In cross-validation UNI beats ViT-L by 0.079 (range 0.045–0.119), and the range excludes zero. That was the rule fixed in advance.
- **Validation agrees on the order but cannot prove it.** With 15 patients the ranges overlap.
- **UNI and H-optimus-0 are tied.**

## Where it fails (UNI, validation, F1 per type)

| N | PB | UDH | FEA | ADH | DCIS | IC |
|---|---|---|---|---|---|---|
| 0.75 | 0.51 | 0.55 | 0.76 | **0.32** | 0.86 | 0.92 |

The clear ends (invasive, DCIS) are easy. ADH is the hardest by far, as in published work (HACT-Net reports 40.4 for ADH on the test set).

## What went wrong on the way

- **The first grid was too narrow.** Every model's best C was at the edge of the grid. We did not score validation, logged a new entry with a wider grid, and reran. The best C then sat inside the grid.
- **A save bug** crashed the first validation run before it printed any score. It was fixed and rerun.

## What this does not say yet

- Nothing about the **test set**. It is locked.
- It is **not comparable** with published numbers yet: those are test-set scores.

## Check yourself

1. Why do we average a region's tiles before the classifier?
2. Why were the settings chosen by cross-validation on training patients, and not on validation?
3. The cross-validation range for UNI − ViT-L excludes zero, but the validation ranges overlap. Which do you trust for "tissue models win", and why?

<details><summary>Answers</summary>

1. The classifier needs one row per region, and regions have different numbers of tiles.
2. Validation sets the promise of the short lists later. Data used to choose settings would make that promise too optimistic.
3. Cross-validation: it uses 106 patients, validation only 15. Validation agrees on the order, which is reassuring, but it is too small to prove it.
</details>

## Words to know

- **Linear probe:** one linear layer trained on frozen features.
- **Cross-validation:** rotate which part of the training data is held back for scoring.
- **Weighted F1:** a score between 0 and 1 that balances missed cases and false alarms, averaged over the types by their size.
- **Regularisation (C):** a pull that keeps the weights small, so the classifier does not memorise.
