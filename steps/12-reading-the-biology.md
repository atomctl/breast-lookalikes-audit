# Step 12 · Reading the biology (gate 1 run on 19 Nov 2026: WEAK)

**Date:** 19 Nov 2026 · **Ledger:** entry 48 · **Idea:** the author's · **Code:** [`tools/stain_xenium.py`](../tools/stain_xenium.py)

## Gate 1 result (entry 48)

**Question:** on tissue where the stain's genes were measured cell by cell, can the H&E image show which duct cells carry them, on a patient the model never saw?

| Held-out patient | Labelled cells (positive) | AUC | Position only, no image | Image minus position | Local test (a positive cell against its negative neighbours) |
|---|---|---|---|---|---|
| 1, counts | 5,527 (656) | 0.784 | 0.660 | +0.124 (0.051 to 0.193) | 0.639 (0.596 to 0.684) |
| 2, counts | 7,912 (264) | 0.852 | 0.799 | +0.052 (−0.001 to 0.097) | 0.533 (0.471 to 0.607) |
| 4, too few cells to count | 967 (127) | 0.692 | 0.599 | +0.093 (0.025 to 0.153) | 0.562 (0.477 to 0.637) |
| 3, too few cells to count | 127 (26) | 0.886 | 0.902 | −0.015 | 0.493 |

**Verdict by the rule: WEAK.** Not a pass and not a stop. Both counting patients clear the AUC bar (0.75). The image beats position clearly in patient 1 and misses by a hair in patient 2. The local test holds in patient 1 only, so the rule's wording is: it sees the duct, not the cell.

**In plain words:** most of what the model knows is where a cell sits in the duct. Position alone gives 0.66 to 0.90. The image adds a little on top (0.05 to 0.12). In one patient of two it can tell a positive cell from its neighbours 64 times in 100: real, and small.

**What was different from the plan:** only 4 patients exist (the public list held one twice), and only 2 have enough positive duct cells to count. Patient 3 is a lobular cancer with almost no ducts. The rule was amended for two patients before any result. KRT15 was dropped because only one panel holds it.

**Not tried after the result:** a wider range of model strength (it sat at the edge of its grid), finer tokens than 8 µm, a better alignment than one affine map, more patients.

Disk: the four samples take 15 GB in `data/xenium/`; 12 GB are free.

## Gate 3 result (entry 49), run without gate 2

**Question:** the stain model has never seen a BRACS image or label. Applied inside the ducts of the gray-zone regions, is its predicted stain higher in UDH than in ADH?

| Type | Median score (training patients) | Patches on the positive side |
|---|---|---|
| UDH | +0.74 | 83% |
| ADH | −0.16 | 41% |
| FEA | −0.21 | 41% |
| DCIS | −0.57 | 23% |

| Test | Result | By the rule |
|---|---|---|
| 1. UDH above ADH | 0.79 (0.69 to 0.88) | Confirmed |
| 1b. Is it only duct size? | Size gives 0.57 and 0.65; the score is clearly above both | It is not size |
| 2. Where the classifier is wrong (146 regions) | 0.44 | No witness |
| 3. Joined with the classifier | −0.002 | Not adopted |
| 4. ADH of ADH-only slides against ADH beside carcinoma | 0.72 (0.57 to 0.88); medians +0.33 and −0.30 | Reported |

**In plain words:** a model taught only by measured genes puts the four types in the order the biology gives, on another hospital's images. That is the first positive result of this line. But it is wrong on the same regions as the classifier, so it does not repair the classifier. To this model too, the disputed ADH regions look like UDH.

**Limits:** the 15 validation patients do not confirm test 1 (0.60). Gate 1 showed the model reads the kind of duct, not the single cell. Nothing here says whether those ADH regions are mislabelled or whether the H&E image does not show what makes them ADH. A real CK5/6 stain on such cases is the only check.

## Gate 1 repair (entries 51 and 52): STOP for the single cell

**Question:** gate 1 said "it sees the duct, not the cell". Was that the tissue, or three faults of our method?

| Fault named in entry 48 | Test | Result |
|---|---|---|
| Cell places off the image | Entry 51: shift per 256 µm block | 0.71 µm in patients 1 and 2 (the floor of the measurement). Not a fault |
| A token carries its whole tile | Entry 52: one 56 µm picture per cell, nucleus in the middle, 0.25 µm per pixel | Worse: AUC 0.745 and 0.644 (before 0.784 and 0.852); local test 0.597 and 0.528 |
| Strength at the edge of its grid | Entry 52: wider grid | The same strength is chosen. Nothing changes |

**Verdict by the rule: STOP** (patient 2 is under 0.65), and "it sees the cell" is not reached. The cell-by-cell reading with UNI ends on these samples. The plan to measure the mosaic inside a duct ends with it. The duct-level score of gate 3 stays.

**In plain words:** the sharp picture of one cell tells less than the coarse token with its wider surroundings. The signal is in the kind of duct and the place in it.

**Not tested:** the pictures of patient 1 are soft and those of patient 2 are sharp, so training and test pictures differ. That can explain part of the lower AUC. It does not explain the local test, which is 0.53 in patient 2 with every description.

---

The plan as written before the run follows.

This page answers one prompt: go deeper on the gray zone, not wider. Every source below was opened on 19 Nov 2026; what was only read as a search summary is marked.

## 1. What ADH is, and the reference that replaces the BRACS label

**Biology.** UDH is a mix of cell types (polyclonal). ADH and low-grade DCIS are one clone of luminal cells. Pathologists make this visible with stains: CK5/6 is positive as a mosaic inside UDH and negative inside ADH and DCIS; ER is patchy in UDH and strong and even in ADH. ADH and DCIS then differ only by extent (about 2 mm or two ducts). Sources: CAP Today, PathologyOutlines, a Modern Pathology study in which a stain panel raised agreement between pathologists (search summaries, not full texts).

**Why not the BRACS label.** Pati et al. (HACT-Net, arXiv 2102.11057, table VIII, read in the PDF) gave the BRACS test regions to three other breast pathologists, each with over 20 years of practice, who saw only the regions:

| | UDH | ADH | FEA | DCIS | Weighted F1 |
|---|---|---|---|---|---|
| Pathologist 1 | 41.9 | 36.0 | 19.1 | 71.6 | 55.3 |
| Pathologist 2 | 25.0 | 35.4 | 65.2 | 68.0 | 57.1 |
| Pathologist 3 | 49.4 | 42.3 | 54.1 | 65.2 | 56.7 |
| HACT-Net | 43.6 | 40.4 | 74.2 | 66.4 | 61.5 |

Two pathologists gave the same answer in 48% to 64% of regions. Our classifier scored a weighted F1 of 64.6 on today's test regions (entry 9; the test set of that paper had 626 regions, today's has 570, so the comparison is rough).

So on cut-out regions, experts reproduce the BRACS label of ADH with an F1 near 0.38. A classifier at 0.55 is not below the ceiling; it is above what one expert reaches. More agreement with this label is not more truth. Earlier I wrote that label doubt "cannot be checked on BRACS". That was wrong: it was measured in 2021.

**The choice.**
- **Now: the stain.** It is a measurement, it is what pathologists themselves use to settle UDH against ADH, and public data holds it in two forms (below).
- **Later, with a hospital: the surgery result.** It is the reference for the real question (who can skip surgery). No public dataset has it.
- **Not available: the measured clone** (gene copy changes per lesion).

**Limits of the stain as a reference.** It cannot separate ADH from DCIS (both negative). Columnar cell lesions and FEA are also CK5/6-negative and ER-positive. A minority of lesions break the rule. Pathologists read the stain together with the shape, never alone.

## 2. The raw material, after opening it

| Source | What it holds | State |
|---|---|---|
| 10x Xenium breast, sample 1 | 167,780 cells with counts of 541 genes, the position of every transcript (0.67 GB), and an H&E image of the same section (1.4 GB). Has KRT5, KRT6B, KRT14, KRT15, ESR1, PGR, KRT8, KRT7, EPCAM and myoepithelial genes. 8,605 inner epithelial cells carry at least 2 basal-keratin transcripts. | Cell table on disk; image and transcripts not yet. Licence CC BY, no login |
| The other Xenium breast samples | 8 more samples; 5 patients in all, every one a cancer case (HEST list v1.1.0) | Not downloaded |
| HISTAI-mixed | **Real CK5/6-stained whole slides.** 151 cases have one; 38 are breast; 22 breast cases have both an H&E slide and a CK5/6 slide (7 also an ER slide); 32 GB. 16 of the 22 are invasive cancers; the ducts around a cancer still show the stain's pattern. Not known yet: whether the two slides are neighbouring cuts of the same block. | **Access not granted** (manual gate on Hugging Face) |
| HISTAI reports | 141 breast reports state a CK5 result in words, 37 of them for non-invasive lesions | On disk |
| HISTAI-breast | 1,925 H&E slides, no stains | Access granted |
| The three pathologists' own answers on BRACS | Exist (Pati et al.), not published as a file | Would need an email to the authors |
| BRACS whole slides | 547 slides; features of 85 are saved; the slides themselves are no longer on disk | Download again if needed |

Disk: 24 GB free. Gate 1 needs about 10 GB. HISTAI must be processed a few cases at a time.

## 3. Methods and tools, step by step

| Step | Method | Tool |
|---|---|---|
| Assign transcripts to cells without leaks from neighbours | Count only transcripts that lie inside the nucleus outline | The `overlaps_nucleus` field of the Xenium transcript file; Proseg or Baysor as a second opinion |
| Put gene counts on the image | The affine matrix 10x publishes for each H&E image | `tifffile`, NumPy |
| Describe the tissue | UNI patch tokens, one per 8 micrometres, as in the duct instrument | The project's `duct_tokens.py` |
| Predict the stain | A small classifier on the tokens; target: an inner duct cell with basal keratins | scikit-learn; a small network only if the linear model shows signal |
| Keep the model honest | Train on some patients, test on a patient never seen; a baseline that knows only the distance to the duct rim | The project's fold and bootstrap code |
| Read the real stain | Separate the brown stain by colour, per duct | Colour deconvolution (scikit-image), QuPath for a check by eye |
| Align a stained slide with its H&E slide | Non-rigid registration of serial sections | VALIS |
| Compare with experts | Agreement models for several readers | Dawid-Skene; iMRMC for a reader study |

## 4. Outside checks, each with its rule fixed before the result

1. **Measured genes on an unseen patient** (gate 1). A measurement, not an opinion.
2. **A real antibody stain from another hospital** (gate 2). Different patients, scanner and chemistry.
3. **Published expert readings** (Pati et al.): the yardstick is what one expert reaches (ADH about 0.38), not 100%.
4. **A pathologist** who looks at 20 ducts where prediction and stain disagree, and later real stains on gray-zone cases (Magdeburg).
5. **The Granada group's released tests**, when they publish them.

## 5. The plan: four gates

| Gate | Question | Rule (to be fixed in the ledger) | Who | My honest chance | Stop if |
|---|---|---|---|---|---|
| 1 | Can the H&E image show which inner duct cells carry basal keratins, on a patient never seen? | AUC of at least 0.75 on inner duct patches in at least 3 of 5 held-out patients, and clearly above the distance-to-rim baseline | I run it alone; about 10 GB and one to two days | About 1 in 2 | Under 0.65: the image does not hold it even where the truth is measured. The idea ends, and that is a finding |
| 2 | Does the prediction agree with real CK5/6 stains from another hospital? | Per duct, predicted share against stained share: rank correlation of at least 0.5, if the slides can be aligned | You request access to HISTAI-mixed (one form). Then I run it. A pathologist checks 20 ducts | About 4 in 10, given gate 1 | No agreement: the model learned the Xenium tissue, not the biology |
| 3 | In the BRACS gray zone, is the predicted mosaic higher in UDH than in ADH? Also for the ADH of ADH-only slides (entry 47), and where the classifier is wrong | AUC of at least 0.70 with the by-patient range above 0.5 | I run it alone; no new data | About 1 in 3, given gates 1 and 2 | At chance: the stain's signal does not reach these regions |
| 4 | Do real stains on disputed cases agree, and does it help a pathologist? | Reader kit with the predicted stain shown; real CK5/6 on a few cases | Magdeburg pathologist; an email to the BRACS authors for the three experts' answers | Not estimated | — |

**The whole chain:** about 1 chance in 10 to 15 that gates 1 to 3 all pass. Each gate still ends in a result that can be written down, because each has a measured truth or a rule. Gate 3 can run without gate 2 if access is refused; it is then weaker.

**What nobody has published, as far as five searches on 19 Nov show:** a stain or gene prediction for UDH against ADH. One 2026 model (mSTAR) is reported to predict CK5 from H&E for cancer subtypes; not read yet. A 2026 systematic review of AI for borderline breast lesions exists (Research Square); not read yet. Both must be read before any claim of "first".

## Known risks

1. All training tissue comes from cancer patients; none is labelled UDH.
2. The H&E images were made after the Xenium run, on another scanner than BRACS.
3. A stain panel raised agreement between pathologists in a published study. That suggests the H&E image alone does not hold everything the stain shows.
4. Four tools have already failed on these regions.

## Techniques to copy from NVIDIA's "Agent for Life Sciences" (read 19 Nov 2026)

Source: the blueprint page and the design documents in `github.com/NVIDIA-BioNeMo/agent-for-lifesciences` (execution flow, tool chaining, view injection, evals, training pipeline). NVIDIA has no model for tissue images; what can be copied is how the work is organised.

| Their technique | What it is there | Use here |
|---|---|---|
| Tasks with a planted defect | They damage a molecule in a known way and check whether the agent names the damage | Build cases whose answer is known by construction: Xenium ducts where the stain's genes were measured, and simulated ducts grown from known rules (the author's simulation idea). The system must recover the known answer before it is trusted on BRACS |
| A verifier and a gate | Each task has an expected outcome; a script turns the scores into pass or fail | Turn each ledger rule into a task a script can rerun, so one command checks every claim |
| "To validate, do this" | The answer states its confidence and the experiment that would confirm it | A third action next to "answer" and "ask a pathologist": **"order the CK5/6 stain"**, when the predicted stain is unsure |
| Typed results and a trace | Each tool returns a fixed kind of result; every call is recorded | Each sentence of a report points to the tool output behind it |
| The viewer knows what is shown | The agent is told what the user sees | For the demo: the report and the evidence view stay in step |

Not copied: their training of a model by reinforcement learning (it needs a GPU cluster), and their full software stack (Linux, Docker, 50 GB).

Order: the agent is only as good as its tools, and the tool that adds new information (the stain predictor) does not exist yet. So gate 1 comes first, built as a tool with a fixed kind of result; the planted-answer tasks become part of how gates 1 and 3 are checked.

## Later, if gate 3 gives a signal

- The author's simulation idea: grow a duct from cell rules (one clone or mixed cells) and fit the rules to the image.
- Gene copy changes predicted from the image, the direct mark of a clone.
