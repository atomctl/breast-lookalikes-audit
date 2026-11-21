# Step 0 · Project log: where we are

*Last updated: 2026-11-18 (parts 0 to 3 finished: classifier, promise, second hospital, evidence view). Every step the AI takes gets a note in this folder; this page is the map.*

## The project in one sentence

A system that sorts breast tissue images into the 7 BRACS types, and says "send this to a pathologist" when it is not sure.

## The 5 steps of the system

| # | Step | Milestone | Status |
|---|---|---|---|
| 1 | Get the data, look at it, split it by patient | M0 | **done** (data card: step 6) |
| 2 | Turn every image into numbers with a frozen, borrowed model | M1 | **done for all 4 models** (ResNet-50, ImageNet ViT-L, UNI, H-optimus-0) |
| 3 | Train a small classifier on those numbers; compare tissue-trained vs. general models | M1 | **done** (steps 7 and 8): tissue models win; final classifier 0.646 on the test set; THUNDER's published result reproduced |
| 4 | Short lists with a promised hit rate, the routing rule, the results page | M2 | **done** (step 8): two-step system and results page built; **on the test set the error promise was not kept (18% against 10%)** and the list promise was missed by one point (89.1%) |
| 5 | Public demo | M3 | a local case viewer exists (step 10); the public site is not started |

Version 2, after all of that: the agent, the second-hospital test, and the pathologist's review.

## The notes so far

1. [Step 1 · Get the data](01-get-the-data.md): how BRACS was downloaded and checked (`tools/bracs_download.py`).
2. [Step 2 · Look at the data](02-look-at-the-data.md): the 7 types, image sizes, and zoom (20× vs. 40×).
3. [Step 3 · Mac or cloud? The speed test](03-mac-or-cloud-speed-test.md): why we compute on your Mac (`tools/speed_test.py`).
4. [Step 4 · Images into numbers](04-images-into-numbers.md): how each tile becomes a list of numbers (`tools/extract_features.py`).
5. [Step 5 · One patient, one group](05-patient-split.md): the patient split check and the region table every later step reads (`tools/patient_split.py`).
6. [Step 6 · The data card](06-data-card.md): what the data is, its licence, its quirks and its limits.
7. [Step 7 · The first classifier](07-first-classifier.md): one linear layer on each model's features, and which model wins (`tools/train_classifier.py`).
8. [Step 8 · Act or ask, and the short list](08-act-or-ask-and-short-lists.md): the better classifier, the two-step system, the checks, and the one test pass (`tools/short_lists.py`, `tools/final_test.py`).

9. [Step 9 · Why the promise broke, and the outside check](09-why-the-promise-broke.md): the cause, the rebuilt "answer alone" step, the second hospital (`tools/patient_promise.py`, `tools/second_hospital.py`).

10. [Step 10 · The evidence view](10-the-evidence-view.md): the four kinds of evidence, their tests, and the case viewer (`tools/build_viewer.py`).
11. [Step 11 · The gray zone](11-the-gray-zone.md): what was tried on UDH, FEA, ADH and DCIS, the guard that passed, the duct instrument, and the reader-study kit (`outputs/reader_study/index.html`).

**The case viewer:** `outputs/viewer/index.html` (56 real regions, the decision and the evidence for each).

**The results page:** `outputs/results/index.html`, rebuilt by `tools/build_results_page.py`.

Every experiment is logged in [`LEDGER.md`](../LEDGER.md) with its rule fixed before the result.

Each step from 4 on has a practice notebook in `learn/` (predict, modify, rebuild). `learn/README.md` explains the loop.

## Next

1. **You:**
   - Finish `learn/04a-an-image-is-numbers.ipynb`.
   - Then solve Deep-ML #265 and #239 (see `learn/deep-ml-map.md`).
   - Then do Step 4's notebook.
   - Then Step 5: Deep-ML #1058 first, then `learn/05-patient-split.ipynb`. Keep the "Result" box closed until you've predicted.
2. **The AI:** M1 and M2 are finished and the test set has been used once. After the test (step 9): the cause of the failed promise was found and step 1 was rebuilt around patients. The system now keeps "at most 10% wrong among lone answers" in 93.7% of 300 batches of 30 patients, asking about two cases in three. On a second hospital (BACH, 4 classes): 1.3% wrong among lone answers, 91% list hit rate. Part 3, the evidence view, is finished too (step 10): an exact "why A and not B" tile map, tile look-alikes, region look-alikes as illustration, and 8 named patterns from the open KEEP model. Next, on your go-ahead: part 5, the public demo site and the audit trail, or first a pathologist's reaction to the viewer.

## Decisions waiting for you

These are the AI's proposals. Each becomes a `DECISIONS.md` entry in **your own words**: the options, what was measured, and why you chose one.

| Decision | Proposal | Measured evidence | Note |
|---|---|---|---|
| Where to compute | your Mac, no cloud | speed test, step 3 | 3 |
| Zoom | 20× | 4.4× fewer tiles; the models expect it | 2 |
| Which tiles | save all tiles, including edge tiles, then choose later | 120,628 tiles with edges vs. 83,585 without | 4 |
| Which models | ResNet-50 + ImageNet ViT-L (general), UNI + H-optimus-0 (tissue) | same-size pair for a fair test | 4 |
| Which split | keep the official BRACS split | the patient check and the random-split comparison | 5 |

## Parked ideas

Ideas for later milestones. They are kept here so they don't pull focus now.

- **Demo layout (M3), from an NVIDIA demo seen on 2026-11-17.** The answer sits on the left: the short list and the routing decision (in version 2, the agent's report). The region sits on the right in a zoomable viewer, with the tile map on top. The brief already plans the short list, the routing decision and the tile map; the new part is the side-by-side layout.
  - Label the map "tiles that drove the score", never "tumour here". The brief lists heatmaps presented as tumour detection as an anti-goal.
  - If the M1 classifier is one linear layer on the average of the tile features, the region's score is exactly the average of the tiles' scores. The map then shows the model's own arithmetic, not an estimate. Worth weighing when M1 picks its classifier.
- **NVIDIA catalog check (2026-11-17): nothing for pathology.** NVIDIA publishes no pathology model, blueprint or skill; its medical models cover CT, MRI, X-ray, ultrasound and surgery. Four references are worth keeping:
  - **The demo seen on 2026-11-17** is NVIDIA's open-source "Agent for Life Sciences" (github.com/NVIDIA-BioNeMo/agent-for-lifesciences, Apache-2.0): a chat next to a Mol* 3D viewer, with a React front end and a Python back end. Read it as a reference for the M3 layout and the version 2 agent. We borrow ideas, not the app.
  - **MONAI** (github.com/Project-MONAI) has a pathology multiple-instance-learning tutorial and a ready head, `MILModel`. It is the main alternative to "average the tiles" when M1 picks its classifier. Plain attention pooling keeps exact tile shares (it is a weighted average); its transformer variants mix tiles together and lose them.
  - **NVIDIA's Agent Skills** (github.com/NVIDIA/skills, Apache-2.0) use the same SKILL.md format as Claude Code. None fits this project. Their convention of shipping evals with each skill is worth copying if version 2 gets its own skill.
  - **The zoomable viewer:** OpenSeadragon (BSD-3). BRACS regions are small enough that an image with a tile overlay in Gradio may be all the demo needs.
- **Demo idea "Two ways of seeing" (M3), judged on 2026-11-17. Not built, not decided.** Each tile is drawn as real tissue and laid out twice: by the model's features and by nuclei measured from the pixels. The tiles move between the two maps.
  - **Licence:** the BRACS site states CC BY-NC 4.0 (https://www.bracs.icar.cnr.it/?p=3037), so tiles in a non-commercial demo are very likely allowed, with credit (authors, paper, licence link, "cropped"). Not checked: terms shown on the login-gated download page.
  - **Prior art:** a web search (7 searches, rival repos not opened) found no public demo that does this. Reference for the comparison: the PLUTO paper (https://arxiv.org/html/2407.10785v2).
  - **Nuclei tool:** StarDist (BSD-3, runs on Apple Silicon). Its H&E model was probably trained at 40×, so 20× tiles need an upscaling test. CellViT has a Commons Clause and expects CUDA: avoid.
  - **Weak spots:** a tile's nuclei cannot show how far a lesion extends, which is part of ADH vs. DCIS; the brief asks for a "simple public demo", so this is an extra layer and the author's call.
  - **Kill test, run on 2026-11-18 (`tools/kill_test_nuclei.py`, results in `outputs/kill_test/`): passed the rule, but weakly.** 3,729 tiles from 304 validation regions of 15 patients; neighbours = the 10 nearest tiles from other patients. Nuclei measurements and UNI share 2.3% of neighbours (range 1.8–2.8%; chance 0.3%). For scale: UNI and H-optimus-0 share 28.7%, UNI and ResNet-50 4.7%. Neighbours of the same tissue type: nuclei 17%, UNI 37%, chance 12%. So the 11 nuclei measurements carry a little tissue-type signal, and the two maps would look almost unrelated. The rule (kill under 2× chance, or at or above the UNI–H-optimus-0 overlap) was set before the result and turned out lenient.
  - **Second kill test, 2026-11-18 (`tools/kill_test_nuclei.py models`): a UNI vs. H-optimus-0 morph is killed.** Same 3,729 tiles. The two models differ for real: they share 40% of 50 neighbours, against 57% for one half of UNI's numbers vs. the other half (ResNet-50: 11%). But where they disagree has nothing to do with how ambiguous the tissue is: rank correlation 0.01 (by-patient range −0.11 to 0.15). So the moving tiles would tell no story.
  - **What the same numbers support:** one map is enough. In UNI's map, ADH and UDH tiles have the fewest same-type neighbours (21% and 20% of 50; FEA 44%, invasive 40%). The look-alikes mix on the map itself. That backs the earlier plan: one mosaic map, the fog from the short lists (M2), and a tour through the mixed areas.

## Who does what

- **The AI:** writes and runs the code; writes a note like these for every step, with a real example; gives you options with numbers.
- **You:** read the notes with the DeepTutor tutor; make the choices; write `DECISIONS.md` in your own words; check that the results make sense.

## How to use these notes in DeepTutor

- **Read:** Learning → Immersive Reading → collection "Cancer or look-alike · project log". Open a step and ask the tutor in the side panel.
- **Ask about code:** the `tools/` folder (all the code) and this `steps/` folder are linked to the knowledge base "Cancer or look-alike guides". In chat you can ask: "Explain `region_tiles` in extract_features.py line by line" or "Quiz me on step 4".
- **Rebuild:** each note ends with a small exercise. Try it before looking at the script.
- **Two Mastery Paths:** "Build it by hand" is the how: notebooks and Deep-ML problems, in build order. "Project decisions" is the why: concepts and decisions per milestone.
