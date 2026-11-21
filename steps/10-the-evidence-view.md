# Step 10 · The evidence view

**Code:** [`tools/evidence_why.py`](../tools/evidence_why.py), [`tools/evidence_tiles.py`](../tools/evidence_tiles.py), [`tools/evidence_lookalikes.py`](../tools/evidence_lookalikes.py), [`tools/evidence_patterns_keep.py`](../tools/evidence_patterns_keep.py), [`tools/build_viewer.py`](../tools/build_viewer.py) · **Ran:** 2026-11-18 · **Part:** 3 · **Ledger:** entries 18 to 22

## In one sentence

When the system asks a pathologist, it now shows why, in four ways, and each way had to pass a test fixed in advance before it was allowed on screen.

**Open it:** `outputs/viewer/index.html`. It holds 56 real regions from the 30 test patients, 8 per type drawn at random, so failures appear as often as they happen.

## A real example

`BRACS_1284_DCIS_10` (in the viewer: `index.html#40`). The agreed label is DCIS.
- The classifier says ADH with 86%. The system asks, and its list is UDH and ADH. **DCIS is not on the list: this is one of its misses.**
- The named patterns say: "comedo necrosis, higher than 91% of regions". Comedo necrosis is a DCIS pattern.

One case proves nothing. But it shows what evidence is for: a pathologist sees a hint that the list did not carry.

## The four kinds of evidence

### 1. Why A and not B (entry 19)

The classifier is one linear layer. So its preference for type A over type B is a plain sum, and the sum can be taken apart:

**tiles + whole-region view + starting point = the classifier's score for A minus its score for B**

- Each tile gets its own share, drawn as a coloured square on the image.
- The whole-region view gets one number. It has no place on a map.
- The starting point is the classifier's built-in lean before it sees anything.

**Test (a), exactness:** the parts add up to the classifier's own score. Largest gap: 0.00002.

**Test (b), pointing:** on patients the classifier never saw, does one tile's share alone tell A-regions from B-regions? Yes for all 21 pairs. Weakest: ADH against DCIS (AUC 0.74) and UDH against ADH (0.75). Easiest: normal against invasive (0.998).

**What to remember:** the tiles carry only 43% to 72% of a decision. The rest is the whole-region view. The viewer shows both, so the map is never read as the whole story.

### 2. Structures like these (entry 21)

For one tile, the viewer shows the most similar tiles from confirmed cases of other patients. They are compared in UNI's own features, which the classifier has not shaped.

**Test:** of a tile's 5 nearest tiles, 61% come from regions of the same type. A random tile: 19%.

Where the classifier is wrong, the neighbours still point to the agreed type in 26% of cases, against 12% by chance. So they are a weak witness that is partly independent of the classifier.

**Limit:** for UDH (16%) and ADH (12%) they are barely better than chance.

### 3. Whole regions the classifier scores alike (entry 18)

For each type on the short list, the three nearest confirmed cases.

**Test:** among asked cases, 53% of the five nearest share the agreed type (random: 15%). But using them to order the list changed its accuracy by 0.0 points.

**So they are shown as illustration only.** They restate what the classifier already believes. The viewer says in words that they are not a second opinion.

### 4. Named patterns (entries 20 and 22)

A model that links images and text scores tiles against a pathologist's words. Twelve patterns were fixed in advance, each with the types it should be highest in.

| Model | Patterns that passed | Verdict |
|---|---|---|
| QuiltNet | 3 of 12 | failed (the bar was 6) |
| **KEEP** (MIT licence) | **8 of 12** | passed |

Passed with KEEP: invasive carcinoma infiltrating the stroma (0.94), columnar cell change (0.89), comedo necrosis (0.89), normal lobular unit (0.79), fibroadenoma (0.78), desmoplastic stroma (0.75), sclerosing adenosis (0.69), usual ductal hyperplasia (0.65).

Failed: cribriform architecture, micropapillary architecture, atypical ductal hyperplasia, apocrine metaplasia.

**Limit:** the four that failed are the ones a pathologist uses to tell ADH from DCIS. Named patterns describe a case; they do not settle the hardest pair.

## What went wrong on the way

- **Region look-alikes were meant to be a second opinion. They are not.** The test showed they add nothing to the classifier, so their role was cut down to illustration.
- **The first text-image model failed.** One fair retry with a stronger one was registered before it ran.
- **KEEP's own loader crashed** on the installed library version. The model was rebuilt from its published definition, with a check that every weight is matched, before any pattern was scored.

## What the viewer does not claim

- The squares mark tiles that drove the score. They do not mark where a lesion is.
- Nothing in it is three-dimensional. A slide is a flat slice, shown flat, with a 100 µm scale bar.
- The agreed label is hidden until you ask for it, so you can decide first.
- Images stay on this machine. Before anything is published: credit BRACS and check its download terms.

## Check yourself

1. Why can the tile shares be called exact and not an estimate?
2. The tile map for ADH against DCIS passed its test with an AUC of 0.74. What does that number ask you to keep in mind?
3. Region look-alikes share the agreed type 53% of the time. Why are they still "not a second opinion"?
4. Tile look-alikes are compared in UNI's features, region look-alikes in the classifier's scores. Why does that make the tile ones more independent?
5. Named patterns passed 8 of 12. Why is that less help for the gray zone than it sounds?

<details><summary>Answers</summary>

1. The classifier is a sum of weights times numbers. Splitting a sum into its parts changes nothing, and the parts add back to the same score.
2. A single tile is only a moderate witness for that pair. Read many tiles together, and remember the whole-region view carries part of the decision.
3. They are found with the classifier's own scores, so they mostly repeat its belief. Adding them did not make the list's order more accurate.
4. UNI was trained without our labels and without our classifier. Its notion of "alike" can disagree with the classifier, and where the classifier is wrong it still points right twice as often as chance.
5. The four patterns that failed (cribriform, micropapillary, ADH, apocrine) include the ones that separate ADH from DCIS, the hardest pair.
</details>

## Words to know

- **Linear layer:** weights times numbers, added up. Because it is a sum, it can be split exactly.
- **AUC:** how well a score separates two groups. 1.0 is perfect, 0.5 is no information.
- **Precision at 5:** of the 5 nearest cases, the share that are of the right type.
- **Zero-shot:** scoring images against words the model was never trained on for this task.
- **Text-image model:** a model that puts pictures and sentences in one space, so they can be compared.
