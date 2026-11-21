# Step 2 · Look at the data

**Code:** short checks run on 2026-11-13/14; the same header trick is in [`tools/speed_test.py`](../tools/speed_test.py) · **Milestone:** M0

## In one sentence

Before building anything, we counted the images per type, measured their sizes, and chose a zoom level.

## A real example

`data/bracs/BRACS_RoI/latest_version/train/1_PB/BRACS_1325_PB_11.png`
- Type: PB, pathological benign.
- Size: 1,784 × 1,353 pixels at 40×. One pixel is 0.25 µm, so the region is about 0.45 × 0.34 mm.

## The 7 types and how many images each has

| # | Type | Group | Train | Val | Test | Total |
|---|---|---|---|---|---|---|
| 0 | N · Normal | benign | 357 | 46 | 81 | 484 |
| 1 | PB · Pathological benign | benign | 714 | 43 | 79 | 836 |
| 2 | UDH · Usual ductal hyperplasia | benign | 389 | 46 | 82 | 517 |
| 3 | FEA · Flat epithelial atypia | atypical | 624 | 49 | 83 | 756 |
| 4 | ADH · Atypical ductal hyperplasia | atypical | 387 | 41 | 79 | 507 |
| 5 | DCIS · Ductal carcinoma in situ | malignant | 665 | 40 | 85 | 790 |
| 6 | IC · Invasive carcinoma | malignant | 521 | 47 | 81 | 649 |
| | **Total** | | **3,657** | **312** | **570** | **4,539** |

What the table tells us:
- **The test set is balanced** (about 80 per type) but **training is not**: PB has twice as many images as Normal. Step 3 will need to correct for that, for example with class weights.
- **Validation is small** (40 to 49 per type). Promises per type that are built from it will be rough (step 4).
- **151 patients, not 189.** The 4,539 regions come from 387 slides of 151 patients (BRACS paper, page 2). Whether the official split keeps each patient on one side is **M0's next check**.

## Sizes and zoom

- Smallest region: 127 × 137 px. Median: 1,784 × 1,353 px. Largest: 17,611 × 9,842 px (all at 40×).
- BRACS was scanned at **40×, 0.25 µm per pixel**. Halving every side gives **20×, 0.5 µm per pixel**.

*(The comparison image is not published: BRACS images are for non-commercial research only.)*

| | 40× tile | 20× tile |
|---|---|---|
| A 224-px tile covers | 56 µm | 112 µm (4× the area) |
| You see | a few cells, detail inside each nucleus | how the cells are arranged, nuclei still clear |
| Tiles for all of BRACS (full tiles only) | 368,072 | 83,585 |

**Proposal: 20×.**
1. H-optimus-0's model card says it expects 0.5 µm per pixel, and UNI was trained at 20×.
2. The look-alikes differ in how cells are arranged, and a 20× tile shows more of that.
3. It means 4.4× fewer tiles.

The cost: less detail inside each nucleus, and 107 regions are smaller than one 20× tile, so they get padded with white.

## The code: reading an image's size without opening it

Opening 4,539 images of about 12 MB each would take minutes. A PNG file stores its width and height near the very start:
- an 8-byte signature,
- then 4 bytes of chunk length and 4 bytes saying "IHDR",
- then width and height, 4 bytes each.

So bytes 16 to 23 are enough:

```python
import struct
with open(path, "rb") as f:
    width, height = struct.unpack(">II", f.read(24)[16:24])   # ">II" = two big-endian 4-byte integers
```

Reading 24 bytes per file takes under a second for all 4,539.

## Rebuild it yourself

1. Use the header trick on 3 images of your choice and print their sizes.
2. Check that the trick is right: open the same images with Pillow (`Image.open(p).size`) and compare.
3. Count the training images per type yourself from the folder names, and compare with the table above.

## Check yourself

1. How many micrometres does a 224-px tile cover at 20×?
2. Why did we read only 24 bytes of each file?
3. Which type has the fewest training images, and why does that matter?

<details><summary>Answers</summary>

1. 224 × 0.5 µm = 112 µm.
2. The width and height sit in bytes 16 to 23. Reading the whole image would take much longer and tell us nothing more.
3. Normal, with 357. A model can lean toward types it saw more often, so training must correct for that, for example with class weights.
</details>

## Words to know

- **Region of interest (RoI):** a part of a slide cut out around a lesion.
- **Magnification (20×, 40×):** zoom. The useful number is µm per pixel.
- **Class balance:** whether every type has a similar number of examples.
- **Patient split:** train, validation and test hold different patients, so the test measures new patients.
