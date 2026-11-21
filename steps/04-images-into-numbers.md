# Step 4 · Images into numbers (feature extraction)

**Code:** [`tools/extract_features.py`](../tools/extract_features.py), [`tools/run_features_overnight.sh`](../tools/run_features_overnight.sh) · **Started:** 2026-11-14 01:17 · **Milestone:** M1

## In one sentence

Each region is cut into tiles. A **frozen** model reads each tile and writes out a list of numbers, its *features*. Nothing learns in this step; learning comes in step 3 of the system (the small classifier).

## A real example

`BRACS_1325_PB_11.png` (type PB):
1. 1,784 × 1,353 px at 40× → halved to 892 × 677 px at 20×.
2. Cut into **16 tiles** of 224 × 224 px: 4 across × 4 down, including edge tiles.
3. ResNet-50 turned each tile into **2,048 numbers**, so the region is now a table of 16 rows × 2,048 columns. The first 5 numbers of two rows:

```
tile 0 (x=0,   y=0): 0.004  0.000  0.017  0.000  0.021 ...
tile 1 (x=224, y=0): 0.046  0.000  0.001  0.000  0.023 ...
```

Many ResNet-50 numbers are exactly 0 because its last step (ReLU) turns negatives into 0. These tables are all the classifier will see: 52 GB of images become about 0.5 GB of numbers per model.

## Why this way

| Choice | Why | Other option |
|---|---|---|
| **Frozen** model, never retrained | 3,657 training images are far too few to retrain hundreds of millions of numbers without memorising. Frozen is also cheap, fair and repeatable | fine-tuning: later, only if it earns its place |
| 224-px tiles at 20× | the size and zoom the models were trained on (step 2) | 40×: 4.4× more tiles |
| **All tiles, including edge tiles** (120,628) | a grid alone misses up to 223 px at the right and bottom edges. Each tile's position is saved, so dropping edge tiles (83,585 full tiles) or blank tiles later needs no re-run | grid only |
| Regions under 224 px padded with white | 107 regions are too small for one tile | 40× for those, or skip them |
| **Half precision** | faster, half the memory. Checked against full precision: cosine similarity at least 0.9999, no NaNs | full precision: slower |
| Models: ResNet-50 and ImageNet ViT-L (general); UNI and H-optimus-0 (tissue) | ImageNet ViT-L and UNI have the same design and size, so any gap comes from tissue training (a fair test). H-optimus-0 has an open licence and is likely the strongest | all 6 candidates: about 5 h more, and more to explain |

**Status at 2026-11-14 18:35: all 4 models done.**
- ResNet-50: done, all 4,539 regions in 5 min, 489 MB.
- ImageNet ViT-L: done, all 4,539 regions in 0.82 h, at 41 tiles/s.
- UNI: done, all 4,539 regions (120,628 tiles) in 0.84 h at about 40 tiles/s, 252 MB, no NaN. Half vs. full precision: cosine 1.00000.
- H-optimus-0: done, all 4,539 regions in 4.12 h at about 8 tiles/s, 370 MB, no NaN. Half vs. full precision: cosine 1.00000.

## The code, part by part (`extract_features.py`)

1. **`load_model(key)`:** loads one of six models with the exact code from its model card. It also returns the model's colour normalisation, and for Virchow2 the recipe "class token + average of patch tokens".
2. **`starts(length)`:** where tiles start along one side: every 224 px, plus one last tile aligned to the edge. For 892 px: 0, 224, 448, 668.
3. **`region_tiles(path)`:**
   - open the PNG and halve it (`reduce(2)` averages each 2×2 block of pixels);
   - pad tiny regions with white;
   - cut the tiles;
   - for each tile, compute the **share of non-white pixels** ("tissue"), so blank tiles can be dropped later.
4. **`Regions` + `DataLoader`:** four helper processes read and cut images in parallel, so the GPU never waits.
5. **`run()`:** pixel values go from 0–255 to 0–1, then each colour is normalised the way that model was trained (ImageNet: subtract 0.485/0.456/0.406, divide by 0.229/0.224/0.225). Then the model runs, and its numbers come back to the CPU.
6. **`--check-fp32`:** before the real run, the same 16 tiles go through in full and in half precision. The script logs how similar the results are (cosine similarity).
7. **The queue and `process()`:** collects tiles from several regions into batches of exactly 64, because small regions alone would leave the GPU half idle. When all of a region's tiles are done, `save()` runs.
8. **`save()`:** one `.npz` file per region, holding:
   - `feats` (tiles × numbers, 16-bit)
   - `coords` (x, y of each tile)
   - `tissue`
   - `size`
   - `padded`
   - `nan_tiles`

   It is written under a temporary name, then renamed, so half-written files never exist.
9. **Progress and the final check:** a log line every minute in `data/features/<model>/_extract.log`. At the end, `DONE: x/4539`, and an exit code of 1 if anything is missing. Running it again skips finished regions.

`run_features_overnight.sh` runs the models one after another. If one fails, it still moves on to the next.

## Rebuild it yourself

Rebuild the tiling for the example region, then check your numbers against the saved file.

```python
import pathlib, numpy as np, torch, timm
from PIL import Image
P = pathlib.Path.home() / "learning/pathology-lookalikes"
img = Image.open(P / "data/bracs/BRACS_RoI/latest_version/train/1_PB/BRACS_1325_PB_11.png").convert("RGB").reduce(2)
a = np.asarray(img)
print(a.shape)                    # predict first

def starts(length):               # your version: every 224 px + one edge-aligned tile
    ...

tiles = ...                       # stack the 224x224 tiles; you should get 16

m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).eval()
x = torch.from_numpy(tiles[:1]).permute(0, 3, 1, 2).float() / 255
x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
with torch.inference_mode():
    mine = m(x)[0]
saved = np.load(P / "data/features/resnet50/train/1_PB/BRACS_1325_PB_11.npz")["feats"][0].astype(np.float32)
print(torch.nn.functional.cosine_similarity(mine, torch.from_numpy(saved), dim=0))   # expect about 1.0
```

Run it with `~/learning/pathology-lookalikes/.venv/bin/python`.

## Check yourself

1. What does "frozen" mean, and why don't we retrain the big model?
2. Why save each tile's coordinates?
3. Why fill batches of 64 tiles from several regions instead of one region at a time?
4. Why are many ResNet-50 numbers exactly 0?

<details><summary>Answers</summary>

1. Its numbers never change; we only use it to translate tiles into features. With 3,657 images, retraining hundreds of millions of numbers would mostly memorise the training images.
2. To draw the tile map later, and to drop edge or blank tiles without running the model again.
3. Many regions have only a few tiles, and the GPU is fastest with full batches.
4. ResNet's last step (ReLU) replaces every negative number with 0.
</details>

## Words to know

- **Feature / embedding:** the list of numbers a model gives for one tile.
- **Frozen backbone:** a borrowed model used as is, never retrained.
- **Foundation model:** a large model trained on huge amounts of data. UNI and H-optimus-0 were trained on hundreds of thousands of tissue slides.
- **Normalisation:** rescaling colours the way the model saw them in training.
- **Batch:** a group of tiles processed together.
- **Cosine similarity:** 1.0 means two lists of numbers point the same way.
- **`.npz`:** a NumPy file holding several arrays.
