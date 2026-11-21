# Step 3 · Mac or cloud? The speed test

**Code:** [`tools/speed_test.py`](../tools/speed_test.py) · **Ran:** 2026-11-14 00:50 · **Milestone:** M1 (preparation)

## In one sentence

Before choosing Kaggle or Colab, we measured how fast your Mac turns tiles into numbers. It is fast enough, so the 52 GB of images never have to leave your Mac.

## A real example

The biggest candidate model, H-optimus-0, has 1.1 billion numbers inside it. On your Mac's GPU it handled **8.5 tiles per second**.
- BRACS has 83,585 tiles at 20× (full tiles only).
- 83,585 ÷ 8.5 ≈ 9,800 seconds ≈ **2.7 hours**. One evening.

## The question and the measurements

The slow part of the whole project is turning every tile into numbers, once per model. Everything after that is quick. Cloud GPUs (Kaggle, Colab) are fast, but the data is the problem:
- the images are 52 GB and would have to be uploaded first;
- Colab's free Google Drive holds only 15 GB.

The trick: **a model's speed depends on its size and shape, not on what it learned.** So we timed stand-in models with random numbers inside (`pretrained=False`). That needs no downloads and no access approval.

| Model size | Numbers inside | Tiles per second | All of BRACS at 20× | At 40× |
|---|---|---|---|---|
| ResNet-50 (general) | 24 million | 422 | 3 min | 15 min |
| ViT-L (UNI size, ImageNet ViT-L) | 303 million | 44 | 32 min | 2.3 h |
| Virchow2 size | 631 million | 15 | 1.5 h | 6.7 h |
| UNI2-h size | 681 million | 14 | 1.6 h | 7.2 h |
| H-optimus-0 size | 1,135 million | 8.5 | 2.7 h | 12 h |

- Reading and cutting the PNGs on one CPU core: 401 tiles per second, so the CPU is not the bottleneck.
- The GPU needed at most 3.3 GB of memory.
- Later, the real pipeline ran at 41 tiles/s for ViT-L, which is 94% of the test's speed.

**Proposal: compute on the Mac at 20×.** The cloud is only needed if we ever choose 40× with the largest model.

## The code, part by part

1. **Count tiles from the PNG headers** (the step 2 trick). This gives exact totals for 20× and 40×.
2. **Build each model shape with random weights:** `timm.create_model(arch, pretrained=False, ...)`. The settings copy each model card, for example UNI2-h's 24 layers, SwiGLU and 8 registers.
3. **Half precision on the Mac GPU:** `.half()` uses 16-bit numbers, which is faster and smaller. `.to("mps")` sends the model to the GPU; MPS is Apple's GPU interface for PyTorch.
4. **Warm up, then time:** two untimed batches first, because the GPU prepares its programs on the first runs. Then 5 timed batches of 40 tiles, which makes 200 tiles.
5. **`torch.mps.synchronize()` around the timer:** the GPU works in the background, and without waiting for it you would time only the "send", not the work.
6. **Time the CPU side:** decode 24 real PNGs, halve them, and cut tiles, on one core.

## Rebuild it yourself

Predict first: how many times faster is the Mac GPU than the CPU for ResNet-50?

```python
import time, torch, timm
m = timm.create_model("resnet50", pretrained=False, num_classes=0).eval()
x = torch.randn(64, 3, 224, 224)
for device in ["cpu", "mps"]:
    mm, xx = m.to(device), x.to(device)
    with torch.inference_mode():
        mm(xx)                                   # warm-up
        if device == "mps": torch.mps.synchronize()
        t = time.perf_counter(); mm(xx)
        if device == "mps": torch.mps.synchronize()
    print(device, round(64 / (time.perf_counter() - t)), "tiles/s")
```

Run it with `~/learning/pathology-lookalikes/.venv/bin/python`. Then remove the two `synchronize` lines and see how the "speed" changes. Why?

## Check yourself

1. Why could we test speed without downloading the real models?
2. What goes wrong if you time a GPU without `synchronize()`?
3. Why is the data, not the GPU, the real problem with Kaggle or Colab here?

<details><summary>Answers</summary>

1. The amount of arithmetic depends on the model's shape. The learned values don't change how many multiplications are done.
2. The timer stops before the GPU has finished, so the speed looks much higher than it really is.
3. 52 GB would have to be uploaded first, and the free storage is too small. Locally the data is already there.
</details>

## Words to know

- **GPU:** a chip that does many multiplications at once.
- **MPS:** Apple's way for PyTorch to use the Mac's GPU.
- **Throughput:** tiles per second.
- **Half precision (fp16):** 16-bit numbers. Faster, and we checked the results stay almost identical (step 4).
- **Warm-up:** untimed first runs.
