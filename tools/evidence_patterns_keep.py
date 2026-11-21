#!/usr/bin/env python3
"""Ledger entry 22: the named-patterns test of entry 20, rerun with the KEEP text-image model.

Usage (from the project folder):
  python -I tools/evidence_patterns_keep.py   # -> outputs/evidence/patterns_keep.json

Training and validation patients only. At most 12 tissue tiles per region (fixed seed).
KEEP: Astaxanthin/KEEP on Hugging Face, MIT licence. Its model file was read before this script was written.
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import evidence_patterns as ep
import extract_features as ef

OUT = ep.OUT / "keep"
PER_REGION = 12


class Keep(torch.nn.Module):
    """KEEP rebuilt from its published model file (modeling_keep.py), without running that file:
    its own loader breaks on the installed timm, which passes extra arguments to the layer it replaces."""

    def __init__(self, config):
        import timm
        from transformers import BertConfig, BertModel
        super().__init__()
        v = config["vision_config"]
        self.visual = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=v["img_size"], patch_size=v["patch_size"],
                                        init_values=v["init_values"], num_classes=v["num_classes"])
        d = config["projection_dim"]
        self.visual_head = torch.nn.Sequential(torch.nn.Linear(self.visual.num_features, d), torch.nn.GELU(), torch.nn.Linear(d, d))
        self.text = BertModel(BertConfig(**config["text_config"]))
        self.logit_scale = torch.nn.Parameter(torch.ones([]))

    def encode_image(self, x):
        return torch.nn.functional.normalize(self.visual_head(self.visual(x)), dim=-1)

    def encode_text(self, t):
        return torch.nn.functional.normalize(self.text(**t).pooler_output, dim=-1)


def load_keep():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import BertTokenizer
    config = json.loads(open(hf_hub_download("Astaxanthin/KEEP", "config.json")).read())
    model = Keep(config)
    state = load_file(hf_hub_download("Astaxanthin/KEEP", "model.safetensors"))
    state = {k.replace(".ls1.weight", ".ls1.gamma").replace(".ls2.weight", ".ls2.gamma"): v for k, v in state.items()}  # timm's name for the same numbers
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "position_ids" not in k]  # a counting buffer, not a weight
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.eval(), BertTokenizer.from_pretrained("Astaxanthin/KEEP")


def embed_regions(paths, model, dev, log=None):
    """Mean KEEP image embedding of at most PER_REGION tissue tiles per region (fixed seed)."""
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    rng = np.random.default_rng(0)
    loader = torch.utils.data.DataLoader(ef.Regions(paths), batch_size=None, shuffle=False, num_workers=4)
    x = [None] * len(paths)
    for i, tiles, coords, tissue, size, padded in loader:
        keep = np.flatnonzero(np.asarray(tissue) >= 0.5)
        keep = keep if len(keep) else np.arange(len(tiles))
        pick = np.sort(rng.choice(keep, min(PER_REGION, len(keep)), replace=False))
        with torch.inference_mode():
            t = tiles[pick].to(dev).permute(0, 3, 1, 2).float().div_(255)
            x[i] = model.encode_image((t - mean) / std).float().cpu().numpy().mean(0)
        if log and i % 400 == 0:
            print(i, "/", len(paths), flush=True)
    return np.array(x)


def main():
    model, tok = load_keep()
    with torch.inference_mode():
        text = []
        for p in list(ep.PATTERNS) + ep.TYPE_NAMES:
            t = tok([s.format(p) for s in ep.TEMPLATES], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            e = model.encode_text(t).mean(0)
            text.append(torch.nn.functional.normalize(e, dim=0).numpy())
    text = np.array(text)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "text.npy", text)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev)
    held = ep.split_rows("train") + ep.split_rows("val")
    cache = OUT / "region_mean.npy"
    if cache.exists():
        x = np.load(cache)
    else:
        x = embed_regions([ep.ROOT / "data/bracs/BRACS_RoI/latest_version" / r["file"] for r in held], model, dev, log=True)
        np.save(cache, x)
    y = np.array([int(r["label"]) for r in held])
    patient = np.array([int(r["patient"]) for r in held])
    result = {"model": "Astaxanthin/KEEP", "tiles_per_region_at_most": PER_REGION, **ep.test_patterns(x @ text.T, y, patient)}
    (ep.OUT / "patterns_keep.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "patterns"}, indent=2))


if __name__ == "__main__":
    main()
