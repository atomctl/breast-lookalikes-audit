#!/usr/bin/env python3
"""Ledger entry 55, milestone 4: the slide-level step on the HISTAI-breast slides that the frozen rule needs.

Run ONLY after the gate has passed (tools/slide_gate.py test). Slides are streamed: download, extract, delete.
The cases come from data/histai_meta/run_plan.json (the UDH cases, the ADH cases, the cases with a stated CK5/6
result, and 40 DCIS cases drawn with seed 0 for the positive control), in a seeded random order, so a run that
stops early leaves a random part. Nothing here reads a label or prints a score.

Usage (from the project folder):
  python -I tools/histai_run.py
"""
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
ROOT = pathlib.Path(__file__).resolve().parent.parent
TMP = ROOT / "data/histai_tmp"
OUT = ROOT / "data/features/slide_step/histai"
MPP = 0.5   # the files carry no usable resolution; the archive states 20x (as read in the Granada paper)


def main():
    from huggingface_hub import hf_hub_download
    import slide_step as ss
    plan = json.loads((ROOT / "data/histai_meta/run_plan.json").read_text())
    random.Random(0).shuffle(plan)
    todo = [(r["case"], n, s["path"]) for r in plan for n, s in enumerate(r["slides"])]
    todo = [t for t in todo if not (OUT / f"{t[0]}__{t[1]}.npz").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    fetch = lambda t: hf_hub_download("histai/HISTAI-breast", t[2], repo_type="dataset", local_dir=TMP)
    pool = ThreadPoolExecutor(1)
    ahead = pool.submit(fetch, todo[0]) if todo else None
    uni = ss.Uni()
    for i, t in enumerate(todo):
        local = pathlib.Path(ahead.result())
        ahead = pool.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
        try:
            out = ss.process(local, MPP, uni)
        except Exception as e:   # one unreadable slide must not stop the run; it is listed and counted
            (OUT / f"{t[0]}__{t[1]}.failed.txt").write_text(f"{type(e).__name__}: {e}")
            print(t[0], t[1], "FAILED", type(e).__name__, flush=True)
        else:
            tmp = OUT / f"{t[0]}__{t[1]}.tmp.npz"
            np.savez(tmp, **out)
            tmp.rename(OUT / f"{t[0]}__{t[1]}.npz")
            print(f"{i + 1}/{len(todo)} {t[0]} slide {t[1]} | tissue tiles {out['tissue_tiles']} | duct windows {len(out['windows'])}", flush=True)
        local.unlink()


if __name__ == "__main__":
    main()
