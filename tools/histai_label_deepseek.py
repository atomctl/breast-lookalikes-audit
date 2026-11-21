#!/usr/bin/env python3
"""Ledger entry 55, labeller A: DeepSeek labels the selected HISTAI-breast reports by the written rule.

It calls the user's DeepSeek harness in one-shot mode (read-only, from an empty folder, so it sees no project file).
Report texts go to the harness's provider (Venice); they are public, de-identified dataset text.

Usage (from the project folder):
  python -I tools/histai_label_deepseek.py   # -> data/histai_meta/labels_deepseek.jsonl
"""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "data/histai_meta"
PATCH = ROOT / "tools/dsh-deepseek-high.patch.yml"
OUT = META / "labels_deepseek.jsonl"
BATCH = 12


def ask(batch):
    """One batch -> {case: label}. Cases missing from the answer are simply absent, and are asked again by the caller."""
    text = (META / "stain-label-rule.md").read_text() + "\n\nUse no tools and read no files. Label these reports.\n\n" + "\n\n".join(
        f"=== REPORT {r['case']} ===\n" + "\n".join(f"{k}: {r[k]}" for k in ("conclusion", "diagnosis", "diff_diagnostic", "micro_protocol", "additional_info") if r[k].strip())
        for r in batch)
    with tempfile.TemporaryDirectory() as empty:
        p = subprocess.run(["dsh", "--profile", "headless", "--patch", str(PATCH), "--json", "-"], input=text, capture_output=True, text=True,
                           cwd=empty, env={**os.environ, "DSH_PERMISSION_MODE": "read-only"}, timeout=900)
    final = ""
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "final":
            final = e["text"]
    want, got = {r["case"] for r in batch}, {}
    for line in final.splitlines():
        line = line.strip().strip("`")
        if line.startswith("{"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("case") in want:
                got[o["case"]] = o
    return got


def main():
    reports = [json.loads(l) for l in (META / "reports_selected.jsonl").read_text().splitlines()]
    done = {json.loads(l)["case"]: json.loads(l) for l in OUT.read_text().splitlines()} if OUT.exists() else {}
    for attempt in range(3):   # a batch that came back short is asked again, at most twice
        todo = [r for r in reports if r["case"] not in done]
        if not todo:
            break
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            for got in pool.map(ask, batches):
                done.update(got)
                OUT.write_text("".join(json.dumps(done[r["case"]]) + "\n" for r in reports if r["case"] in done))
                print(f"attempt {attempt + 1}: labelled {len(done)}/{len(reports)}", flush=True)
    print("missing after 3 attempts:", [r["case"] for r in reports if r["case"] not in done])


if __name__ == "__main__":
    main()
