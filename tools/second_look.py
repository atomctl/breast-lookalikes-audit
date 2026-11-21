#!/usr/bin/env python3
"""Ledger entry 14: a second look at the 30 test patients with the rebuilt step 1. Contaminated; not a clean test.

Usage (from the project folder):
  python -I tools/second_look.py   # -> outputs/promise/second_look.json

Reads the per-region table saved by the one test pass (outputs/final/regions_test.csv). Nothing is chosen here.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl

with open(sl.ROOT / "outputs/final/regions_test.csv") as f:
    rows = list(csv.DictReader(f))
prob = np.array([[float(r[f"p_{t}"]) for t in sl.TYPES] for r in rows])
y = np.array([sl.TYPES.index(r["label"]) for r in rows])
patient = np.array([int(r["patient"]) for r in rows])
act, lists = pp.apply_system(prob, patient)
result = {"note": "contaminated second look, not a clean test (ledger entry 14)",
          **sl.measure_two_step(act, lists, y, patient, prob, np.random.default_rng(0))}
old = np.array([r["decision"] == "act" for r in rows])
wrong = prob.argmax(1) != y
result["entry_9_for_comparison"] = {"asked_share": round(float(1 - old.mean()), 4), "error_among_acted": round(float(wrong[old].sum() / old.sum()), 4)}
(pp.OUT / "second_look.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
