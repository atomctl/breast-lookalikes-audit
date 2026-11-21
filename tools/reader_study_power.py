#!/usr/bin/env python3
"""Reader study: how many cases and readers are needed to see a given gain in agreement with the consensus?

Usage (from the project folder):
  python -I tools/reader_study_power.py   # -> outputs/reader_study/power.json

Simulation. Every reader reads every case twice (without and with the tool). Cases differ in difficulty and readers in skill
(both on the logit scale); the tool shifts every reading by the same amount. Test: is the mean gain per case above zero
(one-sided paired t-test over cases, readers averaged)? Assumptions are stated in the output; they are guesses until a pilot.
"""
import json
import pathlib

import numpy as np
from scipy import stats

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs/reader_study"
CASE_SD, READER_SD, RUNS = 1.0, 0.3, 2000


def logit(p):
    return np.log(p / (1 - p))


def shift_for(p_target, base, rng):
    """The logit shift that moves the average agreement from the base level to p_target."""
    z = rng.normal(0, np.hypot(CASE_SD, READER_SD), 200_000)
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(base + mid + z)))).mean() < p_target else (lo, mid)
    return (lo + hi) / 2


def power(cases, readers, p0, p1, rng):
    base = shift_for(p0, 0.0, rng)
    gain = shift_for(p1, base, rng)
    hits = 0
    for _ in range(RUNS):
        z = base + rng.normal(0, CASE_SD, (cases, 1)) + rng.normal(0, READER_SD, (1, readers))
        a = rng.random((cases, readers)) < 1 / (1 + np.exp(-z))
        b = rng.random((cases, readers)) < 1 / (1 + np.exp(-(z + gain)))
        d = b.mean(1) - a.mean(1)
        hits += stats.ttest_1samp(d, 0, alternative="greater").pvalue < 0.05
    return hits / RUNS


def main():
    rng = np.random.default_rng(0)
    result = {"assumptions": {"agreement_without_tool": 0.60, "case_difficulty_sd_logit": CASE_SD, "reader_skill_sd_logit": READER_SD,
                              "test": "one-sided paired t-test over cases at 5%", "runs": RUNS,
                              "note": "0.60 is a guess between the 48% for atypia and 84% for DCIS in Elmore et al. 2015; a pilot should replace it"}, "power": {}}
    for gain in (0.05, 0.10, 0.15):
        for readers in (2, 3, 5):
            for cases in (60, 80, 100, 120):
                result["power"][f"gain {gain:.2f} | {readers} readers | {cases} cases"] = round(power(cases, readers, 0.60, 0.60 + gain, rng), 3)
    assert power(80, 3, 0.60, 0.60, rng) < 0.09  # no gain: the test fires about 5% of the time
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power.json").write_text(json.dumps(result, indent=1))
    for k, v in result["power"].items():
        print(k, v)


if __name__ == "__main__":
    main()
