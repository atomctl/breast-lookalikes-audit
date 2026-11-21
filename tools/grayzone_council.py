#!/usr/bin/env python3
"""Ledger entry 45: does disagreement in a council of differently built experts find the classifier's wrong answers?

Usage (from the project folder):
  python -I tools/grayzone_council.py   # -> outputs/grayzone/council.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl
from train_classifier import patient_draws, range95

G = sl.ROOT / "outputs/grayzone"
GRAY = [2, 3, 4, 5]


def main():
    prob, y, patient, *_ = sl.load()
    experts = {"E": prob, "T": np.load(G / "oof_transformer_with_tile_positions.npy"), "A": np.load(G / "oof_attention_pooling.npy"), "H": np.load(G / "oof_B_hoptimus0.npy")}
    assert all(len(v) == len(y) for v in experts.values())
    top = prob.argmax(1)
    wrong = (top != y).astype(float)
    council = np.mean([p[np.arange(len(y)), top] for p in experts.values()], axis=0)
    sig = {"patient context (in use)": pp.signals(prob, prob, patient)["patient context"], "council": council,
           "council with patient context": (council + pp.patient_mean(council, patient)) / 2}
    draws = patient_draws(patient, np.random.default_rng(0))
    gray = np.isin(y, GRAY)
    base = "patient context (in use)"
    result = {"regions": len(y), "experts": list(experts), "expert_accuracy": {k: round(float((v.argmax(1) == y).mean()), 3) for k, v in experts.items()},
              "all_four_agree_share": round(float(np.all([v.argmax(1) == top for v in experts.values()], axis=0).mean()), 3)}
    agree = np.all([v.argmax(1) == top for v in experts.values()], axis=0)
    result["error_when_all_agree"] = round(float(wrong[agree].mean()), 4)
    result["error_when_not_all_agree"] = round(float(wrong[~agree].mean()), 4)
    keep = int(round(0.34 * len(y)))
    for name, s in sig.items():
        order = np.argsort(-s, kind="stable")[:keep]
        r = {"aurc_all": round(pp.aurc(s, wrong), 4), "aurc_gray_zone": round(pp.aurc(s[gray], wrong[gray]), 4),
             "error_among_the_34pct_most_confident": round(float(wrong[order].mean()), 4)}
        if name != base:
            d = [pp.aurc(s[i], wrong[i]) - pp.aurc(sig[base][i], wrong[i]) for i in draws]
            r.update({"aurc_minus_in_use": round(r["aurc_all"] - pp.aurc(sig[base], wrong), 4), "range95": range95(d), "adopted": bool(np.percentile(d, 97.5) < 0)})
        result[name] = r
    (G / "council.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
