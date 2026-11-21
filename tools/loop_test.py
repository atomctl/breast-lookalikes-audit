#!/usr/bin/env python3
"""Ledger entry 11: does a cut-off that corrects itself from checked answers keep the error promise?

Usage (from the project folder):
  python -I tools/loop_test.py   # -> outputs/loop/result.json

Uses the 121 training and validation patients only, with scores from classifiers that never saw them.
The test patients are not read. The rule that decides is entry 11 in LEDGER.md.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl

OUT = sl.ROOT / "outputs/loop"
BUDGET, HISTORY, RUNS = 0.10, 40, 200


def stream(conf, wrong, start, step, check_share, rng):
    """One pass over arriving cases. Returns (answered alone, checked) as boolean arrays.
    The cut-off moves only after an answered-alone case that was checked."""
    cut = start  # an infinite start (no cut-off passed on the history) answers nothing; the first correction needs an answer, so it stays there
    act, checked = np.zeros(len(conf), bool), np.zeros(len(conf), bool)
    draws = rng.random(len(conf))
    for i in range(len(conf)):
        if conf[i] >= cut:
            act[i] = True
            if step and draws[i] < check_share:
                checked[i] = True
                cut = min(1.0, max(0.0, cut + step * (wrong[i] - BUDGET)))
    return act, checked


def summarise(runs):
    a = np.array(runs, float)  # columns: error among answered alone, work share, error in the second half, classifier error
    err = a[:, 0]
    return {"mean_error_among_answered_alone": round(float(np.nanmean(err)), 4),
            "error_5th_to_95th_percentile": [round(float(v), 4) for v in np.nanpercentile(err, [5, 95])],
            "runs_with_error_at_most_12%": round(float(np.mean(err <= 0.12)), 3),
            "mean_error_second_half": round(float(np.nanmean(a[:, 2])), 4),
            "mean_work_share_for_pathologist": round(float(a[:, 1].mean()), 4),
            "mean_classifier_error_on_arrivals": round(float(a[:, 3].mean()), 4)}


def main():
    prob, y, patient, _, vprob, vy, vpatient = sl.load()
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    conf, wrong = prob.max(1), (prob.argmax(1) != y).astype(float)
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    accuracy = np.array([1 - wrong[where[p]].mean() for p in ids])
    best_first = ids[np.argsort(-accuracy)]
    variants = {"fixed cut-off": (0.0, 0.0), "self-correcting, 10% spot checks (decides)": (0.02, 0.10),
                "self-correcting, every answer checked": (0.02, 1.0),
                "spot checks, step 0.01": (0.01, 0.10), "spot checks, step 0.05": (0.05, 0.10)}
    result = {"patients": int(len(ids)), "regions": len(y), "history_patients": HISTORY, "runs": RUNS}
    for scenario in ("E, no shift", "S, harder arrivals"):
        rng = np.random.default_rng(0)
        runs = {k: [] for k in variants}
        for _ in range(RUNS):
            if scenario.startswith("E"):
                order = rng.permutation(ids)
                history, arrivals = order[:HISTORY], order[HISTORY:]
            else:
                history = rng.choice(best_first[:60], HISTORY, replace=False)
                arrivals = rng.permutation(best_first[60:])
            h = np.concatenate([where[p] for p in history])
            s = np.concatenate([where[p] for p in arrivals])
            start = sl.act_cut_off(conf[h], wrong[h].astype(bool), BUDGET)
            seed = int(rng.integers(1 << 30))
            for name, (step, share) in variants.items():
                act, checked = stream(conf[s], wrong[s], start, step, share, np.random.default_rng(seed))
                half = np.arange(len(s)) >= len(s) // 2
                err = lambda m: wrong[s][m].sum() / m.sum() if m.any() else np.nan
                runs[name].append([err(act), (~act).mean() + checked.mean(), err(act & half), wrong[s].mean()])
        result[scenario] = {k: summarise(v) for k, v in runs.items()}
    e, s = result["E, no shift"], result["S, harder arrivals"]
    key, m = "self-correcting, 10% spot checks (decides)", "mean_error_among_answered_alone"
    shift_made = s["fixed cut-off"][m] > 0.13
    passed = s[key][m] <= 0.13 and s["fixed cut-off"][m] - s[key][m] >= 0.03 and 0.07 <= e[key][m] <= 0.13
    result["verdict"] = "inconclusive: the scenario made no shift" if not shift_made else ("PASS" if passed else "FAIL")
    # self-checks: a zero step never moves the cut-off; a cut-off of 1.01 answers nothing
    c, w = np.array([0.9, 0.5, 0.95]), np.array([1.0, 0.0, 0.0])
    assert stream(c, w, 0.8, 0.0, 1.0, np.random.default_rng(0))[0].tolist() == [True, False, True]
    assert not stream(c, w, 1.01, 0.0, 1.0, np.random.default_rng(0))[0].any()
    assert stream(c, w, 0.8, 0.2, 1.0, np.random.default_rng(0))[0].tolist() == [True, False, False]  # one mistake lifts the cut-off to 0.98
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
