#!/usr/bin/env python3
"""Ledger entry 36: ask whenever ADH is a live possibility, on top of the cut-off of entry 17.

Usage (from the project folder):
  python -I tools/grayzone_guard.py   # -> outputs/grayzone/guard.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import patient_promise as pp
import short_lists as sl

ADH, DCIS = 4, 5


def guard_value(prob, y, patient):
    """The ADH probability that at least 90% of truly ADH calibration regions reach (patients counted)."""
    m = y == ADH
    return 1 - sl.threshold(1 - prob[m, ADH], sl.PROMISE, len(np.unique(patient[m])))


def main():
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    top = prob.argmax(1)
    wrong = (top != y).astype(float)
    score = pp.signals(prob, prob, patient)["patient context"]
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    rng = np.random.default_rng(0)
    rows = {"without": [], "with": []}
    guards = []
    for n in range(pp.SPLITS):
        order = rng.permutation(ids)
        cal, new = np.concatenate([where[p] for p in order[pp.NEW_PATIENTS:]]), np.concatenate([where[p] for p in order[:pp.NEW_PATIENTS]])
        t = pp.strict_cut_off(score[cal], wrong[cal], patient[cal], rng)
        g = guard_value(prob[cal], y[cal], patient[cal])
        guards.append(g)
        for name, act_c, act_n in (("without", score[cal] >= t, score[new] >= t),
                                   ("with", (score[cal] >= t) & (prob[cal, ADH] < g), (score[new] >= t) & (prob[new, ADH] < g))):
            asked_cal = cal[~act_c]
            hit = np.nan
            if (~act_n).any() and len(asked_cal):
                lists = pp.lists_for_asked(prob, y, patient, asked_cal, prob[new][~act_n])
                hit = lists[np.arange((~act_n).sum()), y[new][~act_n]].mean()
            yn, wn, tn = y[new], wrong[new].astype(bool), top[new]
            adh, dcis = yn == ADH, yn == DCIS
            rows[name].append([wn[act_n].sum() / act_n.sum() if act_n.any() else 0.0, 1 - act_n.mean(), hit,
                               (act_n & adh & wn).sum() / max(adh.sum(), 1), (act_n & adh).sum() / max(adh.sum(), 1),
                               (act_n & (tn == ADH)).sum(), (act_n & dcis & wn).sum() / max(dcis.sum(), 1)])
        if (n + 1) % 50 == 0:
            print(n + 1, "splits", flush=True)
    result = {"splits": pp.SPLITS, "guard_adh_probability_median": round(float(np.median(guards)), 4),
              "guard_adh_probability_5th_to_95th": [round(float(v), 4) for v in np.percentile(guards, [5, 95])]}
    for name, r in rows.items():
        a = np.array(r, float)
        result[name + "_guard"] = {"budget_kept_in_share_of_splits": round(float((a[:, 0] <= pp.BUDGET).mean()), 3),
                                   "mean_error_among_answered_alone": round(float(a[:, 0].mean()), 4),
                                   "mean_asked_share": round(float(a[:, 1].mean()), 4),
                                   "mean_hit_rate_among_asked": round(float(np.nanmean(a[:, 2])), 4),
                                   "mean_share_of_adh_regions_answered_alone_and_wrong": round(float(a[:, 3].mean()), 4),
                                   "mean_share_of_adh_regions_answered_alone": round(float(a[:, 4].mean()), 4),
                                   "mean_lone_answers_of_adh_per_batch": round(float(a[:, 5].mean()), 2),
                                   "mean_share_of_dcis_regions_answered_alone_and_wrong": round(float(a[:, 6].mean()), 4)}
    w, o = result["with_guard"], result["without_guard"]
    checks = {"a_adh_wrong_alone_halved": bool(w["mean_share_of_adh_regions_answered_alone_and_wrong"] <= 0.5 * o["mean_share_of_adh_regions_answered_alone_and_wrong"]),
              "b_promise_kept": bool(w["budget_kept_in_share_of_splits"] >= 0.90),
              "c_asked_share_rise_at_most_10_points": bool(w["mean_asked_share"] - o["mean_asked_share"] <= 0.10)}
    result["checks"] = checks
    result["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    (sl.ROOT / "outputs/grayzone/guard.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
