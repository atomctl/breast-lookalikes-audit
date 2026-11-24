#!/usr/bin/env python3
"""M2: short lists with a promised hit rate (conformal prediction), the routing rule, and the checks before the test.

Usage (from the project folder):
  python -I tools/short_lists.py check    # ledger entry 6 -> outputs/m2/check.json    (training and validation patients only)
  python -I tools/short_lists.py check2   # ledger entry 7 -> outputs/m2/check2.json   (the two-step design)
  python -I tools/short_lists.py check3   # ledger entry 8 -> outputs/m2/check3.json   (list thresholds count patients)

The rules for these experiments are entries 6 and 7 in LEDGER.md. The test set is not read here; tools/final_test.py does that, once.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path

import folds as foldlib
from train_classifier import ROOT, TYPES, classifier, patient_draws, pooled, range95, rows

OUT = ROOT / "outputs/m2"
ENGINE = {"model": "uni", "min_tissue": 0.5, "C": 0.001, "whole_view": True}  # ledger entry 10: tile average + whole-region view
PROMISE = 0.90
DIAL = [0.80, 0.85, 0.90, 0.95, 0.99]


# ---------- the conformal part ----------

def threshold(scores, promise, units=None):
    """The cut-off that at least `promise` of new cases stay under, given calibration scores.
    `units`: count this many independent cases (patients) instead of len(scores); stricter (ledger entry 8)."""
    n = units or len(scores)
    k = int(np.ceil((n + 1) * promise))
    if k > n:
        return np.inf
    return float(np.sort(scores)[int(np.ceil(k / n * len(scores))) - 1])


def lac_scores(prob):
    return 1 - prob  # a type's score: how far its probability is from 1


def aps_scores(prob, strict=False):
    """A type's score: the total probability of all types at least as likely as it (strict: more likely than it)."""
    order = np.argsort(-prob, axis=1)
    sorted_prob = np.take_along_axis(prob, order, axis=1)
    cum = np.cumsum(sorted_prob, axis=1) - (sorted_prob if strict else 0)
    out = np.empty_like(prob)
    np.put_along_axis(out, order, cum, axis=1)
    return out


METHODS = {"LAC, one threshold per type": (lac_scores, True), "LAC, one threshold": (lac_scores, False), "APS": (aps_scores, False)}


def calibrate(method, prob, y, promise, patient=None):
    """Thresholds (one per type) from calibration cases. With `patient`, patients are counted instead of regions."""
    score_fn, per_type = METHODS[method]
    true_score = score_fn(prob)[np.arange(len(y)), y]
    units = (lambda m: len(np.unique(patient[m]))) if patient is not None else (lambda m: None)
    if per_type:
        return np.array([threshold(true_score[y == k], promise, units(y == k)) for k in range(7)])
    return np.full(7, threshold(true_score, promise, units(slice(None))))


def short_lists(method, prob, thresholds):
    """Boolean table, one row per region: which types are on its list."""
    if method == "APS":  # a type is on the list while the types more likely than it have not yet reached the threshold,
        return aps_scores(prob, strict=True) < thresholds  # so the top type is always on it
    return METHODS[method][0](prob) <= thresholds


# ---------- measuring ----------

def measure(lists, y, patient, prob, rng):
    size = lists.sum(1)
    hit = lists[np.arange(len(y)), y]
    act = size == 1                       # the routing rule: exactly one type = act, anything else = ask
    wrong_act = act & ~hit
    draws = patient_draws(patient, rng, 1000)
    share = lambda v, i=slice(None): float(v[i].mean())
    acted_error = lambda i=slice(None): float(wrong_act[i].sum() / max(act[i].sum(), 1))
    return {
        "hit_rate": round(share(hit), 4), "hit_rate_range95": range95([share(hit, i) for i in draws]),
        "hit_rate_by_type": {t: round(share(hit[y == k]), 4) for k, t in enumerate(TYPES)},
        "list_length": round(float(size.mean()), 3),
        "list_length_by_type": {t: round(float(size[y == k].mean()), 3) for k, t in enumerate(TYPES)},
        "asked_share": round(1 - share(act), 4), "asked_share_range95": range95([1 - share(act, i) for i in draws]),
        "asked_share_by_type": {t: round(1 - share(act[y == k]), 4) for k, t in enumerate(TYPES)},
        "empty_share": round(share(size == 0), 4),
        "error_among_acted": round(acted_error(), 4), "error_among_acted_range95": range95([acted_error(i) for i in draws]),
        "error_without_lists": round(share(prob.argmax(1) != y), 4),
    }


def cut_off_matched(prob_cal, prob_new, y_new, asked_share):
    """Baseline: act when the top probability is high enough, asking the same share as the short lists do."""
    conf = prob_new.max(1)
    t = np.quantile(prob_cal.max(1), asked_share)  # set on calibration cases
    act = conf >= t
    return {"asked_share": round(float(1 - act.mean()), 4),
            "error_among_acted": round(float(((prob_new.argmax(1) != y_new) & act).sum() / max(act.sum(), 1)), 4)}


def conformal_selection(prob_cal, y_cal, prob_new, y_new, budget):
    """Act only on cases whose top type is right, with the error share among acted cases held under `budget`
    (Jin and Candes 2023: conformal p-values against the wrong calibration cases, then Benjamini-Hochberg)."""
    conf_cal, wrong_cal = prob_cal.max(1), prob_cal.argmax(1) != y_cal
    conf = prob_new.max(1)
    p = (1 + (conf_cal[wrong_cal][None, :] >= conf[:, None]).sum(1)) / (len(y_cal) + 1)
    order = np.argsort(p)
    passed = np.flatnonzero(p[order] <= budget * np.arange(1, len(p) + 1) / len(p))
    act = np.zeros(len(p), bool)
    if len(passed):
        act[order[:passed[-1] + 1]] = True
    return act


def act_cut_off(conf, wrong, budget, confidence=0.90):
    """Step 1 of the two-step design: the lowest cut-off whose error among acted calibration cases is under
    `budget` with the given confidence. Cut-offs are tested from the highest down; the first failure stops the search."""
    from scipy.stats import binom
    best = np.inf
    for asked in np.arange(0.95, -0.001, -0.01):  # start where at least 5% of cases are acted on
        t = np.quantile(conf, max(asked, 0.0))  # the last step of the range can land a hair under zero
        act = conf >= t
        if binom.cdf(int(wrong[act].sum()), int(act.sum()), budget) > 1 - confidence:  # cannot rule out "error above budget"
            break
        best = t
    return float(best)


def two_step(prob_cal, y_cal, prob_new, budget, promise, method="LAC, one threshold per type", patient_cal=None):
    """Returns (act, lists, cut-off) for new cases. Lists are calibrated on the calibration cases that would be asked.
    With `patient_cal`, the list thresholds count patients instead of regions (ledger entry 8)."""
    t = act_cut_off(prob_cal.max(1), prob_cal.argmax(1) != y_cal, budget)
    asked_cal = prob_cal.max(1) < t
    lists = short_lists(method, prob_new, calibrate(method, prob_cal[asked_cal], y_cal[asked_cal], promise,
                                                    None if patient_cal is None else patient_cal[asked_cal]))
    return prob_new.max(1) >= t, lists, t


def measure_two_step(act, lists, y, patient, prob, rng):
    wrong, hit, asked = prob.argmax(1) != y, lists[np.arange(len(y)), y], ~act
    draws = patient_draws(patient, rng, 1000)
    acted_error = lambda i=slice(None): float((wrong & act)[i].sum() / max(act[i].sum(), 1))
    asked_hit = lambda i=slice(None): float((hit & asked)[i].sum() / max(asked[i].sum(), 1))
    return {
        "asked_share": round(float(asked.mean()), 4), "asked_share_range95": range95([float(asked[i].mean()) for i in draws]),
        "asked_share_by_type": {t: round(float(asked[y == k].mean()), 4) for k, t in enumerate(TYPES)},
        "error_among_acted": round(acted_error(), 4), "error_among_acted_range95": range95([acted_error(i) for i in draws]),
        "acted": int(act.sum()), "acted_wrong": int((wrong & act).sum()),
        "hit_rate_among_asked": round(asked_hit(), 4), "hit_rate_among_asked_range95": range95([asked_hit(i) for i in draws]),
        "hit_rate_among_asked_by_type": {t: round(float(hit[asked & (y == k)].mean()), 4) for k, t in enumerate(TYPES)},
        "asked_by_type": {t: int((asked & (y == k)).sum()) for k, t in enumerate(TYPES)},
        "list_length_among_asked": round(float(lists[asked].sum(1).mean()), 3),
        "list_length_among_asked_by_type": {t: round(float(lists[asked & (y == k)].sum(1).mean()), 3) for k, t in enumerate(TYPES)},
        "error_without_system": round(float(wrong.mean()), 4),
        "missed_overall": round(float(((wrong & act) | (~hit & asked)).mean()), 4),  # acted and wrong, or asked and not on the list
    }


BUDGET = 0.10
BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20]


def check2(count_patients=False):
    prob, y, patient, folds, vprob, vy, vpatient = load()
    rng = np.random.default_rng(0)
    pc = (lambda i: patient[i]) if count_patients else (lambda i: None)

    def fold_run(budget, method="LAC, one threshold per type"):
        act, lists = np.zeros(len(y), bool), np.zeros((len(y), 7), bool)
        for a, b in folds:
            act[b], lists[b], _ = two_step(prob[a], y[a], prob[b], budget, PROMISE, method, pc(a))
        return act, lists

    result = {"engine": ENGINE, "budget": BUDGET, "promise": PROMISE, "thresholds_count_patients": count_patients}
    result["A_fold_check"] = measure_two_step(*fold_run(BUDGET), y, patient, prob, rng)
    act, lists, t = two_step(prob, y, vprob, BUDGET, PROMISE, patient_cal=pc(slice(None)))
    result["B_validation"] = {"cut_off": round(t, 4), **measure_two_step(act, lists, vy, vpatient, vprob, rng)}
    result["budgets_fold_check"] = {}
    for budget in BUDGETS:
        m = measure_two_step(*fold_run(budget), y, patient, prob, rng)
        result["budgets_fold_check"][f"{budget:.0%}"] = {k: m[k] for k in ("asked_share", "error_among_acted", "hit_rate_among_asked", "list_length_among_asked")}
    result["list_methods_on_asked_fold_check"] = {}
    for method in METHODS:
        m = measure_two_step(*fold_run(BUDGET, method), y, patient, prob, rng)
        result["list_methods_on_asked_fold_check"][method] = {k: m[k] for k in ("hit_rate_among_asked", "hit_rate_among_asked_by_type", "list_length_among_asked")}
    a = result["A_fold_check"]
    result["stop_rules"] = {"error_among_acted_at_most_10%": a["error_among_acted"] <= 0.10,
                            "hit_rate_among_asked_at_least_87%": a["hit_rate_among_asked"] >= 0.87,
                            "every_type_at_least_80%": min(a["hit_rate_among_asked_by_type"].values()) >= 0.80}
    result["stop_rules"]["final_system"] = all(result["stop_rules"].values())
    # self-checks: a zero budget acts on nothing it cannot prove; the APS list always holds the top type
    assert not np.isfinite(act_cut_off(np.array([0.9, 0.8, 0.7] * 50), np.array([True, False, False] * 50), 0.0))
    aps = short_lists("APS", prob, calibrate("APS", prob, y, PROMISE))
    assert aps[np.arange(len(y)), prob.argmax(1)].all()
    (OUT / ("check3.json" if count_patients else "check2.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ---------- data ----------

def engine_features(split, held=None):
    """The classifier's input, one row per region: UNI's tile average next to UNI's view of the whole region."""
    return np.hstack([pooled(ENGINE["model"], split, ENGINE["min_tissue"]), np.load(ROOT / "outputs/m1/thunder" / f"uni_{split}.npy")])


def load():
    """Calibration material: out-of-fold probabilities for the training patients, and validation probabilities
    from the classifier fitted on all training patients (saved as outputs/m2/engine.joblib)."""
    import joblib
    train, held = rows("train"), rows("val")
    y = np.array([int(r["label"]) for r in train])
    patient = np.array([int(r["patient"]) for r in train])
    folds = list(foldlib.split(5, y, patient))  # the same folds as M1, from outputs/folds.json
    oof_cache, val_cache = OUT / "oof_prob_engine.npy", OUT / "val_prob_engine.npy"
    if not (oof_cache.exists() and val_cache.exists()):
        x = engine_features("train")
        prob = np.zeros((len(y), 7))
        for a, b in folds:
            prob[b] = classifier(ENGINE["C"]).fit(x[a], y[a]).predict_proba(x[b])
        clf = classifier(ENGINE["C"]).fit(x, y)
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, OUT / "engine.joblib")
        np.save(oof_cache, prob)
        np.save(val_cache, clf.predict_proba(engine_features("val")))
    vy = np.array([int(r["label"]) for r in held])
    vpatient = np.array([int(r["patient"]) for r in held])
    return np.load(oof_cache), y, patient, folds, np.load(val_cache), vy, vpatient


def check():
    prob, y, patient, folds, vprob, vy, vpatient = load()
    rng = np.random.default_rng(0)
    result = {"engine": ENGINE, "promise": PROMISE, "A_fold_check": {}, "B_validation": {}, "dial_fold_check": {}}

    def fold_lists(method, promise):
        lists = np.zeros((len(y), 7), bool)
        for a, b in folds:
            lists[b] = short_lists(method, prob[b], calibrate(method, prob[a], y[a], promise))
        return lists

    for method in METHODS:
        result["A_fold_check"][method] = measure(fold_lists(method, PROMISE), y, patient, prob, rng)
        result["B_validation"][method] = measure(short_lists(method, vprob, calibrate(method, prob, y, PROMISE)), vy, vpatient, vprob, rng)
    primary = "LAC, one threshold per type"
    for promise in DIAL:
        m = measure(fold_lists(primary, promise), y, patient, prob, rng)
        result["dial_fold_check"][str(promise)] = {k: m[k] for k in ("hit_rate", "list_length", "asked_share", "error_among_acted")}

    asked = result["A_fold_check"][primary]["asked_share"]
    acted, wrong, cut = np.zeros(len(y), bool), prob.argmax(1) != y, {"asked": 0, "wrong": 0, "acted": 0}
    for a, b in folds:
        c = cut_off_matched(prob[a], prob[b], y[b], asked)
        cut["acted"] += round((1 - c["asked_share"]) * len(b)); cut["wrong"] += c["error_among_acted"] * round((1 - c["asked_share"]) * len(b))
    result["act_or_ask_fold_check"] = {
        "short lists (primary)": {"asked_share": asked, "error_among_acted": result["A_fold_check"][primary]["error_among_acted"]},
        "confidence cut-off, same asked share": {"asked_share": round(1 - cut["acted"] / len(y), 4), "error_among_acted": round(cut["wrong"] / max(cut["acted"], 1), 4)}}
    for budget in (0.05, 0.10):
        acted[:] = False
        for a, b in folds:
            acted[b] = conformal_selection(prob[a], y[a], prob[b], y[b], budget)
        result["act_or_ask_fold_check"][f"conformal selection, error budget {budget:.0%}"] = {
            "asked_share": round(float(1 - acted.mean()), 4), "error_among_acted": round(float((wrong & acted).sum() / max(acted.sum(), 1)), 4)}

    worst = min(result["A_fold_check"][primary]["hit_rate_by_type"].values())
    result["stop_rule"] = {"lowest_type_hit_rate_fold_check": worst, "primary_method_usable": bool(worst >= 0.85)}
    # self-checks: a promise of 100% puts every type on every list; lists only grow with the promise
    assert short_lists(primary, prob, calibrate(primary, prob, y, 1.0)).all()
    assert (fold_lists(primary, 0.95) >= fold_lists(primary, 0.80)).all()
    (OUT / "check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"check": check, "check2": check2, "check3": lambda: check2(count_patients=True)}[sys.argv[1]]()
