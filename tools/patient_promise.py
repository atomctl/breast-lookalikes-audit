#!/usr/bin/env python3
"""Ledger entry 12: rebuild the "answer alone" promise as a promise about new patients.

Usage (from the project folder):
  python -I tools/patient_promise.py   # -> outputs/promise/result.json

Uses the 121 training and validation patients only (out-of-fold scores). The test patients are not read.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import classifier, pooled, range95

OUT = sl.ROOT / "outputs/promise"
BUDGET, CONFIDENCE, SPLITS, NEW_PATIENTS = 0.10, 0.90, 300, 30
SECOND = {"model": "hoptimus0", "min_tissue": 0.0, "C": 0.003}  # the H-optimus-0 classifier of ledger entry 4


def second_opinion_prob(y, folds):
    """Out-of-fold probabilities of the H-optimus-0 classifier for the training patients, then validation."""
    cache = OUT / "second_oof.npy"
    if not cache.exists():
        x = pooled(SECOND["model"], "train", SECOND["min_tissue"])
        prob = np.zeros((len(y), 7))
        for a, b in folds:
            prob[b] = classifier(SECOND["C"]).fit(x[a], y[a]).predict_proba(x[b])
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(cache, prob)
    return np.vstack([np.load(cache), np.load(sl.ROOT / "outputs/m1/val_probs_hoptimus0.npz")["prob"]])


def patient_mean(values, patient):
    ids, inv = np.unique(patient, return_inverse=True)
    return (np.bincount(inv, weights=values) / np.bincount(inv))[inv]


def signals(prob, second, patient):
    """Confidence signals for "answer alone". All need no labels."""
    top = prob.argmax(1)
    part = np.sort(prob, axis=1)
    s = {"top probability": prob.max(1), "margin": part[:, -1] - part[:, -2],
         "second opinion": (prob.max(1) + second[np.arange(len(top)), top]) / 2}
    s["patient context"] = (s["top probability"] + patient_mean(s["top probability"], patient)) / 2
    s["second opinion + patient context"] = (s["second opinion"] + patient_mean(s["second opinion"], patient)) / 2
    return s


def aurc(score, wrong):
    """Area under the risk-coverage curve: answer the most confident cases first and average the error so far."""
    w = wrong[np.argsort(-score, kind="stable")]
    return float((np.cumsum(w) / np.arange(1, len(w) + 1)).mean())


def patient_cut_off(score, wrong, patient, rng, budget=BUDGET, confidence=CONFIDENCE, resamples=300, batch=None):
    """The lowest cut-off whose error among answered-alone cases stays under the budget when whole patients are resampled.
    `batch`: draw this many patients per resample (ledger entry 13), so the promise is about a batch of that size."""
    ids, inv = np.unique(patient, return_inverse=True)
    weights = rng.multinomial(batch or len(ids), np.ones(len(ids)) / len(ids), size=resamples)
    best = np.inf
    for asked in np.arange(0.95, -0.001, -0.01):
        t = np.quantile(score, max(asked, 0.0))
        act = score >= t
        acted = np.bincount(inv, weights=act, minlength=len(ids))
        errors = np.bincount(inv, weights=act * wrong, minlength=len(ids))
        if np.quantile((weights @ errors) / np.maximum(weights @ acted, 1), confidence) > budget:
            break
        best = t
    return float(best)


LEVELS = [0.90, 0.95, 0.975, 0.99, 0.995]
CHOSEN_LEVELS = []  # filled by calibrated_cut_off, for the report


def cut_offs_at_levels(score, wrong, patient, rng, levels=LEVELS, resamples=300, batch=NEW_PATIENTS, budget=BUDGET):
    """Entry 13's batch test at several strictness levels in one pass. Returns {level: cut-off}."""
    ids, inv = np.unique(patient, return_inverse=True)
    weights = rng.multinomial(batch, np.ones(len(ids)) / len(ids), size=resamples).astype(float)
    best, alive = {q: np.inf for q in levels}, sorted(levels)
    for t in np.quantile(score, np.clip(np.arange(0.95, -0.001, -0.01), 0, 1)):
        act = score >= t
        acted = np.bincount(inv, weights=act, minlength=len(ids))
        errors = np.bincount(inv, weights=act * wrong, minlength=len(ids))
        bounds = np.quantile((weights @ errors) / np.maximum(weights @ acted, 1), alive)
        alive = [q for q, b in zip(alive, bounds) if b <= budget]  # a level that fails once is out for good
        for q in alive:
            best[q] = float(t)
        if not alive:
            break
    return best


def calibrated_cut_off(score, wrong, patient, rng, inner_splits=100, target=0.90, levels=None):
    """Ledger entry 16: the method audits itself on its own calibration patients, picks the lowest strictness level
    that keeps the budget in `target` of the inner splits, and sets the cut-off at that level."""
    levels = levels or LEVELS
    ids, inv = np.unique(patient, return_inverse=True)
    kept = {q: 0 for q in levels}
    for _ in range(inner_splits):
        new = np.isin(inv, rng.choice(len(ids), NEW_PATIENTS, replace=False))
        cuts = cut_offs_at_levels(score[~new], wrong[~new], patient[~new], rng, levels=levels)
        for q, t in cuts.items():
            act = score[new] >= t
            kept[q] += (wrong[new][act].sum() / act.sum() if act.any() else 0.0) <= BUDGET
    level = next((q for q in levels if kept[q] / inner_splits >= target), levels[-1])
    CHOSEN_LEVELS.append(level)
    return cut_offs_at_levels(score, wrong, patient, rng, levels=[level], resamples=1000)[level]


STRICT_LEVELS = LEVELS + [0.999]


def strict_cut_off(score, wrong, patient, rng, inner_splits=100):
    """Ledger entry 17: the self-audit must keep the budget in 95% of its inner splits."""
    return calibrated_cut_off(score, wrong, patient, rng, inner_splits=inner_splits, target=0.95, levels=STRICT_LEVELS)


def strict():
    """Ledger entry 17: outer audit of the stricter self-audit, and whether harder batches are asked about more."""
    from scipy.stats import spearmanr
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    wrong = (prob.argmax(1) != y).astype(float)
    score = signals(prob, prob, patient)["patient context"]
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    rng = np.random.default_rng(0)
    [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(1000)]  # the same random draws as main(), so the 300 splits below are the same
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    rows = []
    for n in range(SPLITS):
        order = rng.permutation(ids)
        cal, new = np.concatenate([where[p] for p in order[NEW_PATIENTS:]]), np.concatenate([where[p] for p in order[:NEW_PATIENTS]])
        rows.append((cal, new))
    CHOSEN_LEVELS.clear()
    out = []
    for cal, new in rows:
        t = strict_cut_off(score[cal], wrong[cal], patient[cal], rng)
        act = score[new] >= t
        asked_cal = cal[score[cal] < t]
        lists = lists_for_asked(prob, y, patient, asked_cal, prob[new][~act]) if (~act).any() and len(asked_cal) else None
        out.append([wrong[new][act].sum() / act.sum() if act.any() else 0.0, 1 - act.mean(),
                    lists[np.arange((~act).sum()), y[new][~act]].mean() if lists is not None else np.nan, wrong[new].mean()])
        if len(out) % 50 == 0:
            print(len(out), "splits", flush=True)
    a = np.array(out)
    rho = float(spearmanr(a[:, 3], a[:, 1])[0])
    draws = np.random.default_rng(1).integers(0, len(a), (1000, len(a)))
    result = {"budget_kept_in_share_of_splits": round(float((a[:, 0] <= BUDGET).mean()), 3),
              "mean_error_among_answered_alone": round(float(a[:, 0].mean()), 4),
              "error_5th_to_95th_percentile": [round(float(v), 4) for v in np.percentile(a[:, 0], [5, 95])],
              "worst_split_error": round(float(a[:, 0].max()), 4),
              "mean_asked_share": round(float(a[:, 1].mean()), 4),
              "mean_hit_rate_among_asked": round(float(np.nanmean(a[:, 2])), 4),
              "levels_chosen": {str(q): CHOSEN_LEVELS.count(q) for q in STRICT_LEVELS},
              "harder_batches_are_asked_more": {"rank_correlation_batch_error_vs_asked_share": round(rho, 3),
                                                "range95": range95([spearmanr(a[i, 3], a[i, 1])[0] for i in draws])}}
    result["pass"] = bool(result["budget_kept_in_share_of_splits"] >= 0.90)
    CHOSEN_LEVELS.clear()
    result["final_cut_off"] = {"signal": "patient context", "cut_off": round(strict_cut_off(score, wrong, patient, np.random.default_rng(0), inner_splits=300), 4),
                               "level_chosen": CHOSEN_LEVELS[-1]}
    (OUT / "result_strict.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def batch_cut_off(score, wrong, patient, rng):
    return patient_cut_off(score, wrong, patient, rng, resamples=1000, batch=NEW_PATIENTS)


def region_cut_off(score, wrong, patient, rng):
    return sl.act_cut_off(score, wrong.astype(bool), BUDGET)  # entry 7: counts regions


def lists_for_asked(prob, y, patient, asked_cal, prob_new):
    method = "LAC, one threshold per type"
    return sl.short_lists(method, prob_new, sl.calibrate(method, prob[asked_cal], y[asked_cal], sl.PROMISE, patient[asked_cal]))


def audit(score, cut_off, prob, y, patient, splits, rng):
    wrong = (prob.argmax(1) != y).astype(float)
    rows = []
    for cal, new in splits:
        t = cut_off(score[cal], wrong[cal], patient[cal], rng)
        act = score[new] >= t
        asked_cal = cal[score[cal] < t]
        hit = np.nan
        if (~act).any() and len(asked_cal):
            lists = lists_for_asked(prob, y, patient, asked_cal, prob[new][~act])
            hit = lists[np.arange((~act).sum()), y[new][~act]].mean()
        rows.append([wrong[new][act].sum() / act.sum() if act.any() else 0.0, 1 - act.mean(), hit])
    a = np.array(rows)
    return {"budget_kept_in_share_of_splits": round(float((a[:, 0] <= BUDGET).mean()), 3),
            "mean_error_among_answered_alone": round(float(a[:, 0].mean()), 4),
            "error_5th_to_95th_percentile": [round(float(v), 4) for v in np.percentile(a[:, 0], [5, 95])],
            "mean_asked_share": round(float(a[:, 1].mean()), 4),
            "mean_hit_rate_among_asked": round(float(np.nanmean(a[:, 2])), 4)}


def main():
    from scipy.stats import spearmanr
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    second = second_opinion_prob(y, folds)
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    wrong = (prob.argmax(1) != y).astype(float)
    sig = signals(prob, second, patient)
    ids = np.unique(patient)
    where = {p: np.flatnonzero(patient == p) for p in ids}
    rng = np.random.default_rng(0)
    result = {"patients": int(len(ids)), "regions": len(y)}

    # (c) which confidence signal ranks the cases best?
    draws = [np.concatenate([where[p] for p in rng.choice(ids, len(ids))]) for _ in range(1000)]
    base = "top probability"
    result["signals"] = {}
    for name, s in sig.items():
        entry = {"aurc": round(aurc(s, wrong), 4)}
        if name != base:
            diff = []
            for i in draws:
                diff.append(aurc(s[i], wrong[i]) - aurc(sig[base][i], wrong[i]))
            entry.update({"minus_top_probability": round(entry["aurc"] - aurc(sig[base], wrong), 4), "range95": range95(diff),
                          "adopted": bool(np.percentile(diff, 97.5) < 0)})
        result["signals"][name] = entry
    adopted = [k for k, v in result["signals"].items() if v.get("adopted")]
    chosen = min(adopted, key=lambda k: result["signals"][k]["aurc"]) if adopted else base
    result["chosen_signal"] = chosen

    # does a patient's average confidence (no labels) say how hard that patient is?
    conf_p = np.array([sig[base][where[p]].mean() for p in ids])
    err_p = np.array([wrong[where[p]].mean() for p in ids])
    boot = [spearmanr(conf_p[i], err_p[i])[0] for i in (rng.integers(0, len(ids), len(ids)) for _ in range(1000))]
    result["patient_confidence_vs_patient_error"] = {"rank_correlation": round(float(spearmanr(conf_p, err_p)[0]), 3), "range95": range95(boot)}

    # (a) and (b): the audit over random splits of patients
    splits = []
    for _ in range(SPLITS):
        order = rng.permutation(ids)
        splits.append((np.concatenate([where[p] for p in order[NEW_PATIENTS:]]), np.concatenate([where[p] for p in order[:NEW_PATIENTS]])))
    result["audit"] = {}
    for sname in dict.fromkeys([base, chosen]):
        for mname, fn in (("region-counting cut-off (entry 7)", region_cut_off), ("patient-level cut-off", patient_cut_off)):
            result["audit"][f"{sname} | {mname}"] = audit(sig[sname], fn, prob, y, patient, splits, rng)
            print(sname, "|", mname, result["audit"][f"{sname} | {mname}"], flush=True)
    if "nested" in sys.argv:  # ledger entry 16
        key = f"{chosen} | self-audited cut-off (entry 16)"
        result["audit"][key] = audit(sig[chosen], calibrated_cut_off, prob, y, patient, splits, rng)
        result["audit"][key]["levels_chosen"] = {str(q): CHOSEN_LEVELS.count(q) for q in LEVELS}
        print(key, result["audit"][key], flush=True)
    elif "batch" in sys.argv:  # ledger entry 13
        for sname in dict.fromkeys([base, chosen]):
            result["audit"][f"{sname} | cut-off for a batch of {NEW_PATIENTS} patients"] = audit(sig[sname], batch_cut_off, prob, y, patient, splits, rng)
            print(sname, "| batch", result["audit"][f"{sname} | cut-off for a batch of {NEW_PATIENTS} patients"], flush=True)
        key = f"{chosen} | cut-off for a batch of {NEW_PATIENTS} patients"
    else:
        key = f"{chosen} | patient-level cut-off"
    result["pass"] = bool(result["audit"][key]["budget_kept_in_share_of_splits"] >= 0.88)

    if "nested" in sys.argv:  # the long harsh-scenario report is not repeated for entry 16
        CHOSEN_LEVELS.clear()
        result["final_cut_off"] = {"signal": chosen, "self_audited_cut_off": round(calibrated_cut_off(sig[chosen], wrong, patient, np.random.default_rng(0), inner_splits=300), 4),
                                   "level_chosen": CHOSEN_LEVELS[-1]}
        (OUT / "result_nested.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("audit", "signals")}, indent=2))
        return
    # the harsh scenario of entry 11: easy history, hard arrivals (reported only)
    best_first = ids[np.argsort(err_p)]
    harsh = []
    for _ in range(100):
        harsh.append((np.concatenate([where[p] for p in rng.choice(best_first[:60], 40, replace=False)]), np.concatenate([where[p] for p in best_first[60:]])))
    result["harsh_scenario"] = {f"{s} | {m}": audit(sig[s], fn, prob, y, patient, harsh, rng)
                                for s in dict.fromkeys([base, chosen])
                                for m, fn in (("region-counting cut-off (entry 7)", region_cut_off), ("patient-level cut-off", patient_cut_off))}

    # the cut-off the system would use from now on, from all 121 patients
    result["final_cut_off"] = {"signal": chosen, "cut_off": round(patient_cut_off(sig[chosen], wrong, patient, rng), 4),
                               "cut_off_for_a_batch_of_30": round(batch_cut_off(sig[chosen], wrong, patient, rng), 4),
                               "region_counting_cut_off_same_signal": round(region_cut_off(sig[chosen], wrong, patient, rng), 4)}
    # self-checks: perfect ranking beats random ranking; a stricter confidence can only raise the cut-off
    w = np.array([0.0] * 80 + [1.0] * 20)
    assert aurc(-w, w) < aurc(np.random.default_rng(1).random(100), w) < aurc(w, w)
    r1, r2 = np.random.default_rng(2), np.random.default_rng(2)
    assert patient_cut_off(sig[base], wrong, patient, r1, confidence=0.99) >= patient_cut_off(sig[base], wrong, patient, r2, confidence=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "audit"}, indent=2))


def freeze():
    """Ledger entries 13 and 14: save the system that is used from now on (signal, cut-off, list thresholds)."""
    prob, y, patient, folds, vprob, vy, vpatient = sl.load()
    prob, y, patient = np.vstack([prob, vprob]), np.concatenate([y, vy]), np.concatenate([patient, vpatient])
    wrong = (prob.argmax(1) != y).astype(float)
    score = signals(prob, prob, patient)["patient context"]  # this signal uses the final classifier only
    nested = OUT / "result_nested.json"
    if "v4" in sys.argv or "v5" in sys.argv:  # ledger entry 17: the stricter self-audit
        cut = json.loads((OUT / "result_strict.json").read_text())["final_cut_off"]["cut_off"]
    elif "v3" in sys.argv:  # ledger entry 16: the self-audited cut-off
        cut = json.loads(nested.read_text())["final_cut_off"]["self_audited_cut_off"]
    else:                 # ledger entry 13
        cut = patient_cut_off(score, wrong, patient, np.random.default_rng(0), resamples=2000, batch=NEW_PATIENTS)
    asked = score < cut
    guard = None
    if "v5" in sys.argv:  # ledger entry 36: also ask whenever ADH is a live possibility
        adh = y == sl.TYPES.index("ADH")
        guard = 1 - sl.threshold(1 - prob[adh, sl.TYPES.index("ADH")], sl.PROMISE, len(np.unique(patient[adh])))
        asked |= prob[:, sl.TYPES.index("ADH")] >= guard
    method = "LAC, one threshold per type"
    thresholds = sl.calibrate(method, prob[asked], y[asked], sl.PROMISE, patient[asked])
    system = {"signal": "patient context: (top probability + its mean over the patient's regions) / 2", "cut_off": cut,
              "list_method": method, "list_thresholds": dict(zip(sl.TYPES, thresholds.tolist())),
              "calibration": {"patients": int(len(np.unique(patient))), "regions": len(y), "asked_share": round(float(asked.mean()), 4),
                              "error_among_answered_alone": round(float(wrong[~asked].mean()), 4)}}
    if guard is not None:
        system["adh_guard"] = guard
        system["adh_guard_rule"] = "ask whenever the ADH probability reaches this value (at least 90% of truly ADH regions do, patients counted)"
    (OUT / ("system_v5.json" if "v5" in sys.argv else "system_v4.json" if "v4" in sys.argv else "system_v3.json" if "v3" in sys.argv else "system_v2.json")).write_text(json.dumps(system, indent=2))
    print(json.dumps(system, indent=2))


def apply_system(prob, patient):
    """(answer alone, lists) for new regions under the frozen system."""
    newest = next(p for p in (OUT / "system_v5.json", OUT / "system_v4.json", OUT / "system_v3.json", OUT / "system_v2.json") if p.exists())  # the latest adopted entry
    system = json.loads(newest.read_text())
    top = prob.max(1)
    act = (top + patient_mean(top, patient)) / 2 >= system["cut_off"]
    if "adh_guard" in system:  # ledger entry 36
        act &= prob[:, sl.TYPES.index("ADH")] < system["adh_guard"]
    lists = sl.short_lists(system["list_method"], prob, np.array([system["list_thresholds"][t] for t in sl.TYPES]))
    return act, lists


if __name__ == "__main__":
    freeze() if "freeze" in sys.argv else strict() if "strict" in sys.argv else main()
