#!/usr/bin/env python3
"""The one pass over the locked test set (ledger entry 9), and a rehearsal that never touches it.

Usage (from the project folder):
  python -I tools/final_test.py rehearse   # validation plays the part of the test set; finds bugs first -> outputs/final/rehearsal.json
  python -I tools/final_test.py unlock     # the real pass, once -> outputs/final/test.json (refuses to run a second time)

Nothing is chosen here. The classifier (entry 4) and the two-step system (entry 8) are frozen; this script only measures.
"""
import csv
import json
import pathlib
import subprocess
import sys

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder off the path
import short_lists as sl
from train_classifier import FEATS, MODELS, ROOT, TYPES, patient_draws, range95, weighted_f1

OUT = ROOT / "outputs/final"
M1 = ROOT / "outputs/m1"
ATYPICAL = [3, 4]  # FEA, ADH


def split_rows(split):
    with open(ROOT / "data/regions.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def pooled(model, held, min_tissue):
    x = []
    for r in held:
        z = np.load(FEATS / model / (r["file"][:-4] + ".npz"))
        feats, keep = z["feats"].astype(np.float32), z["tissue"] >= min_tissue
        x.append(feats[keep].mean(0) if keep.any() else feats.mean(0))
    return np.array(x)


def run(split, use_validation_for_calibration):
    held = split_rows(split)
    y = np.array([int(r["label"]) for r in held])
    patient = np.array([int(r["patient"]) for r in held])
    rng = np.random.default_rng(0)
    draws = patient_draws(patient, rng)
    settings = json.loads((M1 / "cv.json").read_text())
    result = {"split": split, "regions": len(y), "patients": int(len(np.unique(patient))), "models": {}}

    # 1. the four classifiers
    pred, prob = {}, None
    for model in MODELS:
        p = joblib.load(M1 / f"classifier_{model}.joblib").predict_proba(pooled(model, held, settings[model]["min_tissue"]))
        pred[model] = p.argmax(1)
        result["models"][model] = {"kind": MODELS[model], "weighted_f1": round(weighted_f1(y, pred[model]), 4),
                                   "range95": range95([weighted_f1(y[i], pred[model][i]) for i in draws]),
                                   "macro_f1": round(f1_score(y, pred[model], average="macro"), 4),
                                   "f1_by_type": dict(zip(TYPES, (round(float(v), 3) for v in f1_score(y, pred[model], average=None))))}
    # the final classifier (entry 10): UNI's tile average next to UNI's view of the whole region
    x = np.hstack([pooled("uni", held, sl.ENGINE["min_tissue"]), np.load(M1 / "thunder" / f"uni_{split}.npy")])
    prob = joblib.load(ROOT / "outputs/m2/engine.joblib").predict_proba(x)
    top = prob.argmax(1)
    result["engine"] = {"weighted_f1": round(weighted_f1(y, top), 4), "range95": range95([weighted_f1(y[i], top[i]) for i in draws]),
                        "macro_f1": round(f1_score(y, top, average="macro"), 4),
                        "f1_by_type": dict(zip(TYPES, (round(float(v), 3) for v in f1_score(y, top, average=None))))}
    a, b = "uni", "vit_l_imagenet"  # best tissue model and best general model in cross-validation (entry 4)
    diff = [weighted_f1(y[i], pred[a][i]) - weighted_f1(y[i], pred[b][i]) for i in draws]
    result["tissue_vs_general"] = {"pair": [a, b], "difference": round(weighted_f1(y, pred[a]) - weighted_f1(y, pred[b]), 4),
                                   "range95": range95(diff), "range_excludes_zero": bool(np.percentile(diff, 2.5) > 0)}
    result["confusion_rows_true_cols_predicted"] = confusion_matrix(y, prob.argmax(1)).tolist()
    by_type = np.array([f1_score(y[i], prob.argmax(1)[i], average=None, labels=range(7), zero_division=0) for i in draws[:1000]])
    result["engine_f1_by_type_range95"] = {t: range95(by_type[:, k]) for k, t in enumerate(TYPES)}

    # 2. calibration cases: training patients (scored out of fold), plus validation in the real pass
    cprob, cy, cpatient, _, vprob, vy, vpatient = sl.load()
    if use_validation_for_calibration:
        cprob, cy, cpatient = np.vstack([cprob, vprob]), np.concatenate([cy, vy]), np.concatenate([cpatient, vpatient])
    assert not set(cpatient) & set(patient)  # no calibration patient is among the measured patients
    result["calibration"] = {"regions": len(cy), "patients": int(len(np.unique(cpatient)))}

    # 3. the frozen two-step system
    act, lists, cut = sl.two_step(cprob, cy, prob, sl.BUDGET, sl.PROMISE, patient_cal=cpatient)
    system = sl.measure_two_step(act, lists, y, patient, prob, rng)
    wrong, hit = prob.argmax(1) != y, lists[np.arange(len(y)), y]
    missed = (wrong & act) | (~hit & ~act)
    aty = np.isin(y, ATYPICAL)
    result["system"] = {"cut_off": round(cut, 4), "budget": sl.BUDGET, "promise": sl.PROMISE, **system,
                        "missed_overall_range95": range95([float(missed[i].mean()) for i in draws])}
    result["like_olsson"] = {
        "all": {"n": len(y), "error_without": round(float(wrong.mean()), 4), "error_with": round(float(missed.mean()), 4), "flagged": round(float((~act).mean()), 4)},
        "atypical (FEA, ADH)": {"n": int(aty.sum()), "error_without": round(float(wrong[aty].mean()), 4),
                                "error_with": round(float(missed[aty].mean()), 4), "flagged": round(float((~act)[aty].mean()), 4)}}

    # 4. other ways to decide "act or ask", measured on the same cases
    ways = {"two-step system (frozen)": {"asked_share": system["asked_share"], "error_among_acted": system["error_among_acted"]}}
    one = sl.short_lists("LAC, one threshold per type", prob, sl.calibrate("LAC, one threshold per type", cprob, cy, sl.PROMISE))
    m = sl.measure(one, y, patient, prob, rng)
    ways["act when the list has one type (entry 6)"] = {"asked_share": m["asked_share"], "error_among_acted": m["error_among_acted"]}
    for budget in (0.05, 0.10):
        sel = sl.conformal_selection(cprob, cy, prob, y, budget)
        ways[f"conformal selection, error budget {budget:.0%}"] = {
            "asked_share": round(float(1 - sel.mean()), 4), "error_among_acted": round(float((wrong & sel).sum() / max(sel.sum(), 1)), 4)}
    result["act_or_ask"] = ways

    # 5. the two dials
    result["budget_dial"], result["promise_dial"] = {}, {}
    keys = ("asked_share", "error_among_acted", "hit_rate_among_asked", "list_length_among_asked")
    for budget in sl.BUDGETS:
        m = sl.measure_two_step(*sl.two_step(cprob, cy, prob, budget, sl.PROMISE, patient_cal=cpatient)[:2], y, patient, prob, rng)
        result["budget_dial"][f"{budget:.0%}"] = {k: m[k] for k in keys}
    for promise in sl.DIAL:
        m = sl.measure_two_step(*sl.two_step(cprob, cy, prob, sl.BUDGET, promise, patient_cal=cpatient)[:2], y, patient, prob, rng)
        result["promise_dial"][f"{promise:.0%}"] = {k: m[k] for k in keys}

    # 6. patient by patient: where the averages hide a badly served person
    per = []
    for p in np.unique(patient):
        i = patient == p
        per.append({"patient": int(p), "regions": int(i.sum()), "acted": int(act[i].sum()), "acted_wrong": int((wrong & act)[i].sum()),
                    "asked": int((~act)[i].sum()), "asked_not_on_list": int((~hit & ~act)[i].sum())})
    result["patients_table"] = per
    result["patients_with_over_a_third_missed"] = sum((r["acted_wrong"] + r["asked_not_on_list"]) > r["regions"] / 3 for r in per)

    # 7. one row per region, for the demo
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"regions_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "patient", "label", "top_type", "top_probability", "decision", "short_list", *[f"p_{t}" for t in TYPES]])
        for n, r in enumerate(held):
            w.writerow([r["file"], r["patient"], TYPES[y[n]], TYPES[prob[n].argmax()], f"{prob[n].max():.4f}", "act" if act[n] else "ask",
                        "" if act[n] else " ".join(t for k, t in enumerate(TYPES) if lists[n, k]), *(f"{v:.4f}" for v in prob[n])])
    return result


def thunder(final):
    out = {}
    for model in ("uni", "hoptimus0"):
        if not (M1 / "thunder" / f"{model}_test.npy").exists():
            out[model] = "embeddings not ready"
            continue
        cmd = [sys.executable, "-I", str(ROOT / "tools/reproduce_thunder.py"), "probe", model] + (["--final-test"] if final else [])
        subprocess.run(cmd, check=True, capture_output=True)
        out[model] = json.loads((M1 / "thunder" / f"result_{model}_{'test' if final else 'val'}.json").read_text())
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rehearse":
        result = run("val", use_validation_for_calibration=False)
        result["thunder"] = thunder(final=False)
        dest = OUT / "rehearsal.json"
    elif mode == "unlock":
        dest = OUT / "test.json"
        if dest.exists():
            sys.exit("The test set has already been used once (outputs/final/test.json exists). Not running again.")
        result = run("test", use_validation_for_calibration=True)
        result["thunder"] = thunder(final=True)
    else:
        sys.exit(__doc__)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "patients_table"}, indent=2))
