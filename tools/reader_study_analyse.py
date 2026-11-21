#!/usr/bin/env python3
"""Reader study: analyse the saved answer files.

Usage (from the project folder):
  python -I tools/reader_study_analyse.py [folder with reader-*.json]   # default outputs/reader_study/responses -> analysis.json there
  python -I tools/reader_study_analyse.py selfcheck                     # simulated readers, checks the arithmetic

Reported: agreement with the agreed type without and with the system (per reader and together, cases resampled for the range),
the system alone on the same cases, agreement between readers (Fleiss' kappa), ADH on its own, and how often a reader moved
from right to wrong after a wrong hint from the system.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
KIT = ROOT / "outputs/reader_study"
TYPES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
GRAY = ["UDH", "FEA", "ADH", "DCIS"]


def fleiss(table):
    """table: cases x types, counts of readers choosing each type."""
    n = table.sum(1)[0]
    p = table.sum(0) / table.sum()
    agree = ((table ** 2).sum(1) - n) / (n * (n - 1))
    return float((agree.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()))


def analyse(readers, key, cases):
    ids = [c["id"] for c in cases]
    ai = {c["id"]: c["ai"] for c in cases}
    truth = np.array([key[i]["label"] for i in ids])
    gray = np.isin(truth, GRAY)
    ans = {}
    for r in readers:
        a = {(x["case"], x["part"]): x["type"] for x in r["answers"]}
        if all((i, p) in a for i in ids for p in (1, 2)):
            ans[r["reader"]] = np.array([[a[(i, p)] for i in ids] for p in (1, 2)])
    names = sorted(ans)
    out = {"readers": names, "cases": len(ids), "gray_zone_cases": int(gray.sum())}
    if not names:
        return out
    right = np.array([ans[n] == truth for n in names])  # readers x part x cases
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(ids), (2000, len(ids)))

    def block(mask):
        w, a = right[:, 0][:, mask].mean(), right[:, 1][:, mask].mean()
        d = right[:, 1][:, mask].mean(0) - right[:, 0][:, mask].mean(0)
        idx = rng.integers(0, mask.sum(), (2000, mask.sum()))
        ai_right = np.array([ai[i]["answer"] == t for i, t in zip(ids, truth)])[mask]
        ai_list = np.array([t in ai[i]["list"] for i, t in zip(ids, truth)])[mask]
        return {"cases": int(mask.sum()), "agreement_without_system": round(float(w), 3), "agreement_with_system": round(float(a), 3), "gain": round(float(a - w), 3),
                "gain_range95": [round(float(v), 3) for v in np.percentile(d[idx].mean(1), [2.5, 97.5])],
                "system_top_answer_right": round(float(ai_right.mean()), 3), "system_list_holds_agreed_type": round(float(ai_list.mean()), 3),
                "per_reader": {n: [round(float(right[k, 0][mask].mean()), 3), round(float(right[k, 1][mask].mean()), 3)] for k, n in enumerate(names)}}

    out["all_cases"], out["gray_zone"], out["adh_only"] = block(np.ones(len(ids), bool)), block(gray), block(truth == "ADH")
    if len(names) > 1:
        for part, label in ((0, "without_system"), (1, "with_system")):
            table = np.array([[sum(ans[n][part][j] == t for n in names) for t in TYPES] for j in range(len(ids))], float)
            out[f"fleiss_kappa_{label}"] = round(fleiss(table), 3)
            out[f"fleiss_kappa_{label}_gray_zone"] = round(fleiss(table[gray]), 3)
    ai_wrong = np.array([ai[i]["answer"] != t for i, t in zip(ids, truth)])
    moved = right[:, 0] & ~right[:, 1]
    out["over_reliance"] = {"right_then_wrong_after_a_wrong_system_answer": int((moved & ai_wrong).sum()),
                            "readings_right_without_system_where_system_was_wrong": int((right[:, 0] & ai_wrong).sum()),
                            "wrong_then_right": int((~right[:, 0] & right[:, 1]).sum()), "right_then_wrong": int(moved.sum())}
    return out


def selfcheck():
    cases = json.loads((KIT / "cases.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    key = json.loads((KIT / "key.json").read_text())
    rng = np.random.default_rng(0)
    readers = []
    for name, skill in (("S1", 0.6), ("S2", 0.7)):
        answers = []
        for part, boost in ((1, 0.0), (2, 0.15)):
            for c in cases:
                t = key[c["id"]]["label"] if rng.random() < skill + boost else str(rng.choice([x for x in TYPES if x != key[c["id"]]["label"]]))
                answers.append({"case": c["id"], "part": part, "type": t, "sure": 1, "seconds": 20})
        readers.append({"reader": name, "answers": answers})
    r = analyse(readers, key, cases)
    assert r["readers"] == ["S1", "S2"] and 0.05 < r["all_cases"]["gain"] < 0.25, r["all_cases"]
    assert abs(fleiss(np.array([[2, 0], [2, 0], [0, 2], [0, 2]], float)) - 1) < 1e-9
    print("selfcheck ok:", {k: r["all_cases"][k] for k in ("agreement_without_system", "agreement_with_system", "gain", "gain_range95")}, "kappa", r["fleiss_kappa_without_system"], r["fleiss_kappa_with_system"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        return selfcheck()
    folder = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else KIT / "responses"
    cases = json.loads((KIT / "cases.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    key = json.loads((KIT / "key.json").read_text())
    readers = [json.loads(f.read_text()) for f in sorted(folder.glob("reader-*.json"))]
    result = analyse(readers, key, cases)
    (folder / "analysis.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
