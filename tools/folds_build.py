#!/usr/bin/env python3
"""Write `outputs/folds.json` once from the live derivation, then replay-check every recorded split.

Usage (from the project folder, with the region table in place):
    python -I tools/folds_build.py

Shapes and rows come from `tools/folds_dump.py`, which drives every call site, so nothing is
hand-listed here. A shape already in the file is only read; a missing one is derived and appended.
Verification is switched on for the run: every replay is compared to a fresh derivation and an
assert fires on any difference.
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import folds
import folds_dump


def main():
    before = len(folds.load())
    os.environ["FOLDS_VERIFY"] = "1"  # folds.split() re-derives and compares from here on
    folds_dump.main(str(pathlib.Path(tempfile.mkdtemp()) / "dump.json"))
    os.environ.pop("FOLDS_VERIFY")
    print(json.dumps({"records_before": before, "records_after": len(folds.load()), "path": str(folds.PATH),
                      "bytes": folds.PATH.stat().st_size, "n_splits": sorted({r["n_splits"] for r in folds.load().values()})}))


if __name__ == "__main__":
    main()
