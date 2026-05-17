#!/usr/bin/env python3
"""Regression check: optional classifier CSV is merged into intermediate.csv (same as pipeline bridge).

Needs Python with pandas (same as json_to_csv.py), e.g. on SCC:
  module load python3/3.9.4 && cd backend && python3 test_classifier_csv_bridge.py
"""
import ast
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_JSON_TO_CSV = Path(__file__).resolve().parent / "json_to_csv.py"


def _write_minimal_val_results(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "job_test",
                    "dataset": "BU",
                    "image": "[]",
                    "conversations_out": [
                        {"answer": "Synthetic preliminary report for classifier bridge test."}
                    ],
                }
            ],
            indent=2,
        )
    )


def _read_zero_shot_column(csv_path: Path) -> str:
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if len(rows) != 1:
        raise ValueError("expected 1 row in %s, got %s" % (csv_path, len(rows)))
    return rows[0]["zero_shot_per_image"]


def main():
    try:
        import pandas  # noqa: F401 — json_to_csv.py requires it
    except ImportError:
        print("pandas required (e.g. module load python3/3.9.4)", file=sys.stderr)
        return 2

    patient_id = "P_bridge_test"
    exam_id = "E_bridge_test"
    zs = [0.1, 0.2, 0.3, 0.4]

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        val_json = tdir / "val_results.json"
        cls_csv = tdir / "classifier.csv"
        out_csv = tdir / "intermediate.csv"

        _write_minimal_val_results(val_json)
        with open(cls_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["patient_id", "exam_id", "zero_shot_per_image", "noise"],
            )
            w.writeheader()
            w.writerow(
                {
                    "patient_id": patient_id,
                    "exam_id": exam_id,
                    "zero_shot_per_image": str(zs),
                    "noise": "ignored",
                }
            )

        r = subprocess.run(
            [
                sys.executable,
                str(_JSON_TO_CSV),
                "--val-results-json",
                str(val_json),
                "--output-csv",
                str(out_csv),
                "--patient-id",
                patient_id,
                "--exam-id",
                exam_id,
                "--classifier-csv",
                str(cls_csv),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("json_to_csv failed:", r.stderr or r.stdout, file=sys.stderr)
            return 1

        got = _read_zero_shot_column(out_csv)
        try:
            parsed = ast.literal_eval(str(got))
        except (ValueError, SyntaxError):
            print("could not parse zero_shot_per_image:", got, file=sys.stderr)
            return 1
        if parsed != zs:
            print("expected zero_shot", zs, "got", parsed, file=sys.stderr)
            return 1

        out2 = tdir / "intermediate_nocsv.csv"
        r2 = subprocess.run(
            [
                sys.executable,
                str(_JSON_TO_CSV),
                "--val-results-json",
                str(val_json),
                "--output-csv",
                str(out2),
                "--patient-id",
                patient_id,
                "--exam-id",
                exam_id,
            ],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            print("json_to_csv (no classifier) failed:", r2.stderr, file=sys.stderr)
            return 1
        z2 = _read_zero_shot_column(out2)
        try:
            parsed2 = ast.literal_eval(str(z2))
        except (ValueError, SyntaxError):
            print("could not parse no-csv column:", z2, file=sys.stderr)
            return 1
        if parsed2 != []:
            print("expected empty list without CSV, got", parsed2, file=sys.stderr)
            return 1

    print("OK: classifier CSV merge and no-CSV baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
