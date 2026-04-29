#!/usr/bin/env python3
"""Bridge: val_results.json → intermediate.csv for final_stage.py."""
import argparse
import ast
import json
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-results-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--patient-id", required=True)
    p.add_argument("--exam-id", required=True)
    p.add_argument("--classifier-csv", default=None,
                   help="Optional CSV with zero_shot_per_image column; filtered by patient_id/exam_id")
    args = p.parse_args()

    with open(args.val_results_json) as f:
        results = json.load(f)

    if not results:
        raise ValueError("val_results.json is empty")

    element = results[0]
    generated_report = element["conversations_out"][0]["answer"].strip()

    zero_shot = []
    if args.classifier_csv:
        df_cls = pd.read_csv(args.classifier_csv)
        mask = (df_cls["patient_id"].astype(str) == str(args.patient_id)) & \
               (df_cls["exam_id"].astype(str) == str(args.exam_id))
        rows = df_cls[mask]
        if not rows.empty and "zero_shot_per_image" in rows.columns:
            raw = rows.iloc[0]["zero_shot_per_image"]
            if isinstance(raw, str):
                try:
                    zero_shot = ast.literal_eval(raw)
                except Exception:
                    zero_shot = []
            elif isinstance(raw, list):
                zero_shot = raw

    row = {
        "patient_id": args.patient_id,
        "exam_id": args.exam_id,
        "generated_report": generated_report,
        "zero_shot_per_image": str(zero_shot),
        "id": element.get("id", f"job_{args.exam_id}"),
        "dataset": element.get("dataset", "BU"),
        "image": element.get("image", ""),
    }
    pd.DataFrame([row]).to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
