"""Simple drift monitoring CLI.

Usage:
    python -m mlops.drift_monitor --baseline mlops/baselines/training_profile.json --input recent.csv

If drift is detected, the script exits with a non-zero code so it can be used in CI/CD gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mlops.data_validation import compare_with_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect drift against baseline profile")
    parser.add_argument("--baseline", required=True, help="Path to baseline profile JSON")
    parser.add_argument("--input", required=True, help="Path to recent dataset (CSV)")
    parser.add_argument("--output", help="Optional path to write drift report JSON")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Z-score tolerance before flagging drift")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    report = compare_with_profile(df, args.baseline, tolerance=args.tolerance)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
