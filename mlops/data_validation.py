"""Data validation and profiling utilities using Great Expectations.

These helpers are light-weight wrappers around Great Expectations for
validating training data and capturing baseline statistics for drift detection.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from great_expectations.dataset import PandasDataset


class ActivityDataset(PandasDataset):
    """Convenience dataset with a couple of helper expectations."""

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "ActivityDataset":
        dataset = ActivityDataset(df.copy())
        dataset.set_default_expectation_argument("catch_exceptions", True)
        dataset.set_default_expectation_argument("result_format", "SUMMARY")
        return dataset


def validate_training_dataframe(df: pd.DataFrame, min_rows: int = 200) -> Dict:
    """Run a small suite of expectations against the synthetic training dataset.

    Raises a ValueError if validation fails. Returns the Great Expectations
    statistics to surface in logs or reports.
    """

    dataset = ActivityDataset.from_dataframe(df)
    dataset.expect_table_row_count_to_be_between(min_value=min_rows, max_value=None)

    if "label" in df.columns:
        dataset.expect_column_values_to_be_in_set("label", {0, 1, 2, 3})

    # Enforce reasonable bounds for request duration if present.
    if "request_duration" in df.columns:
        dataset.expect_column_values_to_be_between("request_duration", min_value=0.0, max_value=60.0)

    # All numerical feature columns should be finite and non-null.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        dataset.expect_column_values_to_not_be_null(col)
        dataset.expect_column_values_to_be_between(col, min_value=-1e6, max_value=1e6)

    result = dataset.validate()
    if not result.get("success", False):
        raise ValueError(
            "Great Expectations validation failed",
            {"statistics": result.get("statistics"), "results": result.get("results", [])[:5]},
        )
    return result.get("statistics", {})


def _numeric_stats(series: pd.Series) -> Dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p95": 0.0}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "p05": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
    }


def save_baseline_profile(df: pd.DataFrame, output_path: str | Path, feature_columns: Iterable[str] | None = None) -> Dict:
    """Persist baseline statistics for numeric columns used in drift detection."""

    feature_columns = list(feature_columns or df.columns)
    numeric_cols = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    profile = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "columns": {col: _numeric_stats(df[col]) for col in numeric_cols},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_baseline_profile(profile_path: str | Path) -> Dict:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline profile not found: {profile_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_with_profile(
    df: pd.DataFrame,
    profile_path: str | Path,
    tolerance: float = 3.0,
) -> Dict:
    """Compare recent data to a baseline profile.

    Computes z-score style drift using the stored mean/std. Any column where the
    absolute z-score exceeds ``tolerance`` is flagged.
    """

    profile = load_baseline_profile(profile_path)
    numeric_cols = profile.get("columns", {})
    alerts: List[Dict] = []
    summary: Dict[str, Dict[str, float]] = {}

    for col, stats in numeric_cols.items():
        if col not in df.columns:
            alerts.append({"column": col, "reason": "missing_in_recent"})
            continue
        recent_stats = _numeric_stats(df[col])
        baseline_std = stats.get("std", 0.0) or 1e-6
        z_score = abs(recent_stats["mean"] - stats.get("mean", 0.0)) / baseline_std
        summary[col] = {"baseline_mean": stats.get("mean", 0.0), "recent_mean": recent_stats["mean"], "z_score": z_score}
        if z_score > tolerance:
            alerts.append({"column": col, "reason": "mean_shift", "z_score": z_score})
    return {"success": not alerts, "alerts": alerts, "summary": summary}
