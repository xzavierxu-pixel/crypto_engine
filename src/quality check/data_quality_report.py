from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_builder import infer_feature_columns


def _select_columns(
    df: pd.DataFrame,
    *,
    model_features_only: bool,
) -> pd.DataFrame:
    if not model_features_only:
        return df
    feature_columns = infer_feature_columns(df)
    return df.loc[:, feature_columns].copy()


def _build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = ((numeric_df == np.inf) | (numeric_df == -np.inf)).sum() if not numeric_df.empty else pd.Series(dtype="int64")
    null_pct = (null_counts / len(df)) * 100 if len(df) > 0 else null_counts * 0
    return pd.DataFrame(
        {
            "column": df.columns,
            "null_count": null_counts.values,
            "inf_count": [int(inf_counts.get(column, 0)) for column in df.columns],
            "null_pct": null_pct.values,
        }
    ).sort_values(["null_pct", "inf_count", "column"], ascending=[False, False, True])


def analyze_frame(df: pd.DataFrame, name: str) -> dict[str, Any]:
    duplicate_groups: list[list[str]] = []
    if df.shape[1] < 2000:
        exact_duplicates: dict[tuple[Any, ...], list[str]] = {}
        for column in df.columns:
            try:
                signature = tuple(df[column].tolist())
            except TypeError:
                continue
            exact_duplicates.setdefault(signature, []).append(column)
        duplicate_groups = [columns for columns in exact_duplicates.values() if len(columns) > 1]

    missing_report = _build_missing_report(df)
    numeric_df = df.select_dtypes(include=[np.number])
    constant_features: list[str] = []
    low_variance_features: list[str] = []
    if not numeric_df.empty:
        nunique = numeric_df.nunique(dropna=False)
        constant_features = nunique[nunique <= 1].index.tolist()
        std_dev = numeric_df.std(ddof=0)
        low_variance_features = std_dev[(std_dev > 0) & (std_dev < 1e-6)].index.tolist()

    categorical_df = df.select_dtypes(exclude=[np.number])
    high_cardinality: dict[str, int] = {}
    highly_imbalanced: dict[str, float] = {}
    if not categorical_df.empty:
        cardinality = categorical_df.nunique(dropna=False).sort_values(ascending=False)
        high_cardinality = {str(column): int(value) for column, value in cardinality[cardinality > 50].items()}
        for column in categorical_df.columns:
            if categorical_df[column].empty:
                continue
            top_freq = categorical_df[column].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > 0.99:
                highly_imbalanced[str(column)] = float(top_freq)

    return {
        "name": name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_row_ratio": float(df.duplicated().mean()) if len(df) else 0.0,
        "duplicate_column_groups": duplicate_groups,
        "duplicate_column_count": int(sum(len(group) - 1 for group in duplicate_groups)),
        "missing_or_inf_columns": missing_report[
            (missing_report["null_count"] > 0) | (missing_report["inf_count"] > 0)
        ].to_dict(orient="records"),
        "constant_features": constant_features,
        "low_variance_features": low_variance_features,
        "high_cardinality_features": high_cardinality,
        "highly_imbalanced_categorical": highly_imbalanced,
    }


def compare_frames(train_report: dict[str, Any], valid_report: dict[str, Any]) -> dict[str, Any]:
    train_columns = set(train_report["columns"])
    valid_columns = set(valid_report["columns"])
    train_missing = {
        row["column"]: float(row["null_pct"])
        for row in train_report["missing_report"]
    }
    valid_missing = {
        row["column"]: float(row["null_pct"])
        for row in valid_report["missing_report"]
    }
    shared_columns = sorted(train_columns & valid_columns)
    missing_drift = []
    for column in shared_columns:
        drift = abs(valid_missing.get(column, 0.0) - train_missing.get(column, 0.0))
        if drift > 5.0:
            missing_drift.append(
                {
                    "column": column,
                    "train_null_pct": train_missing.get(column, 0.0),
                    "valid_null_pct": valid_missing.get(column, 0.0),
                    "abs_diff_pct": drift,
                }
            )
    return {
        "columns_only_in_train": sorted(train_columns - valid_columns),
        "columns_only_in_valid": sorted(valid_columns - train_columns),
        "missingness_drift_gt_5pct": missing_drift,
    }


def run_dqc(df: pd.DataFrame, name: str) -> dict[str, Any]:
    report = analyze_frame(df, name)
    report["columns"] = list(df.columns)
    report["missing_report"] = _build_missing_report(df).to_dict(orient="records")
    return report


def _render_report(report: dict[str, Any]) -> list[str]:
    lines = [
        f"{'=' * 20} Data Quality Report: {report['name']} {'=' * 20}",
        f"[1] Dimensions: {report['shape']['rows']} rows, {report['shape']['columns']} columns",
        f"[2] Duplicate Rows: {report['duplicate_rows']} ({report['duplicate_row_ratio']:.2%})",
        f"[3] Duplicate Columns (Exact Equality): {report['duplicate_column_count']}",
    ]
    for group in report["duplicate_column_groups"]:
        lines.append(f"    - Group: {', '.join(group)}")
    lines.append(f"[4] Columns with Missing or Inf Values: {len(report['missing_or_inf_columns'])}")
    for row in report["missing_or_inf_columns"]:
        lines.append(
            f"    - {row['column']}: null_count={row['null_count']}, inf_count={row['inf_count']}, null_pct={row['null_pct']:.2f}"
        )
    lines.append(f"[5] Constant Features: {len(report['constant_features'])}")
    for feature in report["constant_features"]:
        lines.append(f"    - {feature}")
    lines.append(f"[6] Near-Constant Features: {len(report['low_variance_features'])}")
    for feature in report["low_variance_features"]:
        lines.append(f"    - {feature}")
    lines.append(f"[7] High Cardinality Features (>50 unique): {len(report['high_cardinality_features'])}")
    for feature, cardinality in report["high_cardinality_features"].items():
        lines.append(f"    - {feature}: {cardinality}")
    lines.append(f"[8] Highly Imbalanced Categorical (>99% same value): {len(report['highly_imbalanced_categorical'])}")
    for feature, freq in report["highly_imbalanced_categorical"].items():
        lines.append(f"    - {feature}: {freq:.2%}")
    return lines


def write_reports(
    *,
    train_path: Path,
    valid_path: Path | None,
    output_dir: Path,
    model_features_only: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = _select_columns(pd.read_parquet(train_path), model_features_only=model_features_only)
    train_report = run_dqc(train_df, "TRAIN")
    payload: dict[str, Any] = {"train": train_report, "scope": "model_features" if model_features_only else "full_frame"}

    if valid_path is not None and valid_path.exists():
        valid_df = _select_columns(pd.read_parquet(valid_path), model_features_only=model_features_only)
        valid_report = run_dqc(valid_df, "VALID")
        payload["valid"] = valid_report
        payload["comparison"] = compare_frames(train_report, valid_report)

    summary_lines = []
    summary_lines.append(
        f"Feature Scope: {'model features only' if model_features_only else 'full training frame'}"
    )
    summary_lines.append("")
    summary_lines.extend(_render_report(train_report))
    if "valid" in payload:
        summary_lines.append("")
        summary_lines.extend(_render_report(payload["valid"]))
        summary_lines.append("")
        summary_lines.append(f"{'=' * 20} Dataset Comparison {'=' * 20}")
        comparison = payload["comparison"]
        if not comparison["columns_only_in_train"] and not comparison["columns_only_in_valid"]:
            summary_lines.append("[OK] Schemas match perfectly.")
        else:
            if comparison["columns_only_in_train"]:
                summary_lines.append(f"[WARN] Columns only in TRAIN: {comparison['columns_only_in_train']}")
            if comparison["columns_only_in_valid"]:
                summary_lines.append(f"[WARN] Columns only in VALID: {comparison['columns_only_in_valid']}")
        if comparison["missingness_drift_gt_5pct"]:
            summary_lines.append("[WARN] Significant missingness drift (>5% difference):")
            for row in comparison["missingness_drift_gt_5pct"]:
                summary_lines.append(
                    f"    - {row['column']}: Train={row['train_null_pct']:.2f}%, Valid={row['valid_null_pct']:.2f}% (Diff={row['abs_diff_pct']:.2f}%)"
                )

    (output_dir / "dqc_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/valid data quality reports from parquet files.")
    parser.add_argument("--train", type=str, required=True, help="Path to train parquet.")
    parser.add_argument("--valid", type=str, default=None, help="Path to valid parquet.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for data quality outputs.")
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Inspect the full parquet schema instead of model feature columns only.",
    )
    args = parser.parse_args()

    payload = write_reports(
        train_path=Path(args.train),
        valid_path=Path(args.valid) if args.valid else None,
        output_dir=Path(args.output_dir),
        model_features_only=not args.full_frame,
    )
    sys.stdout.write(f"Wrote DQC report to {Path(args.output_dir)} for datasets: {', '.join(payload.keys())}\n")


if __name__ == "__main__":
    main()
