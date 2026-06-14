from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "price_estimator" / "experiment" / "positive_gap_metrics_summary.csv"
COVERAGE_MARGIN = 0.01


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_id(prediction_path: Path) -> str:
    rel = prediction_path.relative_to(ROOT)
    if rel.parts[:3] == ("price_estimator", "upper_bound_mlp", "reports"):
        metrics = _read_json(ROOT / "price_estimator" / "upper_bound_mlp" / "reports" / "upper_bound_mlp_metrics.json")
        return str(metrics.get("experiment_id") or "upper_bound_mlp")
    if rel.parts[:2] == ("price_estimator", "data"):
        return "quantile_baseline"
    if rel.parts[:2] == ("price_estimator", "experiment") and len(rel.parts) >= 4:
        return rel.parts[2]
    return prediction_path.parent.parent.name


def _source_type(prediction_path: Path) -> str:
    if rel_is_baseline_quantile(prediction_path):
        return "quantile"
    text = str(prediction_path).replace("\\", "/")
    if "/upper_bound_mlp/" in text or "upper_bound_mlp" in _experiment_id(prediction_path):
        return "mlp"
    if "quantile" in text:
        return "quantile"
    return "unknown"


def _prediction_columns(df: pd.DataFrame, prediction_path: Path) -> list[tuple[str, str]]:
    exp_id = _experiment_id(prediction_path)
    columns: list[tuple[str, str]] = []
    if "p_pred" in df.columns:
        columns.append(("p_pred", "selected_prediction"))
    elif "p_upper_bound" in df.columns:
        columns.append(("p_upper_bound", "selected_prediction"))

    if "quantile" in exp_id or rel_is_baseline_quantile(prediction_path):
        for col in sorted(c for c in df.columns if c.startswith("pred_q")):
            columns.append((col, col.replace("pred_", "")))
    return columns


def rel_is_baseline_quantile(prediction_path: Path) -> bool:
    rel = prediction_path.relative_to(ROOT)
    return rel.parts[:2] == ("price_estimator", "data")


def _positive_gap_stats(gap: pd.Series) -> dict[str, float]:
    positive = gap[gap > 0].dropna()
    out: dict[str, float] = {
        "positive_gap_count": float(len(positive)),
        "positive_gap_share": float(len(positive) / len(gap)) if len(gap) else math.nan,
    }
    if positive.empty:
        for name in ("q10", "q25", "median", "mean", "q75", "q90", "max", "q90_q10"):
            out[f"positive_gap_{name}"] = math.nan
        return out

    q10 = float(positive.quantile(0.10))
    q90 = float(positive.quantile(0.90))
    out.update(
        {
            "positive_gap_q10": q10,
            "positive_gap_q25": float(positive.quantile(0.25)),
            "positive_gap_median": float(positive.median()),
            "positive_gap_mean": float(positive.mean()),
            "positive_gap_q75": float(positive.quantile(0.75)),
            "positive_gap_q90": q90,
            "positive_gap_max": float(positive.max()),
            "positive_gap_q90_q10": q90 - q10,
        }
    )
    return out


def _row_for_prediction(prediction_path: Path, df: pd.DataFrame, pred_col: str, variant: str) -> dict:
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    target = pd.to_numeric(df["target_raw"], errors="coerce")
    valid = pred.notna() & target.notna()
    gap = pred[valid] - target[valid]
    violation = (COVERAGE_MARGIN - gap).clip(lower=0.0)
    row = {
        "experiment_id": _experiment_id(prediction_path),
        "variant": variant,
        "prediction_column": pred_col,
        "source_type": _source_type(prediction_path),
        "prediction_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
        "sample_count": int(len(gap)),
        "coverage_margin": COVERAGE_MARGIN,
        "coverage": float((gap >= COVERAGE_MARGIN).mean()) if len(gap) else math.nan,
        "mean_gap": float(gap.mean()) if len(gap) else math.nan,
        "median_gap": float(gap.median()) if len(gap) else math.nan,
        "gap_p05": float(gap.quantile(0.05)) if len(gap) else math.nan,
        "gap_p95": float(gap.quantile(0.95)) if len(gap) else math.nan,
        "gap_p95_p05_range": float(gap.quantile(0.95) - gap.quantile(0.05)) if len(gap) else math.nan,
        "p99_violation": float(violation.quantile(0.99)) if len(gap) else math.nan,
        "max_violation": float(violation.max()) if len(gap) else math.nan,
    }
    row.update(_positive_gap_stats(gap))
    return row


def collect_rows() -> pd.DataFrame:
    prediction_paths = sorted(
        (ROOT / "price_estimator").glob("**/predictions_validation.parquet"),
        key=lambda p: str(p).lower(),
    )
    rows: list[dict] = []
    for prediction_path in prediction_paths:
        df = pd.read_parquet(prediction_path)
        if "target_raw" not in df.columns:
            continue
        seen: set[str] = set()
        for pred_col, variant in _prediction_columns(df, prediction_path):
            if pred_col in seen:
                continue
            seen.add(pred_col)
            rows.append(_row_for_prediction(prediction_path, df, pred_col, variant))
    return pd.DataFrame(rows)


def main() -> None:
    summary = collect_rows()
    if summary.empty:
        raise SystemExit("No prediction rows found.")
    summary = summary.sort_values(
        ["coverage", "positive_gap_q90_q10", "mean_gap", "experiment_id", "variant"],
        ascending=[False, True, True, True, True],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
