from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.regime_reversal_prd_audit import build_audit


def _write_report(path: Path, coverage: float = 0.91) -> None:
    metric_names = [
        "sample_count",
        "coverage",
        "accepted_sample_accuracy",
        "precision_up",
        "precision_down",
        "balanced_precision",
        "all_sample_accuracy",
        "selected_t_up",
        "selected_t_down",
        "accepted_count",
        "up_prediction_count",
        "down_prediction_count",
        "share_up_predictions",
        "share_down_predictions",
        "roc_auc",
        "brier_score",
        "log_loss",
        "utility",
        "downside_risk",
        "selection_score",
        "continuation_sample_count",
        "continuation_coverage",
        "continuation_accepted_accuracy",
        "continuation_accepted_count",
        "reversal_sample_count",
        "reversal_coverage",
        "reversal_accepted_accuracy",
        "reversal_accepted_count",
        "reversal_loss_contribution",
        "trend_following_loss_contribution",
    ]
    metrics = {name: 1.0 for name in metric_names}
    metrics["coverage"] = coverage
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train_metrics": metrics, "validation_metrics": metrics}), encoding="utf-8")


def test_prd_audit_reports_missing_second_level_store(tmp_path: Path) -> None:
    (tmp_path / "experiments/configs").mkdir(parents=True)
    (tmp_path / "experiments/configs/20260521_polymarket_resolved_baseline_coverage_090.yaml").write_text("x", encoding="utf-8")
    (tmp_path / "experiments/configs/20260521_regime_reversal_second_agg_features.yaml").write_text("x", encoding="utf-8")
    _write_report(tmp_path / "artifacts/data_v2/experiments/20260521_baseline_coverage_090/report.json")
    (tmp_path / "artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines").mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.date_range("2026-05-15", periods=2, tz="UTC")}).to_parquet(
        tmp_path / "artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet",
        index=False,
    )
    (tmp_path / "artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m").mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.date_range("2026-05-15", periods=2, tz="UTC")}).to_parquet(
        tmp_path / "artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/polymarket_resolved_extended_training_frame.parquet",
        index=False,
    )

    audit = build_audit(tmp_path)

    assert audit["complete"] is False
    assert "materialized second-level feature store is missing" in audit["missing_or_blocked"]
