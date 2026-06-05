from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_OBJECTIVE_METRICS = [
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
]

REQUIRED_REVERSAL_METRICS = [
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _path_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def _timestamp_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        columns = pd.read_parquet(path, columns=["timestamp"])
        ts = pd.to_datetime(columns["timestamp"], utc=True)
    except Exception as exc:  # noqa: BLE001 - audit should report failures, not crash.
        return {"exists": True, "readable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "exists": True,
        "readable": True,
        "row_count": int(len(ts)),
        "start": ts.min().isoformat() if len(ts) else None,
        "end": ts.max().isoformat() if len(ts) else None,
    }


def _metric_section_audit(report: dict[str, Any] | None, section: str, min_coverage: float) -> dict[str, Any]:
    if report is None:
        return {"exists": False, "passed": False, "missing": [*REQUIRED_OBJECTIVE_METRICS, *REQUIRED_REVERSAL_METRICS]}
    metrics = report.get(section) or {}
    missing = [name for name in [*REQUIRED_OBJECTIVE_METRICS, *REQUIRED_REVERSAL_METRICS] if name not in metrics]
    coverage = metrics.get("coverage")
    return {
        "exists": True,
        "missing": missing,
        "coverage": coverage,
        "coverage_constraint_satisfied": bool(coverage is not None and float(coverage) >= min_coverage),
        "selection_score": metrics.get("selection_score"),
        "utility": metrics.get("utility"),
        "accepted_sample_accuracy": metrics.get("accepted_sample_accuracy"),
        "accepted_count": metrics.get("accepted_count"),
        "passed": not missing and bool(coverage is not None and float(coverage) >= min_coverage),
    }


def _replay_audit(path: Path, min_coverage: float) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {"exists": False, "passed": False}
    metrics = payload.get("metrics") or {}
    coverage = metrics.get("coverage")
    return {
        "exists": True,
        "window_start": payload.get("source_replay_window_start_utc"),
        "window_end": payload.get("source_replay_window_end_utc"),
        "coverage": coverage,
        "accepted_sample_accuracy": metrics.get("accepted_sample_accuracy"),
        "selection_score": metrics.get("selection_score"),
        "coverage_constraint_satisfied": bool(coverage is not None and float(coverage) >= min_coverage),
        "reversal_trend_metrics_available": bool(payload.get("reversal_trend_metrics_available")),
        "passed": bool(coverage is not None and float(coverage) >= min_coverage),
    }


def _feature_config_audit(path: Path, min_coverage: float) -> dict[str, Any]:
    payload = _load_yaml(path)
    if payload is None:
        return {"path": str(path), "exists": False, "passed": False}
    second_level = payload.get("second_level") or {}
    packs = (((second_level.get("profiles") or {}).get(second_level.get("feature_profile", "")) or {}).get("packs") or [])
    objective = payload.get("objective") or {}
    threshold_search = payload.get("threshold_search") or {}
    checks = {
        "min_coverage": objective.get("min_coverage"),
        "min_coverage_satisfied": float(objective.get("min_coverage", 0.0)) >= min_coverage,
        "hard_constraint": threshold_search.get("hard_constraint"),
        "hard_constraint_satisfied": threshold_search.get("hard_constraint") == "coverage_only",
        "second_level_enabled": second_level.get("enabled") is True,
        "require_agg_trade_through_last_second": second_level.get("require_agg_trade_through_last_second") is True,
        "max_agg_trade_lag_seconds": second_level.get("max_agg_trade_lag_seconds"),
        "first_minute_impulse_pack_enabled": "second_level_first_minute_impulse" in packs,
        "agg_trade_pack_enabled": "second_level_trade_microstructure" in packs,
        "book_microstructure_pack_enabled": "second_level_book_microstructure" in packs,
    }
    passed = all(
        bool(checks[name])
        for name in (
            "min_coverage_satisfied",
            "hard_constraint_satisfied",
            "second_level_enabled",
            "require_agg_trade_through_last_second",
            "first_minute_impulse_pack_enabled",
            "agg_trade_pack_enabled",
            "book_microstructure_pack_enabled",
        )
    )
    return {"path": str(path), "exists": True, "checks": checks, "passed": passed}


def _execution_config_audit(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    if payload is None:
        return {"path": str(path), "exists": False, "passed": False}
    orders = payload.get("orders") or {}
    first = orders.get("first") or {}
    second = orders.get("second") or {}
    checks = {
        "first_enabled": first.get("enabled") is True,
        "first_price_mode": first.get("price_mode"),
        "first_price_mode_satisfied": first.get("price_mode") == "min_best_bid_offset_and_cap",
        "first_best_bid_offset": first.get("best_bid_offset"),
        "first_best_bid_offset_satisfied": float(first.get("best_bid_offset", 0.0)) == -0.05,
        "first_price_cap": first.get("price_cap"),
        "first_price_cap_satisfied": float(first.get("price_cap", 0.0)) == 0.65,
        "second_enabled": second.get("enabled"),
        "second_disabled_satisfied": second.get("enabled") is False,
    }
    passed = all(
        bool(checks[name])
        for name in (
            "first_enabled",
            "first_price_mode_satisfied",
            "first_best_bid_offset_satisfied",
            "first_price_cap_satisfied",
            "second_disabled_satisfied",
        )
    )
    return {"path": str(path), "exists": True, "checks": checks, "passed": passed}


def _prd_check(check_id: str, requirement: str, status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check_id,
        "requirement": requirement,
        "status": status,
        "passed": status == "passed",
        "evidence": evidence,
    }


def build_audit(repo_root: Path, *, min_coverage: float = 0.90) -> dict[str, Any]:
    baseline_run = repo_root / "artifacts/data_v2/experiments/20260521_baseline_coverage_090"
    feature_run = repo_root / "artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features"
    second_level_store = repo_root / "artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT"
    baseline_config_path = repo_root / "experiments/configs/20260521_polymarket_resolved_baseline_coverage_090.yaml"
    feature_config_path = repo_root / "experiments/configs/20260521_regime_reversal_second_agg_features.yaml"
    baseline_report = _load_json(baseline_run / "report.json")
    feature_report = _load_json(feature_run / "report.json")
    replay_0515 = repo_root / "artifacts/reports/execution_engine/replay_20260521_baseline_coverage_090_20260515_window.json"
    replay_0520 = repo_root / "artifacts/reports/execution_engine/replay_20260521_baseline_coverage_090_20260520_window.json"
    normalized_1m = repo_root / "artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet"
    dataset_frame = repo_root / "artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/polymarket_resolved_extended_training_frame.parquet"

    feature_replay_0515 = (
        repo_root / "artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260515_window.json"
    )
    feature_replay_0520 = (
        repo_root / "artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260520_window.json"
    )

    baseline_validation = _metric_section_audit(baseline_report, "validation_metrics", min_coverage)
    baseline_train = _metric_section_audit(baseline_report, "train_metrics", min_coverage)
    feature_validation = _metric_section_audit(feature_report, "validation_metrics", min_coverage)
    baseline_replay_0515 = _replay_audit(replay_0515, min_coverage)
    baseline_replay_0520 = _replay_audit(replay_0520, min_coverage)
    feature_replay_0515_audit = _replay_audit(feature_replay_0515, min_coverage)
    feature_replay_0520_audit = _replay_audit(feature_replay_0520, min_coverage)
    feature_config = _feature_config_audit(feature_config_path, min_coverage)
    execution_config = _execution_config_audit(repo_root / "execution_engine/config.yaml")
    normalized_range = _timestamp_range(normalized_1m)
    training_frame_range = _timestamp_range(dataset_frame)
    diagnostics_path = repo_root / "artifacts/data_v2/reports/reversal_diagnostics/baseline_reversal_trend_slices_090.json"

    checks = {
        "baseline_config": _path_status(baseline_config_path),
        "feature_config": feature_config,
        "baseline_report": _metric_section_audit(baseline_report, "validation_metrics", min_coverage),
        "baseline_train_metrics": baseline_train,
        "feature_report": feature_validation,
        "second_level_store_exists": second_level_store.exists(),
        "second_level_store_path": str(second_level_store),
        "normalized_1m_range": normalized_range,
        "polymarket_training_frame_range": training_frame_range,
        "baseline_replay_20260515": baseline_replay_0515,
        "baseline_replay_20260520": baseline_replay_0520,
        "new_feature_replay_20260515": feature_replay_0515_audit,
        "new_feature_replay_20260520": feature_replay_0520_audit,
        "baseline_reversal_diagnostics": _path_status(diagnostics_path),
        "execution_config": execution_config,
    }

    deliverables = [
        _prd_check(
            "baseline_config_saved",
            "Save exact baseline config with objective.min_coverage >= 0.90 and threshold_search.hard_constraint=coverage_only.",
            "passed" if checks["baseline_config"]["exists"] else "missing",
            checks["baseline_config"],
        ),
        _prd_check(
            "baseline_report_coverage_090",
            "Recompute current baseline at coverage >= 0.90 with required objective and reversal/trend metrics.",
            "passed" if baseline_validation["passed"] and baseline_train["passed"] else "incomplete",
            {"validation": baseline_validation, "train": baseline_train, "report_path": str(baseline_run / "report.json")},
        ),
        _prd_check(
            "baseline_reversal_diagnostics",
            "Write baseline reversal/trend diagnostic report.",
            "passed" if checks["baseline_reversal_diagnostics"]["exists"] else "missing",
            checks["baseline_reversal_diagnostics"],
        ),
        _prd_check(
            "feature_config_saved",
            "Save new second-level/aggTrade/first-minute feature experiment config at coverage >= 0.90.",
            "passed" if feature_config["passed"] else "incomplete",
            feature_config,
        ),
        _prd_check(
            "second_level_store_materialized",
            "Enable and validate existing second-level and aggTrade inputs with offline feature availability.",
            "passed" if second_level_store.exists() else "blocked",
            {"path": str(second_level_store), "exists": second_level_store.exists()},
        ),
        _prd_check(
            "feature_artifact_report",
            "Train and validate the new feature artifact with report.json and required slice metrics.",
            "passed" if feature_validation["passed"] else "missing",
            {"report_path": str(feature_run / "report.json"), **feature_validation},
        ),
        _prd_check(
            "mandatory_replay_windows_baseline",
            "Report mandatory baseline replay windows for 2026-05-15/16 and 2026-05-20/21.",
            "passed" if baseline_replay_0515["exists"] and baseline_replay_0520["exists"] else "missing",
            {"20260515": baseline_replay_0515, "20260520": baseline_replay_0520},
        ),
        _prd_check(
            "mandatory_replay_windows_feature",
            "Report mandatory new feature replay windows for 2026-05-15/16 and 2026-05-20/21.",
            "passed" if feature_replay_0515_audit["exists"] and feature_replay_0520_audit["exists"] else "missing",
            {"20260515": feature_replay_0515_audit, "20260520": feature_replay_0520_audit},
        ),
        _prd_check(
            "replay_coverage_gate",
            "Baseline and new feature replay reports should be evaluated against coverage >= 0.90 and include reversal/trend availability.",
            "passed"
            if baseline_replay_0515.get("passed")
            and baseline_replay_0520.get("passed")
            and feature_replay_0515_audit.get("passed")
            and feature_replay_0520_audit.get("passed")
            and baseline_replay_0515.get("reversal_trend_metrics_available")
            and baseline_replay_0520.get("reversal_trend_metrics_available")
            and feature_replay_0515_audit.get("reversal_trend_metrics_available")
            and feature_replay_0520_audit.get("reversal_trend_metrics_available")
            else "incomplete",
            {
                "baseline_20260515": baseline_replay_0515,
                "baseline_20260520": baseline_replay_0520,
                "feature_20260515": feature_replay_0515_audit,
                "feature_20260520": feature_replay_0520_audit,
            },
        ),
        _prd_check(
            "local_data_replay_coverage",
            "Local normalized data must cover mandatory replay windows through 2026-05-21T00:23:40Z.",
            "passed" if normalized_range.get("end", "") >= "2026-05-21T00:23:40+00:00" else "blocked",
            normalized_range,
        ),
        _prd_check(
            "execution_first_leg_policy",
            "Configure first_leg_price = min(best_bid - 0.05, 0.65) and keep second leg disabled.",
            "passed" if execution_config["passed"] else "incomplete",
            execution_config,
        ),
    ]

    missing_or_blocked = []
    if not checks["second_level_store_exists"]:
        missing_or_blocked.append("materialized second-level feature store is missing")
    if feature_report is None:
        missing_or_blocked.append("new feature experiment report is missing")
    if not checks["new_feature_replay_20260515"]["exists"] or not checks["new_feature_replay_20260520"]["exists"]:
        missing_or_blocked.append("new feature replay-window reports are missing")
    for name in ("baseline_replay_20260515", "baseline_replay_20260520"):
        if checks[name].get("exists") and not checks[name].get("coverage_constraint_satisfied"):
            missing_or_blocked.append(f"{name} coverage is below {min_coverage}")
    for name in ("new_feature_replay_20260515", "new_feature_replay_20260520"):
        if checks[name].get("exists") and not checks[name].get("coverage_constraint_satisfied"):
            missing_or_blocked.append(f"{name} coverage is below {min_coverage}")
    if normalized_range.get("end") and normalized_range["end"] < "2026-05-21T00:23:40+00:00":
        missing_or_blocked.append("local normalized BTCUSDT 1m data does not cover mandatory replay windows")
    if not execution_config["passed"]:
        missing_or_blocked.append("execution first-leg/second-leg policy config is incomplete")
    for item in deliverables:
        if item["status"] in {"missing", "blocked", "incomplete"} and item["id"] not in {
            "second_level_store_materialized",
            "feature_artifact_report",
            "mandatory_replay_windows_feature",
            "replay_coverage_gate",
            "local_data_replay_coverage",
        }:
            missing_or_blocked.append(f"{item['id']} is {item['status']}")

    return {
        "objective": "regime/reversal feature optimization PRD 2026-05-21",
        "completion_audit_rule": "complete only if every explicit PRD deliverable has concrete artifact evidence and no gate remains missing, blocked, or incomplete",
        "min_coverage": min_coverage,
        "deliverables": deliverables,
        "checks": checks,
        "missing_or_blocked": sorted(set(missing_or_blocked)),
        "complete": not missing_or_blocked and all(item["passed"] for item in deliverables),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completion of the regime/reversal PRD artifacts.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="artifacts/data_v2/reports/regime_reversal_prd_audit_20260521.json")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    payload = build_audit(repo_root)
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
