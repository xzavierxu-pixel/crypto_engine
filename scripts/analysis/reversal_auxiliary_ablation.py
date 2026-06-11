from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (  # noqa: E402
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_auxiliary_ablation.yaml")
LEAKAGE_BLOCKLIST = {
    "target",
    "future_close",
    "abs_return",
    "signed_return",
    "stage1_target",
    "stage2_target",
    "original_btc_direction_target",
}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline_inputs(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = _read_json(experiment_dir / "artifact_manifest.json")
    feature_columns = list(manifest["feature_columns"])
    train_frame = pd.read_parquet(experiment_dir / "development_frame.parquet")
    validation_frame = pd.read_parquet(experiment_dir / "validation_frame.parquet")
    train_predictions = pd.read_parquet(experiment_dir / "train_predictions.parquet")
    validation_predictions = pd.read_parquet(experiment_dir / "validation_predictions.parquet")
    return train_frame, validation_frame, train_predictions, validation_predictions, feature_columns


def _first_minute_up(frame: pd.DataFrame) -> pd.Series:
    if "fm_ret" in frame.columns:
        ret = pd.to_numeric(frame["fm_ret"], errors="coerce")
    elif "ret_1" in frame.columns:
        ret = pd.to_numeric(frame["ret_1"], errors="coerce")
    else:
        raise ValueError("reversal target requires fm_ret or ret_1.")
    return ret >= 0.0


def _reversal_target(frame: pd.DataFrame) -> pd.Series:
    return (_first_minute_up(frame) != (frame["target"].astype(int) == 1)).astype(int)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _select_features(all_features: list[str], variant: dict[str, Any]) -> list[str]:
    safe = [name for name in all_features if name not in LEAKAGE_BLOCKLIST and "target" not in name.lower()]
    mode = variant.get("mode", "all")
    patterns = _compile_patterns(list(variant.get("patterns", [])))
    if mode == "all":
        return safe
    if mode == "drop_regex":
        return [name for name in safe if not any(pattern.search(name) for pattern in patterns)]
    if mode == "keep_regex":
        selected = [name for name in safe if any(pattern.search(name) for pattern in patterns)]
        if not selected:
            raise ValueError(f"feature variant {variant.get('name')} selected no features.")
        return selected
    raise ValueError(f"unsupported feature variant mode: {mode}")


def _fit_model(train_frame: pd.DataFrame, y_train: pd.Series, features: list[str], config: dict[str, Any]) -> lgb.LGBMClassifier:
    params = dict(config.get("model", {}))
    params.pop("type", None)
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    if positives:
        params.setdefault("scale_pos_weight", negatives / positives)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_frame[features], y_train)
    return model


def _prepare_decision_frame(validation_predictions: pd.DataFrame, reversal_risk: pd.Series) -> pd.DataFrame:
    frame = validation_predictions.copy()
    frame["p_base"] = pd.to_numeric(frame["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    frame["base_decision"] = frame["decision"]
    frame["reversal_risk"] = reversal_risk.reindex(frame.index).astype("float64").clip(0.0, 1.0)
    return frame


def _apply_aux_gate(frame: pd.DataFrame, *, reversal_risk_threshold: float, base_band: float) -> pd.Series:
    decision = frame["base_decision"].astype("object")
    p_base = frame["p_base"].astype("float64")
    risk = frame["reversal_risk"].astype("float64")
    fm_side = frame["first_minute_side"].astype("object")
    base_side = decision.replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
    base_accepted = decision != "ABSTAIN"
    base_continuation = base_accepted & (base_side == fm_side)
    base_reversal = base_accepted & (base_side != fm_side)
    weak_base = (p_base - 0.5).abs() <= float(base_band)
    keep = base_accepted & ~(base_continuation & (risk >= reversal_risk_threshold) & weak_base)
    keep &= ~base_reversal | (risk >= reversal_risk_threshold)
    output = pd.Series("ABSTAIN", index=frame.index, dtype="object")
    output.loc[keep] = decision.loc[keep]
    return output


def _search_aux_gate(frame: pd.DataFrame, config: dict[str, Any], baseline_score: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    search = config["threshold_search"]
    min_coverage = float(config["objective"]["min_coverage"])
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in search["reversal_risk_thresholds"]:
        for band in search["base_bands"]:
            decisions = _apply_aux_gate(frame, reversal_risk_threshold=float(threshold), base_band=float(band))
            metrics = compute_decision_metrics(
                frame["target"],
                frame["p_base"],
                decisions,
                selected_t_up=float(threshold),
                selected_t_down=float(band),
            )
            row = {"reversal_risk_threshold": float(threshold), "base_band": float(band), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["utility"] > 0 and metrics["accepted_sample_accuracy"] > 0.50:
                eligible.append(row)
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["selection_score"],
            row["selection_score"] > baseline_score,
            row["utility"],
            row["coverage"],
            row["accepted_count"],
        ),
    )
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "selection_score",
        "hard_constraint": "coverage_only",
        "fallback_reason": None if eligible else "no candidate satisfied coverage/accuracy/utility constraints",
    }


def _alert_metrics(y_reversal: pd.Series, reversal_risk: pd.Series, threshold: float) -> dict[str, float]:
    y = y_reversal.astype(int)
    risk = reversal_risk.astype("float64").clip(0.0, 1.0)
    alerts = risk >= threshold
    alert_count = int(alerts.sum())
    false_alerts = int(((y == 0) & alerts).sum())
    continuation_count = int((y == 0).sum())
    return {
        "reversal_alert_count": float(alert_count),
        "reversal_alert_precision": float(precision_score(y, alerts.astype(int), zero_division=0)),
        "reversal_alert_recall": float(recall_score(y, alerts.astype(int), zero_division=0)),
        "continuation_false_alert_rate": float(false_alerts / continuation_count) if continuation_count else 0.0,
        "alert_reversal_share": float(y.loc[alerts].mean()) if alert_count else 0.0,
        "reversal_risk_roc_auc": float(roc_auc_score(y, risk)) if y.nunique() == 2 else 0.0,
        "reversal_risk_average_precision": float(average_precision_score(y, risk)) if y.nunique() == 2 else 0.0,
    }


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True) if "timestamp" in frame.columns else pd.to_datetime(frame["market_t0"], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    baseline_experiment = Path(config["baseline_experiment"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_baseline_inputs(baseline_experiment)
    y_train_reversal = _reversal_target(train_frame)
    y_valid_reversal = _reversal_target(validation_frame)
    baseline_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_predictions["p_up"],
        validation_predictions["decision"],
        selected_t_up=float(validation_predictions["selected_t_up"].iloc[0]),
        selected_t_down=float(validation_predictions["selected_t_down"].iloc[0]),
    )
    baseline_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_predictions["decision"]))

    variants = []
    best_variant: dict[str, Any] | None = None
    best_frontier: pd.DataFrame | None = None
    for variant_config in config["feature_variants"]:
        variant_name = str(variant_config["name"])
        features = _select_features(all_features, variant_config)
        model = _fit_model(train_frame, y_train_reversal, features, config)
        train_risk = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index)
        valid_risk = pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index)
        train_decision_frame = _prepare_decision_frame(train_predictions, train_risk)
        decision_frame = _prepare_decision_frame(validation_predictions, valid_risk)
        frontier, best = _search_aux_gate(decision_frame, config, baseline_metrics["selection_score"])
        train_decisions = _apply_aux_gate(
            train_decision_frame,
            reversal_risk_threshold=float(best["reversal_risk_threshold"]),
            base_band=float(best["base_band"]),
        )
        best_decisions = _apply_aux_gate(
            decision_frame,
            reversal_risk_threshold=float(best["reversal_risk_threshold"]),
            base_band=float(best["base_band"]),
        )
        train_metrics = compute_decision_metrics(
            train_decision_frame["target"],
            train_decision_frame["p_base"],
            train_decisions,
            selected_t_up=float(best["reversal_risk_threshold"]),
            selected_t_down=float(best["base_band"]),
        )
        train_metrics.update(compute_reversal_continuation_metrics(train_decision_frame, train_decisions))
        best_metrics = compute_decision_metrics(
            decision_frame["target"],
            decision_frame["p_base"],
            best_decisions,
            selected_t_up=float(best["reversal_risk_threshold"]),
            selected_t_down=float(best["base_band"]),
        )
        best_metrics.update(compute_reversal_continuation_metrics(decision_frame, best_decisions))
        alerts = _alert_metrics(y_valid_reversal, valid_risk, float(best["reversal_risk_threshold"]))
        payload = {
            "variant": variant_name,
            "feature_count": len(features),
            "dropped_feature_count": len(all_features) - len(features),
            "best_gate": best,
            "train_metrics": train_metrics,
            "validation_metrics": best_metrics,
            "alert_metrics": alerts,
            "accepted": bool(
                best_metrics["selection_score"] > baseline_metrics["selection_score"]
                and best_metrics["coverage"] >= float(config["objective"]["min_coverage"])
                and best_metrics["utility"] > 0.0
                and best_metrics["accepted_sample_accuracy"] > 0.50
            ),
        }
        variants.append(payload)
        frontier.to_csv(output_dir / f"{variant_name}_frontier.csv", index=False)
        if best_variant is None or best_metrics["selection_score"] > best_variant["validation_metrics"]["selection_score"]:
            best_variant = payload
            best_frontier = frontier

    if best_variant is None or best_frontier is None:
        raise RuntimeError("no feature variants were evaluated.")
    summary = pd.DataFrame(
        [
            {
                "variant": item["variant"],
                "accepted": item["accepted"],
                "feature_count": item["feature_count"],
                "selection_score": item["validation_metrics"]["selection_score"],
                "utility": item["validation_metrics"]["utility"],
                "coverage": item["validation_metrics"]["coverage"],
                "accepted_sample_accuracy": item["validation_metrics"]["accepted_sample_accuracy"],
                "accepted_count": item["validation_metrics"]["accepted_count"],
                "reversal_alert_precision": item["alert_metrics"]["reversal_alert_precision"],
                "continuation_false_alert_rate": item["alert_metrics"]["continuation_false_alert_rate"],
            }
            for item in variants
        ]
    ).sort_values("selection_score", ascending=False)
    summary_path = output_dir / "variant_summary.csv"
    report_path = output_dir / "report.json"
    summary.to_csv(summary_path, index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "baseline_experiment": str(baseline_experiment),
        "primary_metric": "validation selection_score with coverage >= min_coverage",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "baseline_validation_metrics": baseline_metrics,
        "best_variant": best_variant["variant"],
        "train_metrics": best_variant["train_metrics"],
        "validation_metrics": best_variant["validation_metrics"],
        "variant_results": variants,
        "variant_summary_path": str(summary_path),
        "accepted": bool(best_variant["accepted"]),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reversal-risk auxiliary model and feature ablation experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    baseline = report["baseline_validation_metrics"]
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": str(Path(report["variant_summary_path"]).with_name("report.json")),
                "accepted": report["accepted"],
                "best_variant": report["best_variant"],
                "baseline_selection_score": baseline["selection_score"],
                "best_selection_score": metrics["selection_score"],
                "baseline_utility": baseline["utility"],
                "best_utility": metrics["utility"],
                "coverage": metrics["coverage"],
                "accepted_count": metrics["accepted_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
