from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.isotonic import IsotonicCalibration  # noqa: E402
from src.calibration.platt_logit import PlattLogitCalibration  # noqa: E402
from src.data.polymarket_l2 import L2_FEATURE_PREFIX, assert_feature_schema_safe  # noqa: E402
from src.model.reversal_hybrid import compute_decision_metrics  # noqa: E402


FORBIDDEN = {
    "target", "future_close", "abs_return", "signed_return", "stage1_target", "stage2_target",
    "future_low_4m", "up_future_low_4m", "down_future_low_4m", "winning_outcome", "correct",
}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _probability_metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def _reliability(y: pd.Series, p: pd.Series, bins: int = 10) -> tuple[list[dict[str, float]], float]:
    bucket = pd.cut(p, np.linspace(0.0, 1.0, bins + 1), include_lowest=True, labels=False)
    rows = []
    ece = 0.0
    for value, index in bucket.groupby(bucket).groups.items():
        if pd.isna(value) or not len(index):
            continue
        confidence = float(p.loc[index].mean())
        accuracy = float(y.loc[index].mean())
        weight = len(index) / len(y)
        ece += weight * abs(confidence - accuracy)
        rows.append({"bucket": int(value), "count": len(index), "mean_probability": confidence, "positive_rate": accuracy})
    return rows, float(ece)


def _select_calibrator(raw: pd.Series, y: pd.Series) -> tuple[str, Any, dict[str, Any]]:
    candidates = {"platt_logit": PlattLogitCalibration(), "isotonic": IsotonicCalibration()}
    report: dict[str, Any] = {}
    for name, calibrator in candidates.items():
        calibrator.fit(raw, y)
        calibrated = calibrator.transform(raw)
        report[name] = _probability_metrics(y, calibrated)
    selected = min(candidates, key=lambda name: (report[name]["log_loss"], report[name]["brier_score"], name))
    return selected, candidates[selected], report


def _decisions(p: pd.Series, t_up: float, t_down: float) -> pd.Series:
    result = pd.Series("ABSTAIN", index=p.index, dtype="object")
    result.loc[p >= t_up] = "UP"
    result.loc[p <= t_down] = "DOWN"
    return result


def _threshold_search(y: pd.Series, p: pd.Series, search: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    records = []
    min_coverage = float(search["min_coverage"])
    values_up = np.arange(search["t_up_min"], search["t_up_max"] + search["step"] / 2, search["step"])
    values_down = np.arange(search["t_down_min"], search["t_down_max"] + search["step"] / 2, search["step"])
    for t_up in values_up:
        for t_down in values_down:
            if t_down >= t_up:
                continue
            metrics = compute_decision_metrics(
                y,
                p,
                _decisions(p, float(t_up), float(t_down)),
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics["coverage_constraint_satisfied"] = bool(metrics["coverage"] >= min_coverage)
            records.append(metrics)
    frontier = pd.DataFrame(records)
    eligible = frontier.loc[frontier["coverage"] >= min_coverage]
    if eligible.empty:
        raise RuntimeError("no threshold candidate satisfies coverage constraint")
    best_index = eligible.sort_values(
        ["selection_score", "utility", "coverage", "accepted_count"], ascending=False
    ).index[0]
    return frontier.loc[best_index].to_dict(), frontier


def _load_l2(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no built L2 features at {path}")
    frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    assert_feature_schema_safe([column for column in frame if column.startswith(L2_FEATURE_PREFIX)])
    if (frame["max_feature_event_time"] > frame["feature_cutoff_time"]).any():
        raise ValueError("post-cutoff L2 event detected")
    if frame["market_t0"].duplicated().any():
        raise ValueError("duplicate L2 market_t0")
    return frame


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / config["baseline_frame_dir"]
    baseline = pd.concat(
        [pd.read_parquet(baseline_dir / "development_frame.parquet"), pd.read_parquet(baseline_dir / "validation_frame.parquet")],
        ignore_index=True,
    ).drop_duplicates("market_t0")
    frozen = pd.read_parquet(ROOT / config["frozen_splits"])
    labels = pd.read_parquet(ROOT / config["resolved_labels"])
    labels = labels.loc[labels["polymarket_label_status"].eq("resolved")].copy()
    label_columns = [
        column for column in ("market_t0", "target", "condition_id", "endDate", "polymarket_slug")
        if column in labels
    ]
    labels = labels[label_columns].drop_duplicates("market_t0").rename(
        columns={column: f"{column}_resolved" for column in label_columns if column != "market_t0"}
    )
    l2 = _load_l2(ROOT / config["l2_feature_dir"])
    baseline_columns = json.loads((ROOT / config["classifier_manifest"]).read_text(encoding="utf-8"))["feature_columns"]
    l2_columns = [column for column in l2 if column.startswith(L2_FEATURE_PREFIX)]
    frame = frozen[["market_slug", "market_t0", "split"]].merge(
        baseline, on="market_t0", how="left", validate="one_to_one", suffixes=("", "_baseline")
    ).merge(labels, on="market_t0", how="left", validate="one_to_one").merge(
        l2[["market_t0", *l2_columns]], on="market_t0", how="left", validate="one_to_one"
    )
    frame["target"] = pd.to_numeric(frame.get("target"), errors="coerce").fillna(
        pd.to_numeric(frame["target_resolved"], errors="coerce")
    )
    frame["baseline_feature_available"] = frame[baseline_columns[0]].notna().astype(float)
    frame["timestamp"] = pd.to_datetime(frame["market_t0"], utc=True)
    frame["feature_timestamp"] = frame.get("feature_timestamp").fillna(frame["timestamp"] + pd.Timedelta(minutes=1))
    frame["decision_time"] = frame.get("decision_time").fillna(frame["feature_timestamp"])
    for column in ("condition_id", "endDate"):
        resolved_column = f"{column}_resolved"
        if resolved_column in frame:
            if column in frame:
                frame[column] = frame[column].fillna(frame[resolved_column])
            else:
                frame[column] = frame[resolved_column]
    coverage = {
        "eligible_market_count": len(frame),
        "baseline_row_coverage": float(frame["baseline_feature_available"].mean()),
        "l2_row_coverage": float(frame[L2_FEATURE_PREFIX + "available"].notna().mean()),
        "missing_baseline_market_count": int((frame["baseline_feature_available"] == 0).sum()),
        "resolved_label_coverage": float(frame["target"].notna().mean()),
    }
    # Preserve the frozen eligible universe. CatBoost handles missing feature values;
    # downstream preprocessors fit numeric imputation on train only.
    modeled = frame.dropna(subset=["target"]).copy()
    reports = {}
    for variant, variant_mode in config["variants"].items():
        include_l2 = bool(variant_mode)
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        features = list(baseline_columns) + (l2_columns if include_l2 else [])
        violations = [column for column in features if column in FORBIDDEN or "future" in column.lower() or "target" in column.lower()]
        if violations:
            raise ValueError(f"leakage columns selected: {violations}")
        train = modeled.loc[modeled["split"].eq("train")].copy()
        calibration = modeled.loc[modeled["split"].eq("calibration")].copy()
        validation = modeled.loc[modeled["split"].eq("validation")].copy()
        if variant_mode == "market_mid":
            model = {"type": "market_mid_probability", "column": "pm_l2_1m_up_mid"}
            raw = {
                name: pd.to_numeric(part["pm_l2_1m_up_mid"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
                for name, part in (("train", train), ("calibration", calibration), ("validation", validation))
            }
        else:
            model = CatBoostClassifier(**config["model"])
            model.fit(train[features], train["target"].astype(int))
            raw = {
                name: pd.Series(model.predict_proba(part[features])[:, 1], index=part.index)
                for name, part in (("train", train), ("calibration", calibration), ("validation", validation))
            }
        calibrator_name, calibrator, calibration_comparison = _select_calibrator(raw["calibration"], calibration["target"].astype(int))
        calibrated = {name: calibrator.transform(values) for name, values in raw.items()}
        best, frontier = _threshold_search(validation["target"].astype(int), calibrated["validation"], config["threshold_search"])
        frontier.to_csv(variant_dir / "threshold_frontier.csv", index=False)
        probability_report = {}
        for name, part in (("train", train), ("calibration", calibration), ("validation", validation)):
            raw_metrics = _probability_metrics(part["target"].astype(int), raw[name])
            calibrated_metrics = _probability_metrics(part["target"].astype(int), calibrated[name])
            reliability, ece = _reliability(part["target"].astype(int), calibrated[name])
            probability_report[name] = {"raw": raw_metrics, "calibrated": calibrated_metrics, "ece": ece, "reliability": reliability}
            prediction_columns = [
                column for column in (
                    "market_slug", "market_t0", "timestamp", "feature_timestamp",
                    "decision_time", "condition_id", "endDate", "split", "target",
                ) if column in part
            ]
            predictions = part[prediction_columns].copy()
            predictions["raw_p_up"] = raw[name]
            predictions["calibrated_p_up"] = calibrated[name]
            predictions["p_up"] = calibrated[name]
            predictions["selected_side"] = _decisions(calibrated[name], best["selected_t_up"], best["selected_t_down"])
            predictions.to_parquet(variant_dir / f"predictions_{name}.parquet", index=False)
        with (variant_dir / "classifier.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        calibrator.save(variant_dir / f"{calibrator_name}.pkl")
        report = {
            "experiment_id": config["experiment_id"] + "_" + variant,
            "variant": variant,
            "include_l2": include_l2,
            "variant_mode": variant_mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "feature_count": len(features),
            "feature_schema_hash": hashlib.sha256("\n".join(features).encode()).hexdigest(),
            "feature_columns": features,
            "forbidden_feature_columns": violations,
            "fit_split": "train only",
            "probability_calibration_fit_split": "calibration only",
            "threshold_fit_split": "validation (optimistic acceptance)",
            "probability_calibrator": calibrator_name,
            "calibration_comparison": calibration_comparison,
            "probability_metrics": probability_report,
            "validation_metrics": best,
            "coverage_constraint_satisfied": bool(best["coverage"] >= config["threshold_search"]["min_coverage"]),
            "data_coverage": coverage,
            "windows": {
                name: {"row_count": len(part), "start": part["market_t0"].min(), "end": part["market_t0"].max()}
                for name, part in (("train", train), ("calibration", calibration), ("validation", validation))
            },
        }
        _atomic_json(report, variant_dir / "report.json")
        reports[variant] = report
    summary = {"experiment_id": config["experiment_id"], "config_path": str(config_path), "data_coverage": coverage, "variants": reports}
    _atomic_json(summary, output_root / "report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config)["data_coverage"], indent=2))


if __name__ == "__main__":
    main()
