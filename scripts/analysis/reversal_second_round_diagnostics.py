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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import compute_decision_metrics, compute_reversal_continuation_metrics  # noqa: E402


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_second_round_diagnostics.yaml")
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


def _decision_from_probability(p_up: pd.Series, *, t_up: float, t_down: float) -> pd.Series:
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= float(t_up)] = "UP"
    decisions.loc[p_up <= float(t_down)] = "DOWN"
    return decisions


def _threshold_values(search: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    step = float(search["step"])
    up = np.round(np.arange(float(search["t_up_min"]), float(search["t_up_max"]) + step / 2.0, step), 6)
    down = np.round(np.arange(float(search["t_down_min"]), float(search["t_down_max"]) + step / 2.0, step), 6)
    return up, down


def _search_probability(
    frame: pd.DataFrame,
    p_up: pd.Series,
    search: dict[str, Any],
    *,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for t_up in _threshold_values(search)[0]:
        for t_down in _threshold_values(search)[1]:
            if float(t_down) >= float(t_up):
                continue
            decisions = _decision_from_probability(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                frame["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(frame, decisions))
            row = {"t_up": float(t_up), "t_down": float(t_down), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["utility"] > 0.0:
                eligible.append(row)
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["accepted_sample_accuracy"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
        ),
    )
    best["constraint_satisfied"] = bool(eligible)
    best["objective"] = "accepted_sample_accuracy"
    best["hard_constraint"] = "coverage_only"
    return pd.DataFrame.from_records(records), best


def _load_artifact_predictions(paths: list[Path], split: str) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for path in paths:
        predictions = pd.read_parquet(path / f"{split}_predictions.parquet")
        join_keys = ["grid_id"] if "grid_id" in predictions.columns else ["timestamp"]
        selected = predictions[[*join_keys, "p_up", "decision"]].copy()
        prefix = path.name
        selected = selected.rename(columns={"p_up": f"{prefix}_p_up", "decision": f"{prefix}_decision"})
        selected[f"{prefix}_accepted"] = (selected[f"{prefix}_decision"] != "ABSTAIN").astype("float64")
        selected = selected.drop(columns=[f"{prefix}_decision"])
        if base is None:
            keep = [*join_keys, "target", "first_minute_side", "first_minute_return"]
            base = predictions[keep].copy()
        base = base.merge(selected, on=join_keys, how="inner", validate="one_to_one")
    if base is None:
        raise ValueError("no prediction stack experiments configured")
    return base


def _augment_prediction_stack(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    fm_yes = out["first_minute_side"].astype("object") == "YES"
    out["meta_fm_yes"] = fm_yes.astype("float64")
    out["meta_fm_ret"] = pd.to_numeric(out["first_minute_return"], errors="coerce").fillna(0.0)
    out["meta_fm_abs_ret"] = out["meta_fm_ret"].abs()
    for column in [name for name in out.columns if name.endswith("_p_up")]:
        out[f"{column}_confidence"] = (out[column].astype("float64") - 0.5).abs()
        out[f"{column}_follow_probability"] = out[column].where(fm_yes, 1.0 - out[column])
    features = [
        column
        for column in out.columns
        if column
        not in {
            "grid_id",
            "timestamp",
            "target",
            "first_minute_side",
            "first_minute_return",
        }
    ]
    return out, features


def _fit_lgbm(x: pd.DataFrame, y: pd.Series, config: dict[str, Any]) -> lgb.LGBMClassifier:
    params = dict(config["model"])
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    model = lgb.LGBMClassifier(**params)
    model.fit(x, y.astype(int))
    return model


def _run_prediction_stack(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    paths = [Path(path) for path in config["prediction_stack_experiments"]]
    train, train_features = _augment_prediction_stack(_load_artifact_predictions(paths, "train"))
    validation, validation_features = _augment_prediction_stack(_load_artifact_predictions(paths, "validation"))
    if train_features != validation_features:
        raise ValueError("prediction stack train/validation features do not match")
    results: list[dict[str, Any]] = []
    frontiers: dict[str, pd.DataFrame] = {}
    for mode in ("final_direction", "follow_relative"):
        if mode == "final_direction":
            y_train = train["target"].astype(int)
        else:
            y_train = ((train["first_minute_side"].astype("object") == "YES") == (train["target"].astype(int) == 1)).astype(int)
        model = _fit_lgbm(train[train_features], y_train, config)
        p_up = pd.Series(model.predict_proba(validation[validation_features])[:, 1], index=validation.index)
        if mode == "follow_relative":
            fm_yes = validation["first_minute_side"].astype("object") == "YES"
            p_up = p_up.where(fm_yes, 1.0 - p_up).clip(0.0, 1.0)
        frontier, best = _search_probability(
            validation,
            p_up,
            config["threshold_search"],
            min_coverage=float(config["objective"]["min_coverage"]),
        )
        results.append({"variant": f"prediction_stack_{mode}", "validation_metrics": best})
        frontiers[f"prediction_stack_{mode}"] = frontier
    return results, frontiers


def _load_baseline(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    path = Path(config["baseline_experiment"])
    manifest = _read_json(path / "artifact_manifest.json")
    return (
        pd.read_parquet(path / "development_frame.parquet"),
        pd.read_parquet(path / "validation_frame.parquet"),
        pd.read_parquet(path / "train_predictions.parquet"),
        pd.read_parquet(path / "validation_predictions.parquet"),
        list(manifest["feature_columns"]),
    )


def _select_features(all_features: list[str], config: dict[str, Any]) -> list[str]:
    patterns = [re.compile(pattern) for pattern in config["feature_set"]["patterns"]]
    safe = [name for name in all_features if name not in LEAKAGE_BLOCKLIST and "target" not in name.lower()]
    return [name for name in safe if any(pattern.search(name) for pattern in patterns)]


def _augment_correctness_features(frame: pd.DataFrame, predictions: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.loc[:, features].copy()
    p_up = pd.to_numeric(predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    side = predictions["decision"].replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
    fm_side = predictions["first_minute_side"].astype("object")
    out["meta_p_up"] = p_up
    out["meta_confidence"] = (p_up - 0.5).abs()
    out["meta_decision_up"] = (predictions["decision"] == "UP").astype("float64")
    out["meta_decision_down"] = (predictions["decision"] == "DOWN").astype("float64")
    out["meta_side_eq_first_minute"] = (side == fm_side).astype("float64")
    out["meta_first_minute_yes"] = (fm_side == "YES").astype("float64")
    out["meta_first_minute_return"] = pd.to_numeric(predictions["first_minute_return"], errors="coerce").fillna(0.0)
    out["meta_first_minute_abs_return"] = out["meta_first_minute_return"].abs()
    return out


def _run_correctness_selector(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_baseline(config)
    features = _select_features(all_features, config)
    train_candidates = train_predictions["decision"] != "ABSTAIN"
    validation_candidates = validation_predictions["decision"] != "ABSTAIN"
    train_x = _augment_correctness_features(train_frame, train_predictions, features).loc[train_candidates]
    y_train = (
        (train_predictions.loc[train_candidates, "decision"] == "UP")
        == (train_predictions.loc[train_candidates, "target"].astype(int) == 1)
    ).astype(int)
    model = _fit_lgbm(train_x, y_train, config)
    validation_x = _augment_correctness_features(validation_frame, validation_predictions, features)
    score = pd.Series(model.predict_proba(validation_x)[:, 1], index=validation_predictions.index)
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for cutoff in np.round(np.arange(0.0, 1.0001, 0.005), 6):
        decisions = pd.Series("ABSTAIN", index=validation_predictions.index, dtype="object")
        keep = validation_candidates & (score >= float(cutoff))
        decisions.loc[keep] = validation_predictions.loc[keep, "decision"]
        metrics = compute_decision_metrics(
            validation_predictions["target"],
            validation_predictions["p_up"],
            decisions,
            selected_t_up=float(cutoff),
            selected_t_down=0.0,
        )
        metrics.update(compute_reversal_continuation_metrics(validation_predictions, decisions))
        row = {"cutoff": float(cutoff), **metrics}
        records.append(row)
        if metrics["coverage"] >= float(config["objective"]["min_coverage"]) and metrics["utility"] > 0.0:
            eligible.append(row)
    best = max(
        eligible if eligible else records,
        key=lambda row: (
            row["accepted_sample_accuracy"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
        ),
    )
    best["constraint_satisfied"] = bool(eligible)
    return {"variant": "baseline_signal_correctness_selector", "validation_metrics": best}, pd.DataFrame.from_records(records)


def _run_price_diagnostics(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    _, _, _, validation_predictions, _ = _load_baseline(config)
    prices = pd.read_parquet(config["price_store"])
    prices = prices[prices["price_status"].eq("ok")].copy()
    prices["market_t0"] = pd.to_datetime(prices["market_t0"], utc=True)
    validation = validation_predictions.copy()
    validation["market_t0"] = pd.to_datetime(validation["market_t0"], utc=True)
    joined = validation.merge(
        prices[["market_t0", "yes_mid_price", "seconds_from_t0"]],
        on="market_t0",
        how="left",
        validate="one_to_one",
    )
    availability = {
        "sample_count": int(len(joined)),
        "missing_price_samples": int(joined["yes_mid_price"].isna().sum()),
        "seconds_from_t0_describe": prices["seconds_from_t0"].describe().to_dict(),
    }
    results: list[dict[str, Any]] = []
    frontiers: dict[str, pd.DataFrame] = {}
    for max_seconds in config["price_threshold_search"]["max_seconds_from_t0"]:
        frame = joined[joined["yes_mid_price"].notna() & (joined["seconds_from_t0"] <= float(max_seconds))].copy()
        frame = frame.reset_index(drop=True)
        p_up = frame["yes_mid_price"].astype("float64").clip(0.0, 1.0)
        frontier, best = _search_probability(
            frame,
            p_up,
            config["price_threshold_search"],
            min_coverage=float(config["objective"]["min_coverage"]),
        )
        name = f"price_only_max_seconds_{int(max_seconds)}"
        results.append({"variant": name, "validation_metrics": best, "sample_count_after_price_filter": int(len(frame))})
        frontiers[name] = frontier
    return results, frontiers, availability


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_results, stack_frontiers = _run_prediction_stack(config)
    correctness_result, correctness_frontier = _run_correctness_selector(config)
    price_results, price_frontiers, price_availability = _run_price_diagnostics(config)
    all_results = [*stack_results, correctness_result, *price_results]
    frontiers = {**stack_frontiers, "baseline_signal_correctness_selector": correctness_frontier, **price_frontiers}
    for name, frontier in frontiers.items():
        frontier.to_csv(output_dir / f"{name}_frontier.csv", index=False)
    best = max(
        all_results,
        key=lambda row: (
            row["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"]),
            row["validation_metrics"]["accepted_sample_accuracy"],
            row["validation_metrics"]["selection_score"],
            row["validation_metrics"]["utility"],
        ),
    )
    summary = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "coverage": row["validation_metrics"]["coverage"],
                "accepted_sample_accuracy": row["validation_metrics"]["accepted_sample_accuracy"],
                "selection_score": row["validation_metrics"]["selection_score"],
                "utility": row["validation_metrics"]["utility"],
                "accepted_count": row["validation_metrics"]["accepted_count"],
                "target_met": bool(
                    row["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"])
                    and row["validation_metrics"]["accepted_sample_accuracy"]
                    >= float(config["objective"]["target_accepted_sample_accuracy"])
                ),
            }
            for row in all_results
        ]
    ).sort_values(["accepted_sample_accuracy", "selection_score"], ascending=False)
    summary_path = output_dir / "variant_summary.csv"
    summary.to_csv(summary_path, index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "primary_metric": "validation accepted_sample_accuracy with coverage >= min_coverage",
        "mode": config["mode"],
        "objective": config["objective"],
        "price_availability": price_availability,
        "best_variant": best["variant"],
        "validation_metrics": best["validation_metrics"],
        "variant_results": all_results,
        "variant_summary_path": str(summary_path),
        "target_met": bool(
            best["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"])
            and best["validation_metrics"]["accepted_sample_accuracy"]
            >= float(config["objective"]["target_accepted_sample_accuracy"])
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run second-round reversal diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": str(Path(report["variant_summary_path"]).with_name("report.json")),
                "target_met": report["target_met"],
                "best_variant": report["best_variant"],
                "coverage": metrics["coverage"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_count": metrics["accepted_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
