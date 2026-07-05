#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW_1M = ROOT / "artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet"
SECOND_LEVEL = ROOT / "artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT"
OLD_DEPLOY = ROOT / "execution_engine/deploy/baseline"
TRAIN_MODEL = ROOT / "scripts/model/train_model.py"
TARGET_BUILDER = ROOT / "price_estimator/expected_return/build_expected_return_target.py"
HAZARD_TRAINER = ROOT / "price_estimator/expected_return/train_low_cdf_and_backtest.py"
BASE_H2_CONFIG = ROOT / ".arbor/sessions/20260703_sum_pnl_no_leak/bdev_h2.yaml"
JOINT_MODULE = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"

FOLDS = [
    ("w1", "2026-02-27", "2026-03-06"),
    ("w2", "2026-03-06", "2026-03-13"),
    ("w3", "2026-03-13", "2026-03-20"),
    ("w4", "2026-03-20", "2026-03-27"),
    ("w5", "2026-03-27", "2026-04-03"),
    ("w6", "2026-04-03", "2026-04-11"),
]
POLICY = {"q_model": "raw_tree_blend", "shrink": 0.0, "gc_floor": 0.85, "min_ev": 0.02}


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def write_yaml_once(path: Path, payload: dict) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def prepare_configs() -> tuple[Path, Path]:
    settings_path = SESSION / "configs/settings_2m.yaml"
    settings = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    settings["decision_alignment"].update(
        {"enabled": True, "mode": "delayed_feature_offset", "feature_offset_minutes": 2,
         "row_policy": "delayed_2m_synthetic_decision_row"}
    )
    write_yaml_once(settings_path, settings)

    target_path = SESSION / "configs/expected_return_2m.yaml"
    target = yaml.safe_load((ROOT / "price_estimator/expected_return/config.yaml").read_text(encoding="utf-8"))
    target["experiment_id"] = "20260704_two_minute_shift"
    target["paths"].update(
        {
            "deploy_artifact_dir": str((SESSION / "direction/deploy").relative_to(ROOT)).replace("\\", "/"),
            "feature_manifest": str((SESSION / "direction/deploy/artifact_manifest.json").relative_to(ROOT)).replace("\\", "/"),
            "train_dataset_source": str((SESSION / "direction/base_split/development_frame.parquet").relative_to(ROOT)).replace("\\", "/"),
            "validation_dataset_source": str((SESSION / "direction/base_split/validation_frame.parquet").relative_to(ROOT)).replace("\\", "/"),
            "experiment_dir": str(SESSION.relative_to(ROOT)).replace("\\", "/"),
            "train_dataset": str((SESSION / "data/expected_return_train.parquet").relative_to(ROOT)).replace("\\", "/"),
            "validation_dataset": str((SESSION / "data/expected_return_validation.parquet").relative_to(ROOT)).replace("\\", "/"),
            "reports_dir": str((SESSION / "target_reports").relative_to(ROOT)).replace("\\", "/"),
        }
    )
    target["target"].update({"include_start": True, "include_end": True})
    forbidden = set(target["features"].get("forbidden_columns", []))
    forbidden.add("stage1_sample_weight")
    target["features"]["forbidden_columns"] = sorted(forbidden)
    write_yaml_once(target_path, target)
    return settings_path, target_path


def build_direction_split_chunked(settings_path: Path, split_dir: Path) -> None:
    from src.core.config import load_settings
    from src.data.dataset_builder import build_training_frame
    from src.data.loaders import load_ohlcv_parquet
    from src.data.second_level_features import sample_second_level_feature_store
    from src.model.train import split_recent_train_validation_frame

    module_path = ROOT / "scripts/analysis/catboost_calendar_coordinate_search.py"
    spec = importlib.util.spec_from_file_location("coordinate_build", module_path)
    coordinate = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(coordinate)
    accepted_cfg = yaml.safe_load((ROOT / "experiments/configs/20260611_catboost_calendar_coordinate_search.yaml").read_text(encoding="utf-8"))

    def load_second_for_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
        stores = [SECOND_LEVEL / "second_features_kline", SECOND_LEVEL / "second_features_agg"]
        sampled_stores = []
        decision_ts = pd.to_datetime(decisions["timestamp"], utc=True)
        for store in stores:
            parts = []
            for day in pd.date_range(decision_ts.min().floor("D"), decision_ts.max().floor("D"), freq="D", tz="UTC"):
                path = store / f"date={day.strftime('%Y-%m-%d')}" / "second_features.parquet"
                if not path.exists():
                    continue
                mask = decision_ts.dt.floor("D") == day
                if not mask.any():
                    continue
                sampled = sample_second_level_feature_store(decisions.loc[mask], pd.read_parquet(path))
                sampled.index = decisions.loc[mask].index
                parts.append(sampled)
            if parts:
                sampled_stores.append(pd.concat(parts).sort_index().reset_index(drop=True))
        if not sampled_stores:
            return decisions.reset_index(drop=True)
        combined = sampled_stores[0]
        for sampled in sampled_stores[1:]:
            columns = [column for column in sampled.columns if column != "timestamp" and column not in combined.columns]
            combined = combined.merge(sampled[["timestamp", *columns]], on="timestamp", how="left")
        return combined
    source = load_ohlcv_parquet(RAW_1M)
    source_ts = pd.to_datetime(source["timestamp"], utc=True)
    chunks = []
    selected_features: list[str] | None = None
    starts = pd.date_range("2025-12-01", "2026-05-01", freq="MS", tz="UTC")
    for start in starts:
        end = min(start + pd.offsets.MonthBegin(1), pd.Timestamp("2026-05-11", tz="UTC"))
        warm_start = start - pd.Timedelta(days=1)
        raw_chunk = source.loc[(source_ts >= warm_start) & (source_ts < end)].copy()
        base_settings = load_settings(settings_path)
        chunk_settings = replace(
            base_settings,
            dataset=replace(
                base_settings.dataset,
                train_start=start.isoformat(),
                train_end=(end - pd.Timedelta(minutes=1)).isoformat(),
            ),
        )
        decision_rows = raw_chunk.loc[
            (pd.to_datetime(raw_chunk["timestamp"], utc=True).dt.minute % 5 == 2), ["timestamp"]
        ].copy()
        second = load_second_for_decisions(decision_rows)
        training = build_training_frame(raw_chunk, chunk_settings, horizon_name="5m", second_level_features_frame=second)
        if selected_features is None:
            selected_features = coordinate._select_features(training.feature_columns, accepted_cfg["feature_set"]["patterns"])
        metadata = [column for column in training.frame.columns if column not in training.feature_columns]
        keep = list(dict.fromkeys([*metadata, *selected_features]))
        chunks.append(training.frame[keep].copy())
        del raw_chunk, decision_rows, second, training
    frame = pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    feature_columns = list(selected_features or [])
    from src.data.dataset_builder import TrainingFrame
    development, validation = split_recent_train_validation_frame(
        TrainingFrame(frame=frame, feature_columns=feature_columns, sample_weight_column="stage1_sample_weight"),
        train_days=115, validation_days=30, purge_rows=1,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    development.frame.to_parquet(split_dir / "development_frame.parquet", index=False)
    validation.frame.to_parquet(split_dir / "validation_frame.parquet", index=False)
    write_json(split_dir / "artifact_manifest.json", {"feature_columns": feature_columns, "feature_count": len(feature_columns),
                                                       "feature_offset_minutes": 2, "build_mode": "monthly_chunked"})


def train_direction(settings_path: Path) -> None:
    split_dir = SESSION / "direction/base_split"
    if not (split_dir / "artifact_manifest.json").exists():
        build_direction_split_chunked(settings_path, split_dir)

    deploy_dir = SESSION / "direction/deploy"
    if (deploy_dir / "artifact_manifest.json").exists():
        return
    module_path = ROOT / "scripts/analysis/catboost_calendar_coordinate_search.py"
    spec = importlib.util.spec_from_file_location("coordinate", module_path)
    coordinate = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(coordinate)
    accepted_cfg = yaml.safe_load((ROOT / "experiments/configs/20260611_catboost_calendar_coordinate_search.yaml").read_text(encoding="utf-8"))
    train = pd.read_parquet(split_dir / "development_frame.parquet")
    validation = pd.read_parquet(split_dir / "validation_frame.parquet")
    train_pred, validation_pred = train, validation
    all_features = json.loads((split_dir / "artifact_manifest.json").read_text(encoding="utf-8"))["feature_columns"]
    features = coordinate._select_features(all_features, accepted_cfg["feature_set"]["patterns"])
    forbidden = [c for c in features if c in coordinate.LEAKAGE_BLOCKLIST or c.startswith("future_") or "sample_weight" in c.lower()]
    if forbidden:
        raise RuntimeError(f"forbidden direction features: {forbidden}")
    model = CatBoostClassifier(**accepted_cfg["model"])
    model.fit(train[features], train["target"].astype(int))
    old_manifest = json.loads((OLD_DEPLOY / "artifact_manifest.json").read_text(encoding="utf-8"))
    frozen = {k: (float(v["t_up"]), float(v["t_down"])) for k, v in old_manifest["threshold_policy"]["thresholds"].items()}
    train_p = pd.Series(model.predict_proba(train[features])[:, 1], index=train.index)
    validation_p = pd.Series(model.predict_proba(validation[features])[:, 1], index=validation.index)
    def frozen_metrics(frame: pd.DataFrame, probability: pd.Series) -> dict:
        decisions = coordinate._decisions(probability, coordinate._day_session(frame), frozen)
        metrics = coordinate.compute_decision_metrics(
            frame["target"], probability, decisions,
            selected_t_up=float(np.mean([value[0] for value in frozen.values()])),
            selected_t_down=float(np.mean([value[1] for value in frozen.values()])),
        )
        metrics["thresholds"] = json.dumps(
            {key: {"t_up": value[0], "t_down": value[1]} for key, value in frozen.items()}, sort_keys=True
        )
        return metrics
    train_metrics = frozen_metrics(train_pred, train_p)
    validation_metrics = frozen_metrics(validation_pred, validation_p)
    report_path = SESSION / "direction/report.json"
    write_json(report_path, {"experiment_id": "20260704_two_minute_shift_direction", "threshold_source": "frozen_accepted_manifest",
                             "train_metrics": train_metrics, "validation_metrics": validation_metrics,
                             "feature_count": len(features), "feature_offset_minutes": 2})
    coordinate._save_deploy_artifact(
        output_dir=deploy_dir, model=model, model_params=accepted_cfg["model"], feature_columns=features,
        train_metrics=train_metrics, validation_metrics=validation_metrics,
        train_window=coordinate._window_summary(train), validation_window=coordinate._window_summary(validation),
        source_config_path=SESSION / "configs/settings_2m.yaml", source_report_path=report_path,
    )
    manifest_path = deploy_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"experiment_id": "20260704_two_minute_shift_direction", "feature_offset_minutes": 2,
                     "threshold_source": "frozen_accepted_manifest"})
    write_json(manifest_path, manifest)


def build_targets(target_path: Path) -> None:
    expected = [SESSION / "data/expected_return_train.parquet", SESSION / "data/expected_return_validation.parquet"]
    if not all(path.exists() for path in expected):
        run([sys.executable, str(TARGET_BUILDER), "--config", str(target_path)])
    for path in expected:
        frame = pd.read_parquet(path, columns=["timestamp", "decision_time", "endDate", "feature_offset_minutes"])
        expected_decision = pd.to_datetime(frame["timestamp"], utc=True) + pd.Timedelta(minutes=2)
        actual_decision = pd.to_datetime(frame["decision_time"], utc=True)
        if not actual_decision.equals(expected_decision) or set(frame["feature_offset_minutes"].dropna().astype(int)) != {2}:
            raise RuntimeError(f"two-minute alignment audit failed: {path}")


def h2_config(train_path: Path, dev_path: Path, output: Path, experiment_id: str, epochs: int) -> Path:
    config_path = output / "config.yaml"
    base = yaml.safe_load(BASE_H2_CONFIG.read_text(encoding="utf-8"))
    base["experiment_id"] = experiment_id
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/")
    base["paths"].update(
        {"train_dataset": rel(train_path), "validation_dataset": rel(dev_path), "models_dir": rel(output / "models"),
         "reports_dir": rel(output / "reports"), "predictions_train": rel(output / "reports/predictions_train.parquet"),
         "predictions_calibration": rel(output / "reports/predictions_calibration.parquet"),
         "predictions_validation": rel(output / "reports/predictions_validation.parquet"),
         "target_summary_source": rel(SESSION / "target_reports/target_build_summary.json")}
    )
    base["training"].update({"epochs": epochs, "early_stop_patience": 6, "device": "cuda", "random_seed": 20260704})
    base["target"].update({"include_start": True, "include_end": True})
    write_yaml_once(config_path, base)
    return config_path


def load_joint():
    spec = importlib.util.spec_from_file_location("joint", JOINT_MODULE)
    joint = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(joint)
    return joint


def run_pretest() -> dict:
    source = SESSION / "data/expected_return_train.parquet"
    frame = pd.read_parquet(source)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    rows = []
    joint = load_joint()
    for name, start, end in FOLDS:
        fold = SESSION / "folds" / name
        train_path, dev_path = fold / "data/train.parquet", fold / "data/dev.parquet"
        if not train_path.exists() or not dev_path.exists():
            train_path.parent.mkdir(parents=True, exist_ok=True)
            start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
            frame.loc[ts < start_ts].to_parquet(train_path, index=False)
            frame.loc[(ts >= start_ts) & (ts < end_ts)].to_parquet(dev_path, index=False)
        cfg = h2_config(train_path, dev_path, fold, f"20260704_two_minute_{name}", 30)
        model_path = fold / "models/hazard_survival_cdf.pt"
        if not model_path.exists():
            run([sys.executable, str(HAZARD_TRAINER), "--config", str(cfg)])
        prepared = joint.prepare(train_path, dev_path, model_path, 20260704)
        metrics = joint.evaluate(prepared, POLICY["q_model"], POLICY["shrink"], POLICY["gc_floor"], POLICY["min_ev"])
        rows.append({"fold": name, "start": start, "end": end, **metrics,
                     "q_brier": prepared[5][POLICY["q_model"]]["brier"],
                     "q_log_loss": prepared[5][POLICY["q_model"]]["log_loss"]})
    table = pd.DataFrame(rows)
    table.to_csv(SESSION / "pretest_fixed_policy.csv", index=False)
    holdout = table.loc[table["fold"].isin(["w5", "w6"])]
    payload = {"policy": POLICY, "folds": rows, "tune_sum_pnl": float(table.iloc[:4]["sum_pnl"].sum()),
               "holdout_sum_pnl": float(holdout["sum_pnl"].sum()),
               "holdout_gate_passed": bool((holdout["sum_pnl"] > 0).all())}
    write_json(SESSION / "pretest_summary.json", payload)
    return payload


def run_btest() -> dict:
    pretest = json.loads((SESSION / "pretest_summary.json").read_text(encoding="utf-8"))
    if not pretest["holdout_gate_passed"]:
        raise RuntimeError("pretest holdout gate failed; B_test remains protected")
    train_path = SESSION / "data/expected_return_train.parquet"
    dev_path = SESSION / "data/expected_return_validation.parquet"
    full = SESSION / "full_h2"
    cfg = h2_config(train_path, dev_path, full, "20260704_two_minute_full_h2", 45)
    model_path = full / "models/hazard_survival_cdf.pt"
    if not model_path.exists():
        run([sys.executable, str(HAZARD_TRAINER), "--config", str(cfg)])
    joint = load_joint()
    prepared = joint.prepare(train_path, dev_path, model_path, 20260704)
    metrics = joint.evaluate(prepared, POLICY["q_model"], POLICY["shrink"], POLICY["gc_floor"], POLICY["min_ev"])
    payload = {"evaluation_kind": "two-minute shift frozen-policy validation milestone", "baseline_sum_pnl": 42.43,
               "policy": POLICY, "metrics": metrics, "q_calibration": prepared[5][POLICY["q_model"]],
               "btest_evaluation_count_this_session": 1}
    write_json(SESSION / "btest_milestone_once.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "pretest", "btest", "all"], default="all")
    args = parser.parse_args()
    settings, target = prepare_configs()
    train_direction(settings)
    build_targets(target)
    if args.phase == "prepare":
        return
    pretest = run_pretest()
    print(json.dumps(pretest, indent=2, allow_nan=True))
    if args.phase in {"btest", "all"}:
        print(json.dumps(run_btest(), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
