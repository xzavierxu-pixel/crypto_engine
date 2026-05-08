from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.dataset_builder import TrainingFrame


@dataclass(frozen=True)
class RollingWindowSpec:
    train_days: int
    validation_days: int
    fold_count: int
    step_days: int
    purge_rows: int = 1


@dataclass(frozen=True)
class RollingTrainingSplit:
    train_days: int
    fold_index: int
    development: TrainingFrame
    validation: TrainingFrame
    window: dict[str, Any]


def _slice_training_frame(training: TrainingFrame, mask: pd.Series) -> TrainingFrame:
    return TrainingFrame(
        frame=training.frame.loc[mask].copy().reset_index(drop=True),
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
    )


def build_recent_rolling_splits(
    training: TrainingFrame,
    *,
    train_days_list: list[int],
    validation_days: int,
    fold_count: int,
    step_days: int,
    purge_rows: int = 1,
) -> list[RollingTrainingSplit]:
    if not train_days_list:
        raise ValueError("train_days_list must not be empty.")
    if any(days <= 0 for days in train_days_list):
        raise ValueError("train_days_list values must be > 0.")
    if validation_days <= 0:
        raise ValueError("validation_days must be > 0.")
    if fold_count <= 0:
        raise ValueError("fold_count must be > 0.")
    if step_days <= 0:
        raise ValueError("step_days must be > 0.")
    if purge_rows < 0:
        raise ValueError("purge_rows must be >= 0.")
    if training.frame.empty:
        raise ValueError("Training frame is empty.")

    frame = training.frame.sort_values("timestamp").reset_index(drop=True)
    sorted_training = TrainingFrame(
        frame=frame,
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
    )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    latest_end = timestamps.max()
    splits: list[RollingTrainingSplit] = []

    for train_days in train_days_list:
        for fold_index in range(fold_count):
            valid_end = latest_end - pd.Timedelta(days=fold_index * step_days)
            valid_start = valid_end - pd.Timedelta(days=validation_days)
            train_start = valid_start - pd.Timedelta(days=train_days)

            train_mask = (timestamps >= train_start) & (timestamps < valid_start)
            if purge_rows:
                train_indices = frame.index[train_mask]
                if len(train_indices) > purge_rows:
                    train_mask.loc[train_indices[-purge_rows:]] = False
                else:
                    train_mask.loc[train_indices] = False
            valid_mask = (timestamps >= valid_start) & (timestamps <= valid_end)
            if not bool(train_mask.any()) or not bool(valid_mask.any()):
                continue

            development = _slice_training_frame(sorted_training, train_mask)
            validation = _slice_training_frame(sorted_training, valid_mask)
            splits.append(
                RollingTrainingSplit(
                    train_days=train_days,
                    fold_index=fold_index,
                    development=development,
                    validation=validation,
                    window={
                        "train_start": str(development.frame["timestamp"].min()),
                        "train_end": str(development.frame["timestamp"].max()),
                        "validation_start": str(validation.frame["timestamp"].min()),
                        "validation_end": str(validation.frame["timestamp"].max()),
                        "train_row_count": int(len(development.frame)),
                        "validation_row_count": int(len(validation.frame)),
                        "purge_rows": int(purge_rows),
                    },
                )
            )
    return splits


def summarize_binary_rolling_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"result_count": 0, "by_train_days": []}

    frame = pd.DataFrame(results)
    metric_columns = [
        "balanced_precision",
        "selection_score",
        "utility",
        "downside_risk",
        "coverage",
        "precision_up",
        "precision_down",
        "accepted_sample_accuracy",
        "roc_auc",
        "t_up",
        "t_down",
    ]
    summaries: list[dict[str, Any]] = []
    for train_days, group in frame.groupby("train_days", sort=True):
        entry: dict[str, Any] = {"train_days": int(train_days), "fold_count": int(len(group))}
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            entry[f"{column}_mean"] = float(values.mean())
            entry[f"{column}_min"] = float(values.min())
            entry[f"{column}_max"] = float(values.max())
        entry["constraint_pass_rate"] = float(pd.to_numeric(group["constraint_satisfied"], errors="coerce").fillna(0).mean())
        entry["side_guardrail_pass_rate"] = float(
            pd.to_numeric(group["side_guardrail_constraint_satisfied"], errors="coerce").fillna(0).mean()
        )
        summaries.append(entry)

    primary_metric = "selection_score" if "selection_score" in frame.columns else "balanced_precision"
    ranked = sorted(
        summaries,
        key=lambda item: (
            item.get(f"{primary_metric}_mean", float("-inf")),
            item.get("coverage_mean", float("-inf")),
            item.get(f"{primary_metric}_min", float("-inf")),
        ),
        reverse=True,
    )
    return {
        "result_count": int(len(results)),
        "best_train_days": ranked[0]["train_days"] if ranked else None,
        "by_train_days": summaries,
    }
