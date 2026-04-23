from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset_builder import TrainingFrame


@dataclass(frozen=True)
class TimeSplit:
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    purge_rows: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def valid_slice(self) -> slice:
        return slice(self.valid_start, self.valid_end)


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_index: int
    split: TimeSplit
    metrics: dict[str, float]
    validation_probabilities: pd.Series


def purged_chronological_time_window_split(
    training: TrainingFrame,
    validation_window_days: int,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TimeSplit]:
    if validation_window_days <= 0:
        raise ValueError("validation_window_days must be > 0.")
    if purge_rows < 0:
        raise ValueError("purge_rows must be >= 0.")

    frame = training.frame
    if frame.empty:
        raise ValueError("Training frame is empty.")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    valid_start_ts = timestamps.max() - pd.Timedelta(days=validation_window_days)
    valid_start = int(timestamps.searchsorted(valid_start_ts, side="left"))
    train_end = max(valid_start - purge_rows, 0)

    if valid_start <= 0 or valid_start >= len(frame):
        raise ValueError("Training frame is too small for the requested validation window.")
    if train_end <= 0:
        raise ValueError("Training frame is too small for the requested validation window and purge.")

    split = TimeSplit(
        train_start=0,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=len(frame),
        purge_rows=purge_rows,
    )

    X = training.X
    y = training.y.astype(int)
    return (
        X.iloc[split.train_slice],
        X.iloc[split.valid_slice],
        y.iloc[split.train_slice],
        y.iloc[split.valid_slice],
        split,
    )


def purged_chronological_split(
    training: TrainingFrame,
    validation_fraction: float = 0.2,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TimeSplit]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if purge_rows < 0:
        raise ValueError("purge_rows must be >= 0.")

    frame_length = len(training.frame)
    if frame_length < 3:
        raise ValueError("Training frame is too small for chronological split.")

    valid_size = max(1, int(frame_length * validation_fraction))
    valid_start = frame_length - valid_size
    train_end = valid_start - purge_rows

    if train_end <= 0 or valid_start >= frame_length:
        raise ValueError("Training frame is too small for the requested split.")

    split = TimeSplit(
        train_start=0,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=frame_length,
        purge_rows=purge_rows,
    )

    X = training.X
    y = training.y.astype(int)
    return (
        X.iloc[split.train_slice],
        X.iloc[split.valid_slice],
        y.iloc[split.train_slice],
        y.iloc[split.valid_slice],
        split,
    )


def build_walk_forward_splits(
    training: TrainingFrame,
    min_train_size: int,
    validation_size: int,
    step_size: int | None = None,
    purge_rows: int = 0,
) -> list[TimeSplit]:
    if min_train_size <= 0:
        raise ValueError("min_train_size must be > 0.")
    if validation_size <= 0:
        raise ValueError("validation_size must be > 0.")
    if purge_rows < 0:
        raise ValueError("purge_rows must be >= 0.")

    step = step_size or validation_size
    if step <= 0:
        raise ValueError("step_size must be > 0.")

    frame_length = len(training.frame)
    splits: list[TimeSplit] = []
    valid_start = min_train_size + purge_rows

    while valid_start + validation_size <= frame_length:
        split = TimeSplit(
            train_start=0,
            train_end=valid_start - purge_rows,
            valid_start=valid_start,
            valid_end=valid_start + validation_size,
            purge_rows=purge_rows,
        )
        if split.train_end <= split.train_start:
            break
        splits.append(split)
        valid_start += step

    return splits


def compute_classification_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")

    y = y_true.astype(int)
    proba = probabilities.astype(float).clip(0.0, 1.0)
    predictions = (proba >= threshold).astype(int)
    positive_rate = float(proba.mean()) if not proba.empty else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, pd.concat([1.0 - proba, proba], axis=1), labels=[0, 1])),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "positive_rate": positive_rate,
        "sample_count": float(len(y)),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def summarize_walk_forward(results: list[WalkForwardFoldResult]) -> dict[str, float | int]:
    if not results:
        return {"enabled": False, "fold_count": 0}

    metric_names = sorted({name for result in results for name in result.metrics})
    summary: dict[str, float | int] = {"enabled": True, "fold_count": len(results)}
    for metric_name in metric_names:
        values = [result.metrics[metric_name] for result in results if metric_name in result.metrics]
        if values:
            summary[f"{metric_name}_mean"] = float(mean(values))
            summary[f"{metric_name}_min"] = float(min(values))
            summary[f"{metric_name}_max"] = float(max(values))
    return summary


def compute_pnl_metrics(
    y_true: pd.Series,
    stage1_probabilities: pd.Series,
    stage2_probabilities: pd.Series,
    stage1_threshold: float,
    buy_threshold: float,
) -> dict[str, float]:
    if not (len(y_true) == len(stage1_probabilities) == len(stage2_probabilities)):
        raise ValueError("y_true, stage1_probabilities, and stage2_probabilities must have the same length.")

    y = y_true.astype(int)
    p_active = stage1_probabilities.astype(float).clip(0.0, 1.0)
    p_up = stage2_probabilities.astype(float).clip(0.0, 1.0)
    active_mask = p_active >= stage1_threshold
    coverage = float(active_mask.mean()) if len(active_mask) else 0.0

    metrics: dict[str, float] = {
        "coverage": coverage,
        "active_sample_count": float(active_mask.sum()),
        "buy_threshold": float(buy_threshold),
        "stage1_threshold": float(stage1_threshold),
    }
    if active_mask.any():
        active_true = y.loc[active_mask]
        active_pred = (p_up.loc[active_mask] >= buy_threshold).astype(int)
        trade_accuracy = float(accuracy_score(active_true, active_pred))
        pnl_per_trade = float(2.0 * trade_accuracy - 1.0)
        metrics.update(
            {
                "trade_accuracy": trade_accuracy,
                "pnl_per_trade": pnl_per_trade,
                "pnl_per_sample": float(coverage * pnl_per_trade),
                "buy_rate": float((active_pred == 1).mean()),
                "sell_rate": float((active_pred == 0).mean()),
            }
        )
    else:
        metrics.update(
            {
                "trade_accuracy": 0.0,
                "pnl_per_trade": 0.0,
                "pnl_per_sample": 0.0,
                "buy_rate": 0.0,
                "sell_rate": 0.0,
            }
        )

    realized = np.where(
        active_mask.to_numpy(),
        np.where((p_up >= buy_threshold).to_numpy() == y.to_numpy(), 1.0, -1.0),
        0.0,
    )
    pnl_series = pd.Series(realized)
    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    metrics["sharpe"] = float(pnl_series.mean() / pnl_series.std()) if pnl_series.std(ddof=0) > 0 else 0.0
    metrics["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else 0.0

    longest_losing_streak = 0
    current_streak = 0
    for value in pnl_series:
        if value < 0:
            current_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_streak)
        else:
            current_streak = 0
    metrics["longest_losing_streak"] = float(longest_losing_streak)
    return metrics
