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


def compute_binary_classification_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")

    y = y_true.astype(int)
    proba = probabilities.astype(float).clip(0.0, 1.0)
    predictions = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, pd.concat([1.0 - proba, proba], axis=1), labels=[0, 1])),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "positive_rate": float(proba.mean()) if not proba.empty else 0.0,
        "coverage": float(predictions.mean()) if not predictions.empty else 0.0,
        "sample_count": float(len(y)),
        "threshold": float(threshold),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def compute_classification_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    return compute_binary_classification_metrics(y_true, probabilities, threshold=threshold)


def compute_stage1_coverage(probabilities: pd.Series, threshold: float) -> float:
    if probabilities.empty:
        return 0.0
    return float((probabilities.astype(float) >= threshold).mean())


def compute_ks_distance(left: pd.Series, right: pd.Series) -> float:
    left_values = np.sort(left.dropna().astype(float).to_numpy())
    right_values = np.sort(right.dropna().astype(float).to_numpy())
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    support = np.sort(np.unique(np.concatenate([left_values, right_values])))
    left_cdf = np.searchsorted(left_values, support, side="right") / len(left_values)
    right_cdf = np.searchsorted(right_values, support, side="right") / len(right_values)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def evaluate_stage2_return_decisions(predicted_returns: pd.Series) -> pd.DataFrame:
    predictions = predicted_returns.astype(float)
    side = pd.Series("NONE", index=predictions.index, dtype="object")
    side.loc[predictions > 0.0] = "YES"
    side.loc[predictions < 0.0] = "NO"
    edge = predictions.abs().rename("edge")
    edge.loc[side == "NONE"] = 0.0
    return pd.DataFrame({"side": side, "edge": edge}, index=predictions.index)


def compute_return_direction_metrics(
    y_true: pd.Series,
    predicted_returns: pd.Series,
) -> dict[str, float]:
    if len(y_true) != len(predicted_returns):
        raise ValueError("y_true and predicted_returns must have the same length.")
    if predicted_returns.empty:
        return {"sample_count": 0.0}

    truth = y_true.astype(float)
    predictions = predicted_returns.astype(float)
    decisions = evaluate_stage2_return_decisions(predictions)
    trade_mask = decisions["side"] != "NONE"
    trade_count = int(trade_mask.sum())
    support_up = int((truth > 0.0).sum())
    support_down = int((truth < 0.0).sum())
    metrics: dict[str, float] = {
        "sample_count": float(len(truth)),
        "stage2_trade_count": float(trade_count),
        "coverage": float(trade_count / len(truth)) if len(truth) else 0.0,
        "predicted_return_mean": float(predictions.mean()),
        "predicted_return_p50": float(predictions.quantile(0.50)),
        "support_up": float(support_up),
        "support_down": float(support_down),
    }
    if trade_count == 0:
        metrics.update(
            {
                "direction_accuracy": 0.0,
                "trade_precision_up": 0.0,
                "trade_precision_down": 0.0,
                "trade_recall_up": 0.0,
                "trade_recall_down": 0.0,
                "class_pnl.up": 0.0,
                "class_pnl.down": 0.0,
                "trade_pnl.pnl_per_trade": 0.0,
                "trade_pnl.pnl_per_sample": 0.0,
            }
        )
        return metrics

    traded_truth = truth.loc[trade_mask]
    traded_side = decisions.loc[trade_mask, "side"]
    yes_mask = traded_side == "YES"
    no_mask = traded_side == "NO"
    yes_correct = (traded_truth.loc[yes_mask] > 0.0) if yes_mask.any() else pd.Series(dtype="bool")
    no_correct = (traded_truth.loc[no_mask] < 0.0) if no_mask.any() else pd.Series(dtype="bool")
    correct_count = int(yes_correct.sum() + no_correct.sum())
    precision_up = float(yes_correct.mean()) if len(yes_correct) else 0.0
    precision_down = float(no_correct.mean()) if len(no_correct) else 0.0
    recall_up = float(yes_correct.sum() / support_up) if support_up else 0.0
    recall_down = float(no_correct.sum() / support_down) if support_down else 0.0
    direction_accuracy = float(correct_count / trade_count)
    pnl_per_trade = float(2.0 * direction_accuracy - 1.0)
    metrics.update(
        {
            "direction_accuracy": direction_accuracy,
            "trade_precision_up": precision_up,
            "trade_precision_down": precision_down,
            "trade_recall_up": recall_up,
            "trade_recall_down": recall_down,
            "class_pnl.up": float(2.0 * precision_up - 1.0) if len(yes_correct) else 0.0,
            "class_pnl.down": float(2.0 * precision_down - 1.0) if len(no_correct) else 0.0,
            "trade_pnl.pnl_per_trade": pnl_per_trade,
            "trade_pnl.pnl_per_sample": float((trade_count / len(truth)) * pnl_per_trade) if len(truth) else 0.0,
        }
    )
    return metrics


def compute_two_stage_end_to_end_metrics(
    y_true: pd.Series,
    stage1_probabilities: pd.Series,
    stage2_predictions: pd.Series,
    *,
    stage1_threshold: float,
) -> dict[str, float]:
    if not (len(y_true) == len(stage1_probabilities) == len(stage2_predictions)):
        raise ValueError("End-to-end inputs must have the same length.")

    y = y_true.astype(float)
    p_active = stage1_probabilities.astype(float).clip(0.0, 1.0)
    active_mask = p_active >= stage1_threshold
    active_predictions = stage2_predictions.loc[active_mask]
    decisions = evaluate_stage2_return_decisions(active_predictions)
    trade_mask = decisions["side"] != "NONE"
    trade_count = int(trade_mask.sum())
    total_count = len(y)
    support_up = int((y > 0.0).sum())
    support_down = int((y < 0.0).sum())

    metrics: dict[str, float] = {
        "sample_count": float(total_count),
        "stage1_selected_count": float(active_mask.sum()),
        "stage1_selected_ratio": float(active_mask.mean()) if total_count else 0.0,
        "stage2_trade_count": float(trade_count),
        "coverage_end_to_end": float(trade_count / total_count) if total_count else 0.0,
        "stage1_threshold": float(stage1_threshold),
        "support_up": float(support_up),
        "support_down": float(support_down),
    }
    if trade_count == 0:
        metrics.update(
            {
                "trade_precision_up": 0.0,
                "trade_precision_down": 0.0,
                "trade_recall_up": 0.0,
                "trade_recall_down": 0.0,
                "class_pnl.up": 0.0,
                "class_pnl.down": 0.0,
                "trade_pnl.pnl_per_trade": 0.0,
                "trade_pnl.pnl_per_sample": 0.0,
            }
        )
        return metrics

    traded_truth = y.loc[active_mask].loc[trade_mask]
    traded_side = decisions.loc[trade_mask, "side"]
    yes_mask = traded_side == "YES"
    no_mask = traded_side == "NO"
    yes_correct = (traded_truth.loc[yes_mask] > 0.0) if yes_mask.any() else pd.Series(dtype="bool")
    no_correct = (traded_truth.loc[no_mask] < 0.0) if no_mask.any() else pd.Series(dtype="bool")
    correct_count = int(yes_correct.sum() + no_correct.sum())
    precision_up = float(yes_correct.mean()) if len(yes_correct) else 0.0
    precision_down = float(no_correct.mean()) if len(no_correct) else 0.0
    recall_up = float(yes_correct.sum() / support_up) if support_up else 0.0
    recall_down = float(no_correct.sum() / support_down) if support_down else 0.0
    trade_accuracy = float(correct_count / trade_count)
    pnl_per_trade = float(2.0 * trade_accuracy - 1.0)
    metrics.update(
        {
            "trade_precision_up": precision_up,
            "trade_precision_down": precision_down,
            "trade_recall_up": recall_up,
            "trade_recall_down": recall_down,
            "class_pnl.up": float(2.0 * precision_up - 1.0) if len(yes_correct) else 0.0,
            "class_pnl.down": float(2.0 * precision_down - 1.0) if len(no_correct) else 0.0,
            "trade_pnl.pnl_per_trade": pnl_per_trade,
            "trade_pnl.pnl_per_sample": float((trade_count / total_count) * pnl_per_trade) if total_count else 0.0,
        }
    )
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
