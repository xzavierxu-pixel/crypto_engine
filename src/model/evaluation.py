from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.constants import (
    DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN,
    DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN,
    DEFAULT_STAGE2_PROBABILITY_UP_COLUMN,
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


def evaluate_selective_binary_decisions(
    probabilities: pd.Series,
    *,
    t_up: float,
    t_down: float,
) -> pd.Series:
    p_up = probabilities.astype("float64")
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= t_up] = "UP"
    decisions.loc[p_up <= t_down] = "DOWN"
    return decisions


def compute_selective_binary_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    t_up: float,
    t_down: float,
) -> dict[str, float]:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")
    if t_down > t_up:
        raise ValueError("t_down must be <= t_up.")

    y = y_true.astype(int)
    p_up = probabilities.astype(float).clip(0.0, 1.0)
    decisions = evaluate_selective_binary_decisions(p_up, t_up=t_up, t_down=t_down)
    accepted = decisions != "ABSTAIN"
    up_mask = decisions == "UP"
    down_mask = decisions == "DOWN"
    accepted_count = int(accepted.sum())
    up_count = int(up_mask.sum())
    down_count = int(down_mask.sum())

    precision_up = float((y.loc[up_mask] == 1).mean()) if up_count else 0.0
    precision_down = float((y.loc[down_mask] == 0).mean()) if down_count else 0.0
    hard_predictions = (p_up >= 0.5).astype(int)
    accepted_correct = ((decisions.loc[accepted] == "UP") == (y.loc[accepted] == 1)) if accepted_count else pd.Series(dtype="bool")
    metrics = {
        "sample_count": float(len(y)),
        "coverage": float(accepted_count / len(y)) if len(y) else 0.0,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "balanced_precision": float((precision_up + precision_down) / 2.0),
        "all_sample_accuracy": float(accuracy_score(y, hard_predictions)) if len(y) else 0.0,
        "accepted_sample_accuracy": float(accepted_correct.mean()) if accepted_count else 0.0,
        "share_up_predictions": float(up_count / accepted_count) if accepted_count else 0.0,
        "share_down_predictions": float(down_count / accepted_count) if accepted_count else 0.0,
        "selected_t_up": float(t_up),
        "selected_t_down": float(t_down),
        "accepted_count": float(accepted_count),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p_up))
    labels = [0, 1]
    metrics["brier_score"] = float(brier_score_loss(y, p_up))
    metrics["log_loss"] = float(log_loss(y, pd.concat([1.0 - p_up, p_up], axis=1), labels=labels))
    return metrics


def search_selective_binary_thresholds(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    t_up_min: float,
    t_up_max: float,
    t_down_min: float,
    t_down_max: float,
    step: float,
    min_coverage: float,
    tie_tolerance: float,
    enforce_min_side_share: bool = False,
    min_side_share: float = 0.20,
    min_up_signals: int = 0,
    min_down_signals: int = 0,
    min_total_signals: int = 0,
) -> tuple[float, float, pd.DataFrame, dict[str, float | bool | str | None]]:
    if step <= 0:
        raise ValueError("threshold search step must be > 0.")
    records: list[dict[str, float]] = []
    eligible: list[dict[str, float]] = []
    up_candidates = [round(float(value), 6) for value in np.arange(t_up_min, t_up_max + step / 2.0, step)]
    down_candidates = [round(float(value), 6) for value in np.arange(t_down_min, t_down_max + step / 2.0, step)]

    for t_up in up_candidates:
        for t_down in down_candidates:
            if t_down > t_up:
                continue
            metrics = compute_selective_binary_metrics(y_true, probabilities, t_up=t_up, t_down=t_down)
            record = {
                "t_up": float(t_up),
                "t_down": float(t_down),
                **metrics,
            }
            records.append(record)
            side_share_ok = (
                not enforce_min_side_share
                or (
                    record["share_up_predictions"] >= min_side_share
                    and record["share_down_predictions"] >= min_side_share
                )
            )
            signal_counts_ok = (
                record["up_prediction_count"] >= min_up_signals
                and record["down_prediction_count"] >= min_down_signals
                and record["accepted_count"] >= min_total_signals
            )
            if record["coverage"] >= min_coverage and side_share_ok and signal_counts_ok:
                eligible.append(record)

    if not records:
        raise ValueError("threshold search produced no candidates.")
    pool = eligible if eligible else records
    best_precision = max(record["balanced_precision"] for record in pool)
    tied = [
        record
        for record in pool
        if record["balanced_precision"] >= best_precision - tie_tolerance
    ]
    best = max(tied, key=lambda record: (record["coverage"], record["balanced_precision"]))
    best_summary: dict[str, float | bool | str | None] = {
        "constraint_satisfied": bool(eligible),
        "fallback_reason": None if eligible else "no threshold set satisfied coverage/side-share/signal-count constraints",
        "t_up": float(best["t_up"]),
        "t_down": float(best["t_down"]),
        "balanced_precision": float(best["balanced_precision"]),
        "coverage": float(best["coverage"]),
        "precision_up": float(best["precision_up"]),
        "precision_down": float(best["precision_down"]),
        "up_prediction_count": float(best["up_prediction_count"]),
        "down_prediction_count": float(best["down_prediction_count"]),
        "accepted_count": float(best["accepted_count"]),
        "min_up_signals": float(min_up_signals),
        "min_down_signals": float(min_down_signals),
        "min_total_signals": float(min_total_signals),
    }
    return float(best["t_up"]), float(best["t_down"]), pd.DataFrame.from_records(records), best_summary


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


def compute_multiclass_classification_metrics(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    *,
    up_threshold: float | None = None,
    down_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> dict[str, float]:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")
    if probabilities.empty:
        return {"sample_count": 0.0}

    ordered = probabilities[
        [
            DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN,
            DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN,
            DEFAULT_STAGE2_PROBABILITY_UP_COLUMN,
        ]
    ].astype(float).clip(0.0, 1.0)
    y = y_true.astype(int)
    predictions = ordered.to_numpy().argmax(axis=1)
    supports = y.value_counts().to_dict()
    metrics = {
        "sample_count": float(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "multiclass_precision_up": float(precision_score(y, predictions, labels=[2], average="macro", zero_division=0)),
        "multiclass_precision_down": float(precision_score(y, predictions, labels=[0], average="macro", zero_division=0)),
        "multiclass_recall_up": float(recall_score(y, predictions, labels=[2], average="macro", zero_division=0)),
        "multiclass_recall_down": float(recall_score(y, predictions, labels=[0], average="macro", zero_division=0)),
        "support_down": float(supports.get(0, 0)),
        "support_flat": float(supports.get(1, 0)),
        "support_up": float(supports.get(2, 0)),
    }
    down_target = (y == 0).astype(int)
    up_target = (y == 2).astype(int)
    if down_target.nunique() == 2:
        metrics["down_auc"] = float(roc_auc_score(down_target, ordered[DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN]))
    if up_target.nunique() == 2:
        metrics["up_auc"] = float(roc_auc_score(up_target, ordered[DEFAULT_STAGE2_PROBABILITY_UP_COLUMN]))
    if up_threshold is not None and down_threshold is not None and margin_threshold is not None:
        metrics.update(
            compute_stage2_subset_trade_metrics(
                y_true,
                probabilities,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                margin_threshold=margin_threshold,
            )
        )
    return metrics


def compute_stage2_subset_trade_metrics(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    *,
    up_threshold: float,
    down_threshold: float,
    margin_threshold: float,
) -> dict[str, float]:
    if probabilities.empty:
        return {
            "class_pnl.up": 0.0,
            "class_pnl.down": 0.0,
            "trade_pnl.pnl_per_trade": 0.0,
            "trade_pnl.pnl_per_sample": 0.0,
            "stage2_trade_count": 0.0,
            "coverage": 0.0,
        }
    y = y_true.astype(int)
    decisions = evaluate_stage2_decisions(
        probabilities,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        margin_threshold=margin_threshold,
    )
    trade_mask = decisions["side"] != "NONE"
    trade_count = int(trade_mask.sum())
    if trade_count == 0:
        return {
            "class_pnl.up": 0.0,
            "class_pnl.down": 0.0,
            "trade_pnl.pnl_per_trade": 0.0,
            "trade_pnl.pnl_per_sample": 0.0,
            "stage2_trade_count": 0.0,
            "coverage": 0.0,
        }
    traded_truth = y.loc[trade_mask]
    traded_side = decisions.loc[trade_mask, "side"]
    yes_mask = traded_side == "YES"
    no_mask = traded_side == "NO"
    yes_correct = (traded_truth.loc[yes_mask] == 2) if yes_mask.any() else pd.Series(dtype="bool")
    no_correct = (traded_truth.loc[no_mask] == 0) if no_mask.any() else pd.Series(dtype="bool")
    trade_accuracy = float((yes_correct.sum() + no_correct.sum()) / trade_count)
    pnl_per_trade = float(2.0 * trade_accuracy - 1.0)
    return {
        "class_pnl.up": float(2.0 * float(yes_correct.mean()) - 1.0) if len(yes_correct) else 0.0,
        "class_pnl.down": float(2.0 * float(no_correct.mean()) - 1.0) if len(no_correct) else 0.0,
        "trade_pnl.pnl_per_trade": pnl_per_trade,
        "trade_pnl.pnl_per_sample": float((trade_count / len(y)) * pnl_per_trade) if len(y) else 0.0,
        "stage2_trade_count": float(trade_count),
        "coverage": float(trade_count / len(y)) if len(y) else 0.0,
    }


def evaluate_stage2_decisions(
    probabilities: pd.DataFrame,
    up_threshold: float,
    down_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    ordered = probabilities[
        [
            DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN,
            DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN,
            DEFAULT_STAGE2_PROBABILITY_UP_COLUMN,
        ]
    ].astype(float)
    p_down = ordered[DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN]
    p_up = ordered[DEFAULT_STAGE2_PROBABILITY_UP_COLUMN]
    up_margin = p_up - p_down
    down_margin = p_down - p_up
    yes_ok = (p_up >= up_threshold) & (up_margin >= margin_threshold)
    no_ok = (p_down >= down_threshold) & (down_margin >= margin_threshold)

    side = pd.Series("NONE", index=ordered.index, dtype="object")
    side.loc[yes_ok] = "YES"
    side.loc[no_ok] = "NO"
    both = yes_ok & no_ok
    side.loc[both] = np.where(up_margin.loc[both] >= down_margin.loc[both], "YES", "NO")
    edge = pd.Series(0.0, index=ordered.index, dtype="float64")
    edge.loc[side == "YES"] = up_margin.loc[side == "YES"]
    edge.loc[side == "NO"] = down_margin.loc[side == "NO"]
    return pd.DataFrame({"side": side, "edge": edge}, index=ordered.index)


def compute_two_stage_end_to_end_metrics(
    y_true: pd.Series,
    stage1_probabilities: pd.Series,
    stage2_probabilities: pd.DataFrame,
    *,
    stage1_threshold: float,
    up_threshold: float,
    down_threshold: float,
    margin_threshold: float,
) -> dict[str, float]:
    if not (len(y_true) == len(stage1_probabilities) == len(stage2_probabilities)):
        raise ValueError("End-to-end inputs must have the same length.")

    y = y_true.astype(int)
    p_active = stage1_probabilities.astype(float).clip(0.0, 1.0)
    active_mask = p_active >= stage1_threshold
    active_probabilities = stage2_probabilities.loc[active_mask]
    decisions = evaluate_stage2_decisions(
        active_probabilities,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        margin_threshold=margin_threshold,
    )
    trade_mask = decisions["side"] != "NONE"
    trade_count = int(trade_mask.sum())
    total_count = len(y)
    support_up = int((y == 2).sum())
    support_down = int((y == 0).sum())

    metrics: dict[str, float] = {
        "sample_count": float(total_count),
        "stage1_selected_count": float(active_mask.sum()),
        "stage1_selected_ratio": float(active_mask.mean()) if total_count else 0.0,
        "stage2_trade_count": float(trade_count),
        "coverage_end_to_end": float(trade_count / total_count) if total_count else 0.0,
        "stage1_threshold": float(stage1_threshold),
        "up_threshold": float(up_threshold),
        "down_threshold": float(down_threshold),
        "margin_threshold": float(margin_threshold),
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
    yes_correct = (traded_truth.loc[yes_mask] == 2) if yes_mask.any() else pd.Series(dtype="bool")
    no_correct = (traded_truth.loc[no_mask] == 0) if no_mask.any() else pd.Series(dtype="bool")
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
