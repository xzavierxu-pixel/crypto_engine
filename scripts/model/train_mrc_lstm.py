from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.core.constants import (
    DEFAULT_ABS_RETURN_COLUMN,
    DEFAULT_SIGNED_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TIMESTAMP_COLUMN,
)
from src.core.validation import normalize_ohlcv_frame
from src.data.preprocess import filter_by_timerange
from src.labels.abs_return import build_abs_return_frame
from src.labels.registry import get_label_builder
from src.horizons.registry import get_horizon_spec
from src.model.dataset_sequence import (
    MRCSequenceDataset,
    build_sequence_sample_positions,
    load_split_sequence_frame,
    resolve_mrc_feature_columns,
)
from src.model.evaluation import compute_selective_binary_metrics, search_selective_binary_thresholds
from src.model.model_mrc_lstm import MRCLSTMClassifier


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _build_label_frame(kline_path: Path, settings_path: Path, horizon_name: str) -> pd.DataFrame:
    settings = load_settings(settings_path)
    horizon = get_horizon_spec(settings, horizon_name)
    kline = normalize_ohlcv_frame(pd.read_parquet(kline_path), timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
    label_builder = get_label_builder(horizon.label_builder)
    labels = label_builder.build(kline, settings, horizon, select_grid_only=True)
    abs_returns = build_abs_return_frame(kline, horizon)
    labels = labels.merge(abs_returns, on=DEFAULT_TIMESTAMP_COLUMN, how="left", validate="one_to_one")
    labels = filter_by_timerange(
        labels,
        start=settings.dataset.train_start,
        end=settings.dataset.train_end,
        timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
    )
    labels = labels.dropna(subset=[DEFAULT_TARGET_COLUMN, DEFAULT_ABS_RETURN_COLUMN, DEFAULT_SIGNED_RETURN_COLUMN])
    return labels.reset_index(drop=True)


def _split_recent(labels: pd.DataFrame, *, train_days: int, validation_days: int, purge_rows: int) -> tuple[np.ndarray, np.ndarray]:
    timestamps = pd.to_datetime(labels[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    valid_end = timestamps.max()
    valid_start = valid_end - pd.Timedelta(days=validation_days)
    train_start = valid_start - pd.Timedelta(days=train_days)
    train_mask = (timestamps >= train_start) & (timestamps < valid_start)
    valid_mask = timestamps >= valid_start
    train_indices = labels.index[train_mask].to_numpy()
    valid_indices = labels.index[valid_mask].to_numpy()
    if purge_rows and len(train_indices) > purge_rows:
        train_indices = train_indices[:-purge_rows]
    if len(train_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Not enough samples for requested train/validation split.")
    return train_indices, valid_indices


def _sample_weights(abs_returns: pd.Series, train_mask: np.ndarray, *, lower: float = 0.35, upper: float = 1.0) -> np.ndarray:
    values = abs_returns.astype("float64").to_numpy()
    median = float(np.nanmedian(values[train_mask]))
    if not np.isfinite(median) or median <= 0:
        return np.ones(len(values), dtype="float32")
    weights = np.clip(values / median, lower, upper)
    return weights.astype("float32")


def _standardize_in_place(values: np.ndarray, train_positions: np.ndarray, sequence_length: int) -> dict[str, list[float]]:
    start = int(train_positions.min()) - sequence_length + 1
    end = int(train_positions.max()) + 1
    train_values = values[start:end]
    mean = train_values.mean(axis=0, dtype="float64").astype("float32")
    std = train_values.std(axis=0, dtype="float64").astype("float32")
    std[std < 1e-6] = 1.0
    values -= mean
    values /= std
    return {"mean": mean.tolist(), "std": std.tolist()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for features, labels, weights in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = (criterion(logits, labels) * weights).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    embeddings_out: list[np.ndarray] = []
    with torch.no_grad():
        for features, _, _ in loader:
            features = features.to(device, non_blocking=True)
            logits, embeddings = model(features, return_embedding=True)
            logits_out.append(logits.detach().cpu().numpy())
            embeddings_out.append(embeddings.detach().cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(embeddings_out)


def _classification_metrics(y_true: np.ndarray, p_up: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p_up, 1e-6, 1.0 - 1e-6)
    predictions = (clipped >= 0.5).astype(int)
    return {
        "sample_count": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1])),
        "positive_rate": float(clipped.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone MRC-LSTM on existing second-level split stores.")
    parser.add_argument("--config", default="config/config_mrc_lstm.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_yaml(config_path)
    settings_path = Path(config["data"]["settings_path"])
    settings = load_settings(settings_path)
    sequence_length = int(config["task"]["sequence_length_seconds"])
    output_dir = Path(args.output_dir or config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _build_label_frame(Path(config["data"]["kline_1m_path"]), settings_path, "5m")
    train_indices, valid_indices = _split_recent(
        labels,
        train_days=int(config["validation"]["train_days"]),
        validation_days=int(config["validation"]["validation_days"]),
        purge_rows=int(config["validation"].get("purge_rows", 1)),
    )

    feature_columns = resolve_mrc_feature_columns(
        config["data"]["second_level_store_path"],
        max_features=int(config["data"].get("max_features", 64)),
    )
    feature_start = pd.to_datetime(labels[DEFAULT_TIMESTAMP_COLUMN].min(), utc=True) - pd.Timedelta(seconds=sequence_length)
    feature_end = pd.to_datetime(labels[DEFAULT_TIMESTAMP_COLUMN].max(), utc=True)
    sequence_frame = load_split_sequence_frame(
        config["data"]["second_level_store_path"],
        feature_columns=feature_columns,
        start=feature_start,
        end=feature_end,
    )
    positions, valid_position_mask = build_sequence_sample_positions(
        feature_timestamps=sequence_frame.timestamps,
        sample_timestamps=labels[DEFAULT_TIMESTAMP_COLUMN],
        sequence_length=sequence_length,
    )
    labels = labels.loc[valid_position_mask].reset_index(drop=True)
    original_indices = np.arange(len(valid_position_mask))[valid_position_mask]
    remap = {old: new for new, old in enumerate(original_indices)}
    train_indices = np.array([remap[index] for index in train_indices if index in remap], dtype="int64")
    valid_indices = np.array([remap[index] for index in valid_indices if index in remap], dtype="int64")
    if len(train_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("No train/validation samples remain after sequence alignment.")

    y = labels[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()
    sample_weights = (
        _sample_weights(labels[DEFAULT_ABS_RETURN_COLUMN], train_indices)
        if bool(config["training"].get("sample_weighting", True))
        else np.ones(len(labels), dtype="float32")
    )
    standardization = _standardize_in_place(sequence_frame.values, positions[train_indices], sequence_length)

    train_dataset = MRCSequenceDataset(
        values=sequence_frame.values,
        positions=positions[train_indices],
        labels=y[train_indices],
        sequence_length=sequence_length,
        sample_weights=sample_weights[train_indices],
    )
    valid_dataset = MRCSequenceDataset(
        values=sequence_frame.values,
        positions=positions[valid_indices],
        labels=y[valid_indices],
        sequence_length=sequence_length,
        sample_weights=np.ones(len(valid_indices), dtype="float32"),
    )
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device(args.device)
    model = MRCLSTMClassifier(
        input_dim=len(feature_columns),
        cnn_hidden_dim=int(config["model"]["cnn_hidden_dim"]),
        lstm_hidden_dim=int(config["model"]["lstm_hidden_dim"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        kernel_sizes=[int(value) for value in config["model"]["kernel_sizes"]],
        dropout=float(config["model"]["dropout"]),
        bidirectional=bool(config["model"].get("bidirectional", False)),
        embedding_dim=int(config["model"].get("embedding_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        valid_loss = _run_epoch(
            model,
            valid_loader,
            optimizer=None,
            device=device,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        valid_logits, _ = _predict(model, valid_loader, device)
        valid_probs = 1.0 / (1.0 + np.exp(-valid_logits))
        valid_auc = float(roc_auc_score(y[valid_indices], valid_probs))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_auc": valid_auc})
        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} valid_auc={valid_auc:.6f}")
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    train_logits, train_embeddings = _predict(model, train_eval_loader, device)
    valid_logits, valid_embeddings = _predict(model, valid_loader, device)
    train_probs = 1.0 / (1.0 + np.exp(-train_logits))
    valid_probs = 1.0 / (1.0 + np.exp(-valid_logits))
    train_metrics = _classification_metrics(y[train_indices], train_probs)
    valid_metrics = _classification_metrics(y[valid_indices], valid_probs)
    fixed_signal_metrics = compute_selective_binary_metrics(
        pd.Series(y[valid_indices]),
        pd.Series(valid_probs),
        t_up=float(config["signal"]["upper_threshold"]),
        t_down=float(config["signal"]["lower_threshold"]),
    )
    t_up, t_down, frontier, best_signal = search_selective_binary_thresholds(
        pd.Series(y[valid_indices]),
        pd.Series(valid_probs),
        t_up_min=float(settings.threshold_search.t_up_min),
        t_up_max=float(settings.threshold_search.t_up_max),
        t_down_min=float(settings.threshold_search.t_down_min),
        t_down_max=float(settings.threshold_search.t_down_max),
        step=float(settings.threshold_search.step),
        min_coverage=float(settings.objective.min_coverage),
        tie_tolerance=float(settings.objective.balanced_precision_tie_tolerance),
        enforce_min_side_share=bool(settings.threshold_search.enforce_min_side_share),
        min_side_share=float(settings.threshold_search.min_side_share),
        min_up_signals=int(settings.threshold_search.min_up_signals),
        min_down_signals=int(settings.threshold_search.min_down_signals),
        min_total_signals=int(settings.threshold_search.min_total_signals),
    )

    valid_output = labels.iloc[valid_indices][[DEFAULT_TIMESTAMP_COLUMN, DEFAULT_TARGET_COLUMN, DEFAULT_SIGNED_RETURN_COLUMN]].copy()
    valid_output = valid_output.rename(columns={DEFAULT_TARGET_COLUMN: "y_true", DEFAULT_SIGNED_RETURN_COLUMN: "future_return"})
    valid_output["sample_id"] = np.arange(len(valid_output))
    valid_output["p_up"] = valid_probs
    valid_output["logit"] = valid_logits
    valid_output["pred_label"] = (valid_probs >= 0.5).astype(int)
    valid_output["signal"] = np.where(valid_probs >= t_up, "UP", np.where(valid_probs <= t_down, "DOWN", "NO_TRADE"))
    valid_output["signal_correct"] = np.where(
        valid_output["signal"] == "NO_TRADE",
        np.nan,
        ((valid_output["signal"] == "UP") == (valid_output["y_true"] == 1)).astype(float),
    )
    valid_output.to_csv(output_dir / "validation_predictions.csv", index=False)

    embeddings = pd.DataFrame(
        valid_embeddings,
        columns=[f"mrc_lstm_embedding_{index + 1}" for index in range(valid_embeddings.shape[1])],
    )
    embeddings.insert(0, "sample_id", np.arange(len(embeddings)))
    embeddings.insert(0, DEFAULT_TIMESTAMP_COLUMN, valid_output[DEFAULT_TIMESTAMP_COLUMN].to_numpy())
    embeddings["mrc_lstm_p_up"] = valid_probs
    embeddings["mrc_lstm_logit"] = valid_logits
    embeddings["mrc_lstm_confidence"] = np.abs(valid_probs - 0.5)
    embeddings.to_parquet(output_dir / "mrc_lstm_embeddings.parquet", index=False)
    frontier.to_csv(output_dir / "threshold_frontier.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "standardization": standardization,
            "config": config,
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
        },
        output_dir / "model_checkpoint.pt",
    )
    shutil.copy2(config_path, output_dir / "config_mrc_lstm.yaml")
    metrics = {
        "best_epoch": int(best_epoch),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "train_window": {
            "row_count": int(len(train_indices)),
            "start": str(labels.iloc[train_indices][DEFAULT_TIMESTAMP_COLUMN].min()),
            "end": str(labels.iloc[train_indices][DEFAULT_TIMESTAMP_COLUMN].max()),
        },
        "validation_window": {
            "row_count": int(len(valid_indices)),
            "start": str(labels.iloc[valid_indices][DEFAULT_TIMESTAMP_COLUMN].min()),
            "end": str(labels.iloc[valid_indices][DEFAULT_TIMESTAMP_COLUMN].max()),
        },
        "train_metrics": train_metrics,
        "validation_metrics": valid_metrics,
        "fixed_signal_metrics": fixed_signal_metrics,
        "threshold_search": {
            "t_up": t_up,
            "t_down": t_down,
            "best": best_signal,
        },
        "history": history,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "validation_auc": valid_metrics["roc_auc"]}, indent=2))


if __name__ == "__main__":
    main()
