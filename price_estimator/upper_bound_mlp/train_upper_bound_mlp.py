#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import (  # noqa: E402
    apply_sample_filter,
    ensure_no_forbidden_features,
    load_config,
    load_deploy_manifest,
    load_feature_columns,
    resolve_path,
    write_json,
)


PRED_BASE_COLS = [
    "timestamp",
    "decision_time",
    "condition_id",
    "polymarket_slug",
    "selected_side",
    "p_up",
    "p_side",
    "p_bin",
    "p_side_bucket",
    "market_time_bucket",
    "target_raw",
    "lowest_trade_price_next4",
    "lowest_trade_time_next4",
    "time_to_lowest_trade_sec",
    "time_to_lowest_trade_sec_bucket",
]

DEFAULT_P_SIDE_BIN_EDGES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.8,
    0.9,
    1.0,
]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def feature_set(config: dict[str, Any], train: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns = load_feature_columns(config)
    added = list(config["features"]["added_columns"])
    available = [c for c in feature_columns if c in train.columns]
    missing = sorted(set(feature_columns) - set(available))
    if missing:
        raise ValueError(f"Missing deploy feature columns in price dataset: {missing[:20]} total={len(missing)}")
    columns = available + added
    ensure_no_forbidden_features(columns, list(config["features"]["forbidden_columns"]))
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in columns]
    return columns, cat_cols


@dataclass
class Preprocessor:
    columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    categories: dict[str, list[str]]
    output_columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str], categorical_columns: list[str]) -> "Preprocessor":
        cat_set = set(categorical_columns)
        numeric_columns = [c for c in columns if c not in cat_set]
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for col in numeric_columns:
            values = pd.to_numeric(df[col], errors="coerce").astype(float)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            if not math.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            medians[col] = median if math.isfinite(median) else 0.0
            means[col] = mean if math.isfinite(mean) else 0.0
            scales[col] = scale

        categories: dict[str, list[str]] = {}
        for col in categorical_columns:
            values = df[col].astype("string").fillna("missing")
            cats = sorted(str(v) for v in values.unique().tolist())
            if "missing" not in cats:
                cats.append("missing")
            categories[col] = cats

        output_columns = list(numeric_columns)
        for col in categorical_columns:
            output_columns.extend([f"{col}={cat}" for cat in categories[col]])
        return cls(columns, categorical_columns, numeric_columns, medians, means, scales, categories, output_columns)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col in self.numeric_columns:
            values = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(self.medians[col]).to_numpy()
            blocks.append(((values - self.means[col]) / self.scales[col]).reshape(-1, 1))
        for col in self.categorical_columns:
            values = df[col].astype("string").fillna("missing").astype(str)
            cats = self.categories[col]
            mapping = {cat: idx for idx, cat in enumerate(cats)}
            arr = np.zeros((len(df), len(cats)), dtype=np.float32)
            idx = values.map(mapping).fillna(mapping.get("missing", 0)).astype(int).to_numpy()
            arr[np.arange(len(df)), idx] = 1.0
            blocks.append(arr)
        if not blocks:
            raise ValueError("No model features available")
        return np.hstack(blocks).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "categories": self.categories,
            "output_columns": self.output_columns,
        }


class IndexedTensorDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p_side: np.ndarray,
        constraint_weight: np.ndarray,
        tightness_weight: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        self.p_side = torch.from_numpy(p_side.astype(np.float32)).view(-1, 1)
        self.constraint_weight = torch.from_numpy(constraint_weight.astype(np.float32)).view(-1, 1)
        self.tightness_weight = torch.from_numpy(tightness_weight.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[idx],
            self.y[idx],
            self.p_side[idx],
            self.constraint_weight[idx],
            self.tightness_weight[idx],
        )


class UpperBoundMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: list[float]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for i, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            if i < 2:
                layers.append(nn.LayerNorm(hidden))
            drop = float(dropout[i]) if i < len(dropout) else 0.0
            if drop > 0:
                layers.append(nn.Dropout(drop))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predictions_from_logits(z: np.ndarray, df: pd.DataFrame | None, config: dict[str, Any]) -> np.ndarray:
    p = sigmoid_np(z)
    output_config = config.get("prediction_output", {})
    output_type = str(output_config.get("type", "sigmoid"))
    if output_type == "sigmoid":
        return p
    if output_type == "p_side_sigmoid":
        if df is None or "p_side" not in df.columns:
            raise ValueError("prediction_output.type=p_side_sigmoid requires p_side column")
        slack = float(output_config.get("cap_slack", 0.0))
        cap = np.clip(pd.to_numeric(df["p_side"], errors="coerce").to_numpy(dtype=float) + slack, 0.0, 1.0)
        return np.clip(cap * p, 0.0, 1.0)
    raise ValueError(f"Unsupported prediction_output.type: {output_type}")


def upper_bound_metrics(
    y: np.ndarray,
    z: np.ndarray,
    epsilon: float,
    tolerance: float,
    config: dict[str, Any] | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, float]:
    config = config or {}
    p = predictions_from_logits(z, df, config)
    gap = p - y
    violation = np.maximum(y + epsilon - p, 0.0)
    violating = p + tolerance < y + epsilon
    gap_p05 = float(np.quantile(gap, 0.05)) if len(y) else float("nan")
    gap_p95 = float(np.quantile(gap, 0.95)) if len(y) else float("nan")
    return {
        "sample_count": float(len(y)),
        "mean_gap": float(np.mean(gap)) if len(y) else float("nan"),
        "median_gap": float(np.median(gap)) if len(y) else float("nan"),
        "gap_p05": gap_p05,
        "gap_p95": gap_p95,
        "gap_p95_p05_range": gap_p95 - gap_p05 if len(y) else float("nan"),
        "min_gap": float(np.min(gap)) if len(y) else float("nan"),
        "violation_rate": float(np.mean(violating)) if len(y) else float("nan"),
        "max_violation": float(np.max(violation)) if len(y) else float("nan"),
        "p90_violation": float(np.quantile(violation, 0.90)) if len(y) else float("nan"),
        "p95_violation": float(np.quantile(violation, 0.95)) if len(y) else float("nan"),
        "p99_violation": float(np.quantile(violation, 0.99)) if len(y) else float("nan"),
        "coverage": float(np.mean(~violating)) if len(y) else float("nan"),
    }


def torch_log_cosh(x: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = x / scale
    abs_scaled = torch.abs(scaled)
    return scale * scale * (abs_scaled + torch.log1p(torch.exp(-2.0 * abs_scaled)) - math.log(2.0))


def batch_predictions_from_logits(z: torch.Tensor, p_side: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    p = torch.sigmoid(z)
    output_config = config.get("prediction_output", {})
    output_type = str(output_config.get("type", "sigmoid"))
    if output_type == "sigmoid":
        return p
    if output_type == "p_side_sigmoid":
        slack = float(output_config.get("cap_slack", 0.0))
        return torch.clamp(p_side + slack, min=0.0, max=1.0) * p
    raise ValueError(f"Unsupported prediction_output.type: {output_type}")


def upper_bound_loss(z: torch.Tensor, y: torch.Tensor, p_side: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    loss_config = config.get("loss", {})
    loss_type = str(loss_config.get("type", "mean_gap_soft_violation"))
    epsilon = float(config["target"]["epsilon"])
    clip_min = float(config["target"]["clip_min"])
    clip_max = float(config["target"]["clip_max"])
    scale = float(loss_config.get("scale", 0.05))
    p = batch_predictions_from_logits(z, p_side, config)
    target = torch.clamp(y + epsilon, min=clip_min, max=clip_max)

    if loss_type == "asymmetric_logcosh":
        under = torch.relu(target - p)
        over = torch.relu(p - target)
        w_under = float(loss_config.get("w_under", 5.0))
        w_over = float(loss_config.get("w_over", 1.0))
        return (w_under * torch_log_cosh(under, scale) + w_over * torch_log_cosh(over, scale)).mean()

    if loss_type == "mean_gap_soft_violation":
        gap = p - y
        violation = torch.relu(target - p)
        penalty = float(loss_config.get("violation_penalty", loss_config.get("C", 10.0)))
        return gap.mean() + penalty * torch_log_cosh(violation, scale).mean()

    raise ValueError(f"Unsupported loss.type: {loss_type}")


def binned_weights_from_y(y: np.ndarray, weighting: dict[str, Any]) -> np.ndarray:
    if not weighting.get("enabled", False):
        return np.ones(len(y), dtype=np.float32)

    edges = np.asarray([float(v) for v in weighting["price_bin_edges"]], dtype=np.float32)
    weights = np.asarray([float(v) for v in weighting["price_bin_weights"]], dtype=np.float32)
    if len(edges) < 2:
        raise ValueError("constraint_weighting.price_bin_edges must contain at least two values")
    if len(weights) != len(edges) - 1:
        raise ValueError("constraint_weighting.price_bin_weights must have len(price_bin_edges) - 1 values")

    idx = np.searchsorted(edges, y, side="right") - 1
    idx = np.clip(idx, 0, len(weights) - 1)
    out = weights[idx].astype(np.float32)
    if bool(weighting.get("normalize_mean", True)) and len(out):
        mean = float(np.mean(out))
        if math.isfinite(mean) and mean > 1e-12:
            out = out / mean
    return out.astype(np.float32)


def constraint_weights_from_y(y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return binned_weights_from_y(y, config.get("constraint_weighting", {}))


def tightness_weights_from_y(y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return binned_weights_from_y(y, config.get("tightness_weighting", {}))


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        out.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def is_better(
    candidate_validation: dict[str, float],
    incumbent_validation: dict[str, float] | None,
    min_validation_coverage: float,
    min_validation_mean_gap: float = float("-inf"),
    max_validation_violation: float | None = None,
) -> bool:
    if incumbent_validation is None:
        return True
    candidate_ok = candidate_validation["coverage"] >= min_validation_coverage
    incumbent_ok = incumbent_validation["coverage"] >= min_validation_coverage
    candidate_gap_ok = candidate_validation["mean_gap"] >= min_validation_mean_gap
    incumbent_gap_ok = incumbent_validation["mean_gap"] >= min_validation_mean_gap
    if candidate_gap_ok != incumbent_gap_ok:
        return candidate_gap_ok
    if candidate_ok != incumbent_ok:
        return candidate_ok
    if candidate_ok and incumbent_ok and max_validation_violation is not None:
        candidate_max_ok = candidate_validation["max_violation"] <= max_validation_violation
        incumbent_max_ok = incumbent_validation["max_violation"] <= max_validation_violation
        if candidate_max_ok != incumbent_max_ok:
            return candidate_max_ok
    if candidate_ok and incumbent_ok:
        if candidate_validation["mean_gap"] != incumbent_validation["mean_gap"]:
            return candidate_validation["mean_gap"] < incumbent_validation["mean_gap"]
        if candidate_validation.get("gap_p95_p05_range") != incumbent_validation.get("gap_p95_p05_range"):
            return candidate_validation.get("gap_p95_p05_range", float("inf")) < incumbent_validation.get(
                "gap_p95_p05_range", float("inf")
            )
        if candidate_validation["p99_violation"] != incumbent_validation["p99_violation"]:
            return candidate_validation["p99_violation"] < incumbent_validation["p99_violation"]
        return candidate_validation["max_violation"] < incumbent_validation["max_violation"]
    if candidate_validation["coverage"] != incumbent_validation["coverage"]:
        return candidate_validation["coverage"] > incumbent_validation["coverage"]
    if candidate_validation["mean_gap"] != incumbent_validation["mean_gap"]:
        return candidate_validation["mean_gap"] < incumbent_validation["mean_gap"]
    return candidate_validation["max_violation"] < incumbent_validation["max_violation"]


def grouped_diagnostics(
    df: pd.DataFrame,
    y: np.ndarray,
    z: np.ndarray,
    epsilon: float,
    tolerance: float,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    p = predictions_from_logits(z, df, config)
    p_side_bin_edges = [float(v) for v in config.get("diagnostics", {}).get("p_side_bin_edges", [])]
    p_side_bin: pd.Series | None = None
    if "p_side" in df.columns and len(p_side_bin_edges) >= 2:
        p_side_bin = p_side_bins(df["p_side"], p_side_bin_edges)
    work = pd.DataFrame(
        {
            "y": y,
            "z_raw": z,
            "p_upper_bound": p,
            "gap": p - y,
        "violation": np.maximum(y + epsilon - p, 0.0),
        "violating": p + tolerance < y + epsilon,
        }
    )
    for col in config.get("diagnostics", {}).get("group_columns", []):
        if col in df.columns:
            values = df[col].astype("string")
            if col == "p_bin" and p_side_bin is not None and "p_side" in df.columns:
                values = values.where(values.notna(), p_side_bin.where(df["p_side"].notna(), "missing"))
            work[col] = values.fillna("missing").astype(str).to_numpy()
    if p_side_bin is not None:
        work["p_side_bin"] = p_side_bin

    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for col in config.get("diagnostics", {}).get("group_columns", []):
        if col not in work.columns:
            continue
        rows: list[dict[str, Any]] = []
        for value, part in work.groupby(col, dropna=False):
            rows.append(group_metrics_row(str(value), part))
        diagnostics[f"by_{col}"] = rows
    if "p_side_bin" in work.columns:
        diagnostics["by_p_side_bin"] = [
            group_metrics_row(str(value), part) for value, part in work.groupby("p_side_bin", dropna=False)
        ]

    price_col = config.get("diagnostics", {}).get("price_bin_column", "target_raw")
    if price_col in df.columns:
        edges = [float(v) for v in config.get("diagnostics", {}).get("price_bin_edges", [])]
        if len(edges) >= 2:
            labels = [f"{edges[i]:.2f}_{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
            work["price_bin"] = pd.cut(
                pd.to_numeric(df[price_col], errors="coerce"),
                bins=edges,
                labels=labels,
                include_lowest=True,
                right=False,
            ).astype("string").fillna("missing")
            diagnostics["by_price_bin"] = [group_metrics_row(str(value), part) for value, part in work.groupby("price_bin", dropna=False)]
    return diagnostics


def p_side_bins(values: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{edges[i]:.2f}_{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype("string").fillna("missing")


def group_metrics_row(value: str, part: pd.DataFrame) -> dict[str, Any]:
    return {
        "value": value,
        "sample_count": int(len(part)),
        "coverage": float((~part["violating"]).mean()) if len(part) else float("nan"),
        "violation_rate": float(part["violating"].mean()) if len(part) else float("nan"),
        "mean_gap": float(part["gap"].mean()) if len(part) else float("nan"),
        "median_gap": float(part["gap"].median()) if len(part) else float("nan"),
        "gap_p05": float(part["gap"].quantile(0.05)) if len(part) else float("nan"),
        "gap_p95": float(part["gap"].quantile(0.95)) if len(part) else float("nan"),
        "gap_p95_p05_range": (
            float(part["gap"].quantile(0.95) - part["gap"].quantile(0.05)) if len(part) else float("nan")
        ),
        "min_gap": float(part["gap"].min()) if len(part) else float("nan"),
        "max_violation": float(part["violation"].max()) if len(part) else float("nan"),
        "p90_violation": float(part["violation"].quantile(0.90)) if len(part) else float("nan"),
        "p95_violation": float(part["violation"].quantile(0.95)) if len(part) else float("nan"),
        "p99_violation": float(part["violation"].quantile(0.99)) if len(part) else float("nan"),
    }


def write_predictions(
    df: pd.DataFrame,
    y: np.ndarray,
    z: np.ndarray,
    alpha: np.ndarray | None,
    path: Path,
    epsilon: float,
    config: dict[str, Any],
) -> None:
    pred = df[[c for c in PRED_BASE_COLS if c in df.columns]].copy()
    pred.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    p = predictions_from_logits(z, df, config)
    p_side_bin_edges = DEFAULT_P_SIDE_BIN_EDGES
    pred["y_raw"] = y
    pred["y"] = y
    pred["target"] = np.clip(y + epsilon, 1e-6, 1.0 - 1e-6)
    pred["z_raw"] = z
    pred["p_pred"] = p
    pred["p_upper_bound"] = p
    pred["gap"] = p - y
    pred["violation"] = np.maximum(y + epsilon - p, 0.0)
    if "p_side" in pred.columns:
        pred["p_bin"] = p_side_bins(pred["p_side"], p_side_bin_edges).where(pred["p_side"].notna(), "missing")
    if "target_raw" in pred.columns:
        price_edges = [i / 10 for i in range(11)]
        pred["price_bin"] = pd.cut(
            pd.to_numeric(pred["target_raw"], errors="coerce"),
            bins=price_edges,
            labels=[f"{price_edges[i]:.2f}_{price_edges[i + 1]:.2f}" for i in range(10)],
            include_lowest=True,
            right=False,
        ).astype("string").fillna("missing")
    pred["alpha"] = alpha if alpha is not None else np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)


def write_diagnostic_csvs(report_dir: Path, split_name: str, diagnostics: dict[str, list[dict[str, Any]]]) -> None:
    mapping = {
        "by_price_bin": f"diagnostics_{split_name}_by_price_bin.csv",
        "by_p_bin": f"diagnostics_{split_name}_by_p_bin.csv",
        "by_selected_side": f"diagnostics_{split_name}_by_selected_side.csv",
    }
    for key, filename in mapping.items():
        rows = diagnostics.get(key)
        if rows is not None:
            pd.DataFrame(rows).to_csv(report_dir / filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/upper_bound_mlp/configs/upper_bound_mlp_aug_lagrangian.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device.type == "auto":
        device = torch.device("cpu")

    train = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    valid = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    manifest = load_deploy_manifest(config)
    train, train_filter = apply_sample_filter(train, config, manifest)
    valid, validation_filter = apply_sample_filter(valid, config, manifest)
    columns, cat_cols = feature_set(config, train)
    preprocessor = Preprocessor.fit(train, columns, cat_cols)
    x_train = preprocessor.transform(train)
    x_valid = preprocessor.transform(valid)

    target_col = str(config["target"]["column"])
    epsilon = float(config["target"]["epsilon"])
    tolerance = float(config["target"]["feasibility_tolerance"])
    y_train = train[target_col].to_numpy(dtype=np.float32)
    y_valid = valid[target_col].to_numpy(dtype=np.float32)
    p_side_train = train["p_side"].to_numpy(dtype=np.float32)
    train_constraint_weight = constraint_weights_from_y(y_train, config)
    train_tightness_weight = tightness_weights_from_y(y_train, config)

    dataset = IndexedTensorDataset(x_train, y_train, p_side_train, train_constraint_weight, train_tightness_weight)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        generator=generator,
    )

    model = UpperBoundMLP(
        input_dim=x_train.shape[1],
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "epoch_metrics.csv"

    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_valid_metrics: dict[str, float] | None = None
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    grad_clip = float(config["training"].get("gradient_clip_norm", 5.0))
    min_validation_coverage = float(config.get("objective", {}).get("min_validation_coverage", 0.90))
    min_validation_mean_gap = float(config.get("objective", {}).get("min_validation_mean_gap", float("-inf")))
    max_validation_violation = config.get("objective", {}).get("max_validation_violation")
    max_validation_violation = float(max_validation_violation) if max_validation_violation is not None else None

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb, p_side_b, constraint_weight_b, tightness_weight_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            p_side_b = p_side_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            z = model(xb)
            loss = upper_bound_loss(z, yb, p_side_b, config)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_z = predict_logits(model, x_train, device, int(config["training"]["batch_size"]))
        valid_z = predict_logits(model, x_valid, device, int(config["training"]["batch_size"]))
        train_metrics = upper_bound_metrics(y_train, train_z, epsilon, tolerance, config, train)
        valid_metrics = upper_bound_metrics(y_valid, valid_z, epsilon, tolerance, config, valid)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"validation_{k}": v for k, v in valid_metrics.items()})
        epoch_rows.append(row)

        if is_better(
            valid_metrics,
            best_valid_metrics,
            min_validation_coverage,
            min_validation_mean_gap,
            max_validation_violation,
        ):
            best_epoch = epoch
            best_train_metrics = train_metrics
            best_valid_metrics = valid_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_mean_gap": train_metrics["mean_gap"],
                        "train_violation_rate": train_metrics["violation_rate"],
                        "validation_mean_gap": valid_metrics["mean_gap"],
                        "validation_violation_rate": valid_metrics["violation_rate"],
                        "validation_coverage": valid_metrics["coverage"],
                    },
                    sort_keys=True,
                )
            )

    if best_state is None or best_train_metrics is None or best_valid_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_z = predict_logits(model, x_train, device, int(config["training"]["batch_size"]))
    valid_z = predict_logits(model, x_valid, device, int(config["training"]["batch_size"]))
    best_train_metrics = upper_bound_metrics(y_train, train_z, epsilon, tolerance, config, train)
    best_valid_metrics = upper_bound_metrics(y_valid, valid_z, epsilon, tolerance, config, valid)
    train_alpha = None

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    checkpoint_path = models_dir / "upper_bound_mlp.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": x_train.shape[1],
            "model": config["model"],
            "preprocessor": preprocessor.to_dict(),
            "target": config["target"],
            "loss": config.get("loss", {}),
            "feature_columns": columns,
            "categorical_columns": cat_cols,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    train_pred_path = resolve_path(config["paths"]["predictions_train"])
    valid_pred_path = resolve_path(config["paths"]["predictions_validation"])
    write_predictions(train, y_train, train_z, train_alpha, train_pred_path, epsilon, config)
    write_predictions(valid, y_valid, valid_z, None, valid_pred_path, epsilon, config)

    train_diagnostics = grouped_diagnostics(train, y_train, train_z, epsilon, tolerance, config)
    validation_diagnostics = grouped_diagnostics(valid, y_valid, valid_z, epsilon, tolerance, config)
    write_diagnostic_csvs(reports_dir, "train", train_diagnostics)
    write_diagnostic_csvs(reports_dir, "validation", validation_diagnostics)
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "primary_metric": "validation mean_gap subject to validation coverage >= objective.min_validation_coverage",
        "objective": {
            "min_validation_coverage": min_validation_coverage,
            "min_validation_mean_gap": min_validation_mean_gap,
            "max_validation_violation": max_validation_violation,
            "optimize_metric": config.get("objective", {}).get("optimize_metric", "mean_gap"),
            "tie_breaker_metric": config.get("objective", {}).get("tie_breaker_metric", "max_violation"),
        },
        "loss": config.get("loss", {}),
        "prediction_output": config.get("prediction_output", {"type": "sigmoid"}),
        "sample_filter": {
            "train": train_filter,
            "validation": validation_filter,
        },
        "model_family": "upper_bound_mlp_asymmetric_logcosh",
        "replaces": "catboost_quantile_q70_q80_q90",
        "baseline_comparison": None,
        "best_epoch": best_epoch,
        "train_metrics": best_train_metrics,
        "validation_metrics": best_valid_metrics,
        "train_diagnostics": train_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "train_window": {
            "row_count": int(len(train)),
            "start": str(pd.to_datetime(train["timestamp"], utc=True).min()) if "timestamp" in train else None,
            "end": str(pd.to_datetime(train["timestamp"], utc=True).max()) if "timestamp" in train else None,
        },
        "validation_window": {
            "row_count": int(len(valid)),
            "start": str(pd.to_datetime(valid["timestamp"], utc=True).min()) if "timestamp" in valid else None,
            "end": str(pd.to_datetime(valid["timestamp"], utc=True).max()) if "timestamp" in valid else None,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "epoch_metrics": str(metrics_path),
            "train_predictions": str(train_pred_path),
            "validation_predictions": str(valid_pred_path),
            "diagnostics_train_by_price_bin": str(reports_dir / "diagnostics_train_by_price_bin.csv"),
            "diagnostics_train_by_p_bin": str(reports_dir / "diagnostics_train_by_p_bin.csv"),
            "diagnostics_train_by_selected_side": str(reports_dir / "diagnostics_train_by_selected_side.csv"),
            "diagnostics_validation_by_price_bin": str(reports_dir / "diagnostics_validation_by_price_bin.csv"),
            "diagnostics_validation_by_p_bin": str(reports_dir / "diagnostics_validation_by_p_bin.csv"),
            "diagnostics_validation_by_selected_side": str(reports_dir / "diagnostics_validation_by_selected_side.csv"),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_train.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "deploy_artifact_dir": config["paths"]["deploy_artifact_dir"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "success_criteria": {
            "validation_coverage_at_least_target": best_valid_metrics["coverage"] >= min_validation_coverage,
            "validation_max_violation_at_most_target": (
                best_valid_metrics["max_violation"] <= max_validation_violation if max_validation_violation is not None else None
            ),
            "validation_mean_gap": best_valid_metrics["mean_gap"],
            "validation_max_violation": best_valid_metrics["max_violation"],
        },
    }
    report_path = reports_dir / "summary_metrics.json"
    write_json(report_path, report)
    print(json.dumps({"train_metrics": best_train_metrics, "validation_metrics": best_valid_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
