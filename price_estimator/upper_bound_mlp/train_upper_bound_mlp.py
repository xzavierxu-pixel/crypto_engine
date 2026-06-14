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
    def __init__(self, x: np.ndarray, y: np.ndarray, target_z: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        self.target_z = torch.from_numpy(target_z.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.target_z[idx], torch.tensor(idx, dtype=torch.long)


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


def upper_bound_metrics(y: np.ndarray, z: np.ndarray, epsilon: float, tolerance: float) -> dict[str, float]:
    p = sigmoid_np(z)
    gap = p - y
    violation = y + epsilon - p
    violating = p + tolerance < y + epsilon
    return {
        "sample_count": float(len(y)),
        "mean_gap": float(np.mean(gap)) if len(y) else float("nan"),
        "min_gap": float(np.min(gap)) if len(y) else float("nan"),
        "violation_rate": float(np.mean(violating)) if len(y) else float("nan"),
        "max_violation": float(np.max(violation)) if len(y) else float("nan"),
        "coverage": float(np.mean(~violating)) if len(y) else float("nan"),
    }


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
) -> bool:
    if incumbent_validation is None:
        return True
    candidate_ok = candidate_validation["coverage"] >= min_validation_coverage
    incumbent_ok = incumbent_validation["coverage"] >= min_validation_coverage
    if candidate_ok != incumbent_ok:
        return candidate_ok
    if candidate_ok and incumbent_ok:
        if candidate_validation["mean_gap"] != incumbent_validation["mean_gap"]:
            return candidate_validation["mean_gap"] < incumbent_validation["mean_gap"]
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
    p = sigmoid_np(z)
    work = pd.DataFrame(
        {
            "y": y,
            "z_raw": z,
            "p_upper_bound": p,
            "gap": p - y,
            "violation": y + epsilon - p,
            "violating": p + tolerance < y + epsilon,
        }
    )
    for col in config.get("diagnostics", {}).get("group_columns", []):
        if col in df.columns:
            work[col] = df[col].astype("string").fillna("missing").astype(str).to_numpy()

    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for col in config.get("diagnostics", {}).get("group_columns", []):
        if col not in work.columns:
            continue
        rows: list[dict[str, Any]] = []
        for value, part in work.groupby(col, dropna=False):
            rows.append(group_metrics_row(str(value), part))
        diagnostics[f"by_{col}"] = rows

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


def group_metrics_row(value: str, part: pd.DataFrame) -> dict[str, Any]:
    return {
        "value": value,
        "sample_count": int(len(part)),
        "coverage": float((~part["violating"]).mean()) if len(part) else float("nan"),
        "violation_rate": float(part["violating"].mean()) if len(part) else float("nan"),
        "mean_gap": float(part["gap"].mean()) if len(part) else float("nan"),
        "min_gap": float(part["gap"].min()) if len(part) else float("nan"),
        "max_violation": float(part["violation"].max()) if len(part) else float("nan"),
    }


def write_predictions(
    df: pd.DataFrame,
    y: np.ndarray,
    z: np.ndarray,
    alpha: np.ndarray | None,
    path: Path,
    epsilon: float,
) -> None:
    pred = df[[c for c in PRED_BASE_COLS if c in df.columns]].copy()
    pred.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    p = sigmoid_np(z)
    pred["y"] = y
    pred["z_raw"] = z
    pred["p_upper_bound"] = p
    pred["gap"] = p - y
    pred["violation"] = y + epsilon - p
    pred["alpha"] = alpha if alpha is not None else np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)


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
    columns, cat_cols = feature_set(config, train)
    preprocessor = Preprocessor.fit(train, columns, cat_cols)
    x_train = preprocessor.transform(train)
    x_valid = preprocessor.transform(valid)

    target_col = str(config["target"]["column"])
    epsilon = float(config["target"]["epsilon"])
    clip_min = float(config["target"]["clip_min"])
    clip_max = float(config["target"]["clip_max"])
    tolerance = float(config["target"]["feasibility_tolerance"])
    y_train = train[target_col].to_numpy(dtype=np.float32)
    y_valid = valid[target_col].to_numpy(dtype=np.float32)
    target_z_train = logit_np(np.clip(y_train + epsilon, clip_min, clip_max)).astype(np.float32)

    dataset = IndexedTensorDataset(x_train, y_train, target_z_train)
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
    alpha = torch.zeros(len(train), dtype=torch.float32, device=device)
    alpha_max = float(config["training"]["alpha_max"])
    rho_values = [float(v) for v in config["training"]["rho_schedule"]]
    rho_index = 0
    rho = rho_values[rho_index] if rho_values else float(config["training"]["rho_initial"])
    rho_patience = int(config["training"]["rho_patience_epochs"])

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "epoch_metrics.csv"

    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_valid_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_since_violation_improved = 0
    best_violation = float("inf")
    epoch_rows: list[dict[str, Any]] = []
    grad_clip = float(config["training"]["gradient_clip_norm"])
    min_validation_coverage = float(config.get("objective", {}).get("min_validation_coverage", 0.90))

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb, target_zb, idxb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            target_zb = target_zb.to(device)
            idxb = idxb.to(device)
            alpha_b = alpha[idxb].view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            z = model(xb)
            p = torch.sigmoid(z)
            g = target_zb - z
            shifted = g + alpha_b / rho
            aug_penalty = 0.5 * rho * (torch.relu(shifted).pow(2) - (alpha_b / rho).pow(2))
            loss = (p - yb).mean() + aug_penalty.mean()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                z_new = model(xb)
                g_new = target_zb - z_new
                alpha[idxb] = torch.clamp(alpha[idxb] + rho * g_new.view(-1), min=0.0, max=alpha_max)
            batch_losses.append(float(loss.detach().cpu()))

        train_z = predict_logits(model, x_train, device, int(config["training"]["batch_size"]))
        valid_z = predict_logits(model, x_valid, device, int(config["training"]["batch_size"]))
        train_metrics = upper_bound_metrics(y_train, train_z, epsilon, tolerance)
        valid_metrics = upper_bound_metrics(y_valid, valid_z, epsilon, tolerance)
        row: dict[str, Any] = {
            "epoch": epoch,
            "rho": rho,
            "loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"validation_{k}": v for k, v in valid_metrics.items()})
        epoch_rows.append(row)

        if train_metrics["violation_rate"] < best_violation:
            best_violation = train_metrics["violation_rate"]
            epochs_since_violation_improved = 0
        else:
            epochs_since_violation_improved += 1
        if epochs_since_violation_improved >= rho_patience and rho_index + 1 < len(rho_values):
            rho_index += 1
            rho = rho_values[rho_index]
            epochs_since_violation_improved = 0

        if is_better(valid_metrics, best_valid_metrics, min_validation_coverage):
            best_epoch = epoch
            best_train_metrics = train_metrics
            best_valid_metrics = valid_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "rho": rho,
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
    best_train_metrics = upper_bound_metrics(y_train, train_z, epsilon, tolerance)
    best_valid_metrics = upper_bound_metrics(y_valid, valid_z, epsilon, tolerance)
    train_alpha = alpha.detach().cpu().numpy()

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
            "feature_columns": columns,
            "categorical_columns": cat_cols,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    train_pred_path = resolve_path(config["paths"]["predictions_train"])
    valid_pred_path = resolve_path(config["paths"]["predictions_validation"])
    write_predictions(train, y_train, train_z, train_alpha, train_pred_path, epsilon)
    write_predictions(valid, y_valid, valid_z, None, valid_pred_path, epsilon)

    top_alpha_count = int(config["training"]["save_top_alpha_count"])
    top_idx = np.argsort(-train_alpha)[:top_alpha_count]
    top_alpha = pd.DataFrame(
        {
            "sample_id": top_idx,
            "alpha": train_alpha[top_idx],
            "y": y_train[top_idx],
            "p_upper_bound": sigmoid_np(train_z[top_idx]),
            "gap": sigmoid_np(train_z[top_idx]) - y_train[top_idx],
        }
    )
    top_alpha_path = reports_dir / "top_alpha_samples.csv"
    top_alpha.to_csv(top_alpha_path, index=False)

    manifest = load_deploy_manifest(config)
    train_diagnostics = grouped_diagnostics(train, y_train, train_z, epsilon, tolerance, config)
    validation_diagnostics = grouped_diagnostics(valid, y_valid, valid_z, epsilon, tolerance, config)
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "primary_metric": "validation mean_gap subject to validation coverage >= objective.min_validation_coverage",
        "objective": {
            "min_validation_coverage": min_validation_coverage,
            "optimize_metric": config.get("objective", {}).get("optimize_metric", "mean_gap"),
            "tie_breaker_metric": config.get("objective", {}).get("tie_breaker_metric", "max_violation"),
        },
        "model_family": "upper_bound_mlp_augmented_lagrangian",
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
            "top_alpha_samples": str(top_alpha_path),
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
            "validation_mean_gap": best_valid_metrics["mean_gap"],
            "validation_max_violation": best_valid_metrics["max_violation"],
        },
    }
    report_path = reports_dir / "upper_bound_mlp_metrics.json"
    write_json(report_path, report)
    print(json.dumps({"train_metrics": best_train_metrics, "validation_metrics": best_valid_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
