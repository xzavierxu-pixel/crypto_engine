from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import TrainingFrame, compute_sample_weight, infer_feature_columns
from src.model.evaluation import compute_selective_binary_metrics
from src.model.train import train_binary_selective_model_from_split

from scripts.run_balanced_precision_holdout_experiment import (
    _git_info,
    _metric_dict,
    _predict_proba,
    _search_thresholds,
    _split_by_time,
    _window_dict,
)


BASELINE = {
    "balanced_precision": 0.5627267617149574,
    "precision_up": 0.6060606060606061,
    "precision_down": 0.5193929173693086,
    "signal_coverage": 0.6108836985311823,
}

DROP_PACK_PATTERNS = {
    "derivatives_options": [
        "option",
        "bvol",
        "mark_iv",
        "delta_",
        "gamma_",
        "vega_",
        "theta_",
    ],
    "derivatives_book_ticker": [
        "book_ticker",
        "best_bid",
        "best_ask",
        "bid_qty",
        "ask_qty",
    ],
    "derivatives_oi": [
        "oi_",
        "oi_level",
        "oi_notional",
    ],
    "second_level_book_microstructure": [
        "sl_spread",
        "sl_bid",
        "sl_ask",
        "sl_microprice",
        "sl_depth",
        "sl_orderbook",
    ],
    "second_level_depth": [
        "sl_depth",
        "sl_level",
        "depth_",
    ],
    "second_level_interaction_bank": [
        "sl_interaction__",
    ],
    "regime_interactions": [
        "_x_regime",
        "regime_x_",
        "_x_vol_regime",
    ],
    "side_specific_transforms": [
        "upside_",
        "downside_",
        "positive_basis_pressure",
        "negative_basis_pressure",
    ],
}


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _remove_pack(payload: dict[str, Any], pack: str) -> None:
    core_packs = payload["features"]["profiles"]["core_5m"]["packs"]
    payload["features"]["profiles"]["core_5m"]["packs"] = [item for item in core_packs if item != pack]
    second_level_profile = payload.get("second_level", {}).get("feature_profile")
    if second_level_profile:
        sl_profiles = payload.get("second_level", {}).get("profiles", {})
        if second_level_profile in sl_profiles:
            packs = sl_profiles[second_level_profile].get("packs", [])
            sl_profiles[second_level_profile]["packs"] = [item for item in packs if item != pack]


def _variant_matrix() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"name": "baseline_replay", "category": "baseline", "overrides": {}, "drop_packs": []},
    ]

    for t_up_max in [0.60, 0.65, 0.70]:
        for t_down_min in [0.40, 0.35, 0.30]:
            variants.append(
                {
                    "name": f"threshold_up{t_up_max:.2f}_down{t_down_min:.2f}".replace(".", "p"),
                    "category": "threshold",
                    "overrides": {
                        "threshold_search.t_up_max": t_up_max,
                        "threshold_search.t_down_min": t_down_min,
                        "threshold_search.step": 0.005,
                    },
                    "drop_packs": [],
                }
            )

    lgbm_variants = [
        ("lgbm_small_regularized", 12, 4, 600, 2.0, 10.0, 0.03, 400),
        ("lgbm_tiny_strong_regularized", 12, 4, 600, 5.0, 20.0, 0.03, 400),
        ("lgbm_mid_regularized", 20, 6, 600, 2.0, 10.0, 0.03, 400),
        ("lgbm_mid_more_l2", 20, 6, 300, 2.0, 20.0, 0.03, 400),
        ("lgbm_deeper_regularized", 31, 8, 600, 2.0, 10.0, 0.03, 400),
        ("lgbm_low_lr_small", 12, 4, 300, 2.0, 10.0, 0.02, 700),
        ("lgbm_low_lr_mid", 20, 6, 300, 2.0, 10.0, 0.02, 700),
        ("lgbm_low_lr_high_l2", 20, 6, 600, 5.0, 20.0, 0.02, 700),
        ("lgbm_less_regularized", 31, 8, 150, 0.8, 5.0, 0.03, 400),
        ("lgbm_shallow_min150", 12, 4, 150, 0.8, 5.0, 0.03, 400),
    ]
    for name, leaves, depth, min_child, reg_alpha, reg_lambda, lr, estimators in lgbm_variants:
        variants.append(
            {
                "name": name,
                "category": "lgbm",
                "overrides": {
                    "model.plugins.lightgbm.num_leaves": leaves,
                    "model.plugins.lightgbm.max_depth": depth,
                    "model.plugins.lightgbm.min_child_samples": min_child,
                    "model.plugins.lightgbm.reg_alpha": reg_alpha,
                    "model.plugins.lightgbm.reg_lambda": reg_lambda,
                    "model.plugins.lightgbm.learning_rate": lr,
                    "model.plugins.lightgbm.n_estimators": estimators,
                },
                "drop_packs": [],
            }
        )

    weight_variants = [
        ("weights_current", True, 0.0001, 0.0003, 0.35),
        ("weights_conservative", True, 0.00015, 0.0005, 0.25),
        ("weights_aggressive", True, 0.00005, 0.0002, 0.50),
        ("weights_disabled", False, 0.0001, 0.0003, 0.35),
    ]
    for name, enabled, min_abs, full_abs, min_weight in weight_variants:
        variants.append(
            {
                "name": name,
                "category": "weights",
                "overrides": {
                    "sample_weighting.enabled": enabled,
                    "sample_weighting.min_abs_return": min_abs,
                    "sample_weighting.full_weight_abs_return": full_abs,
                    "sample_weighting.min_weight": min_weight,
                },
                "drop_packs": [],
            }
        )

    for pack in DROP_PACK_PATTERNS:
        variants.append(
            {
                "name": f"drop_{pack}",
                "category": "ablation",
                "overrides": {},
                "drop_packs": [pack],
            }
        )

    for top_n in [50, 100, 150, 200, 300, 400, 450, 500, 550, 600, 700, 800, 1200]:
        variants.append(
            {
                "name": f"feature_select_top_{top_n}",
                "category": "feature_selection",
                "overrides": {},
                "drop_packs": [],
                "top_n_features": top_n,
            }
        )
    for top_n in [100, 200, 300, 400, 500, 700]:
        variants.append(
            {
                "name": f"feature_corr_top_{top_n}",
                "category": "feature_correlation",
                "overrides": {},
                "drop_packs": [],
                "corr_top_n_features": top_n,
            }
        )
    return variants


def _combo_variants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for row in sorted(records, key=lambda item: item["validation_balanced_precision"], reverse=True):
        if not row["constraints_satisfied"] or row["name"] == "baseline_replay":
            continue
        winners.setdefault(row["category"], row)

    combos: list[dict[str, Any]] = []
    best_threshold = winners.get("threshold")
    best_lgbm = winners.get("lgbm")
    best_weights = winners.get("weights")
    best_ablation = winners.get("ablation")
    best_feature = winners.get("feature_selection") or winners.get("feature_correlation")

    candidates = [
        ("combo_lgbm_threshold", [best_lgbm, best_threshold]),
        ("combo_lgbm_weights", [best_lgbm, best_weights]),
        ("combo_lgbm_ablation", [best_lgbm, best_ablation]),
        ("combo_lgbm_threshold_weights", [best_lgbm, best_threshold, best_weights]),
        ("combo_best_three", [best_lgbm, best_threshold, best_ablation]),
        ("combo_feature_lgbm", [best_feature, best_lgbm]),
        ("combo_feature_weights", [best_feature, best_weights]),
        ("combo_feature_ablation", [best_feature, best_ablation]),
        ("combo_feature_lgbm_weights", [best_feature, best_lgbm, best_weights]),
        ("combo_feature_lgbm_ablation", [best_feature, best_lgbm, best_ablation]),
    ]
    for name, parts in candidates:
        selected_parts = [part for part in parts if part is not None]
        if len(selected_parts) < 2:
            continue
        overrides: dict[str, Any] = {}
        drop_packs: list[str] = []
        parents: list[str] = []
        combo_top_n: int | None = None
        for part in selected_parts:
            overrides.update(part["variant"]["overrides"])
            drop_packs.extend(part["variant"]["drop_packs"])
            parents.append(part["name"])
            if "top_n_features" in part["variant"]:
                combo_top_n = int(part["variant"]["top_n_features"])
        combos.append(
            {
                "name": name,
                "category": "combo",
                "overrides": overrides,
                "drop_packs": sorted(set(drop_packs)),
                "parents": parents,
                **({"top_n_features": combo_top_n} if combo_top_n else {}),
            }
        )
    return combos


def _apply_variant_config(base_payload: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    for key, value in variant.get("overrides", {}).items():
        _set_nested(payload, key, value)
    for pack in variant.get("drop_packs", []):
        _remove_pack(payload, pack)
    payload.setdefault("experiment", {})
    payload["experiment"]["variant"] = variant
    return payload


def _drop_feature_columns(
    feature_columns: list[str],
    drop_packs: list[str],
    *,
    top_features: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if top_features is not None:
        allowed = set(top_features)
        kept = [column for column in feature_columns if column in allowed]
        dropped = [column for column in feature_columns if column not in allowed]
        return kept, dropped

    patterns = [pattern for pack in drop_packs for pattern in DROP_PACK_PATTERNS.get(pack, [])]
    if not patterns:
        return feature_columns, []
    kept: list[str] = []
    dropped: list[str] = []
    for column in feature_columns:
        lowered = column.lower()
        if any(pattern.lower() in lowered for pattern in patterns):
            dropped.append(column)
        else:
            kept.append(column)
    return kept, dropped


def _load_top_features(feature_importance_path: Path, top_n: int) -> list[str]:
    importance = pd.read_csv(feature_importance_path)
    if "feature" not in importance.columns:
        raise ValueError(f"Feature importance missing feature column: {feature_importance_path}")
    sort_columns = [column for column in ["gain", "split"] if column in importance.columns]
    if sort_columns:
        importance = importance.sort_values(sort_columns, ascending=False)
    return importance["feature"].head(top_n).astype(str).to_list()


def _select_correlation_top_features(
    *,
    frame: TrainingFrame,
    top_n: int,
    dev_days: int,
    validation_days: int,
    purge_rows: int,
) -> list[str]:
    development, _, _ = _split_by_time(
        frame,
        dev_days=dev_days,
        validation_days=validation_days,
        purge_rows=purge_rows,
    )
    X = development.X
    y = development.y.astype("float64")
    scores: list[tuple[str, float]] = []
    y_std = float(y.std(ddof=0))
    if y_std == 0:
        return frame.feature_columns[:top_n]
    for column in frame.feature_columns:
        values = pd.to_numeric(X[column], errors="coerce")
        std = float(values.std(ddof=0))
        if std == 0 or values.isna().all():
            continue
        corr = values.corr(y)
        if pd.notna(corr):
            scores.append((column, abs(float(corr))))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [column for column, _ in scores[:top_n]]


def _constraints_satisfied(metrics: dict[str, Any], settings: Any) -> bool:
    return (
        metrics["up_signal_count"] >= settings.threshold_search.min_up_signals
        and metrics["down_signal_count"] >= settings.threshold_search.min_down_signals
        and metrics["total_signal_count"] >= settings.threshold_search.min_total_signals
        and metrics["signal_coverage"] >= settings.objective.min_coverage
    )


def _run_variant(
    *,
    base_config: dict[str, Any],
    training_frame: pd.DataFrame,
    base_feature_columns: list[str],
    variant: dict[str, Any],
    output_root: Path,
    config_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant_name = variant["name"]
    output_dir = output_root / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"{variant_name}.yaml"
    payload = _apply_variant_config(base_config, variant)
    _write_yaml(config_path, payload)
    settings = load_settings(config_path)

    top_features = None
    if "top_n_features" in variant:
        top_features = _load_top_features(args.feature_importance, int(variant["top_n_features"]))
    feature_columns, dropped_columns = _drop_feature_columns(
        base_feature_columns,
        variant.get("drop_packs", []),
        top_features=top_features,
    )
    sample_weight_column = "stage1_sample_weight" if "stage1_sample_weight" in training_frame.columns else None
    run_frame = training_frame
    if sample_weight_column and "abs_return" in training_frame.columns:
        run_frame = training_frame.copy()
        run_frame[sample_weight_column] = compute_sample_weight(run_frame["abs_return"], settings=settings)
    frame = TrainingFrame(
        frame=run_frame,
        feature_columns=feature_columns,
        target_column="target",
        sample_weight_column=sample_weight_column,
    )
    if "corr_top_n_features" in variant:
        top_features = _select_correlation_top_features(
            frame=frame,
            top_n=int(variant["corr_top_n_features"]),
            dev_days=args.dev_days,
            validation_days=args.validation_days,
            purge_rows=args.purge_rows,
        )
        feature_columns, dropped_columns = _drop_feature_columns(base_feature_columns, [], top_features=top_features)
        frame = TrainingFrame(
            frame=run_frame,
            feature_columns=feature_columns,
            target_column="target",
            sample_weight_column=sample_weight_column,
        )
    development, validation, split_info = _split_by_time(
        frame,
        dev_days=args.dev_days,
        validation_days=args.validation_days,
        purge_rows=args.purge_rows,
    )

    started = time.perf_counter()
    artifacts = train_binary_selective_model_from_split(
        development=development,
        validation=validation,
        settings=settings,
        weighted=sample_weight_column is not None and settings.sample_weighting.enabled,
    )
    validation_proba = _predict_proba(artifacts, validation)
    selected, frontier = _search_thresholds(settings, validation.y, validation_proba)
    t_up = float(selected["t_up"])
    t_down = float(selected["t_down"])
    metrics = {
        "development": _metric_dict(
            development.y,
            _predict_proba(artifacts, development),
            t_up=t_up,
            t_down=t_down,
            settings=settings,
        ),
        "validation": _metric_dict(
            validation.y,
            validation_proba,
            t_up=t_up,
            t_down=t_down,
            settings=settings,
        ),
    }
    duration_seconds = time.perf_counter() - started

    frontier_path = output_dir / "threshold_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    feature_set_path = output_dir / "feature_set.json"
    feature_set_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    dropped_feature_path = output_dir / "dropped_features.json"
    dropped_feature_path.write_text(json.dumps(dropped_columns, indent=2), encoding="utf-8")
    if not artifacts.feature_importance.empty:
        artifacts.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    with (output_dir / "model_artifact.pkl").open("wb") as handle:
        pickle.dump(artifacts, handle)

    validation_metrics = metrics["validation"]
    constraints_satisfied = _constraints_satisfied(validation_metrics, settings)
    report_path = output_dir / "report.json"
    report = {
        "experiment_id": args.experiment_id,
        "variant": variant,
        "baseline": BASELINE,
        "training_frame": str(args.training_frame),
        "config_copy": str(config_path),
        "feature_count": len(feature_columns),
        "dropped_feature_count": len(dropped_columns),
        "top_n_features": variant.get("top_n_features"),
        "model_settings": asdict(settings.model),
        "sample_weighting_settings": asdict(settings.sample_weighting),
        "objective_settings": asdict(settings.objective),
        "threshold_search_settings": asdict(settings.threshold_search),
        "split_info": split_info,
        "thresholds": {
            "t_up": t_up,
            "t_down": t_down,
            "selection_reason": selected["selection_reason"],
            "validation_selected_row": selected,
        },
        "metrics": metrics,
        "train_metrics": metrics["development"],
        "train_window": _window_dict(split_info, "development"),
        "validation_metrics": metrics["validation"],
        "validation_window": _window_dict(split_info, "validation"),
        "validation_delta_vs_baseline": {
            "balanced_precision": validation_metrics["balanced_precision"] - BASELINE["balanced_precision"],
            "precision_up": validation_metrics["precision_up"] - BASELINE["precision_up"],
            "precision_down": validation_metrics["precision_down"] - BASELINE["precision_down"],
            "signal_coverage": validation_metrics["signal_coverage"] - BASELINE["signal_coverage"],
        },
        "constraints_satisfied": constraints_satisfied,
        "duration_seconds": duration_seconds,
        "git": _git_info(),
        "artifacts": {
            "report": str(report_path),
            "config": str(config_path),
            "threshold_frontier": str(frontier_path),
            "feature_set": str(feature_set_path),
            "dropped_features": str(dropped_feature_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_variant_summary(output_dir / "summary.md", report)
    return {
        "name": variant_name,
        "category": variant["category"],
        "variant": variant,
        "report_path": str(report_path),
        "config_path": str(config_path),
        "validation_balanced_precision": validation_metrics["balanced_precision"],
        "validation_precision_up": validation_metrics["precision_up"],
        "validation_precision_down": validation_metrics["precision_down"],
        "validation_signal_coverage": validation_metrics["signal_coverage"],
        "validation_up_signal_count": validation_metrics["up_signal_count"],
        "validation_down_signal_count": validation_metrics["down_signal_count"],
        "validation_total_signal_count": validation_metrics["total_signal_count"],
        "validation_overall_signal_accuracy": validation_metrics["overall_signal_accuracy"],
        "validation_roc_auc": validation_metrics.get("roc_auc"),
        "validation_brier_score": validation_metrics.get("brier_score"),
        "validation_log_loss": validation_metrics.get("log_loss"),
        "train_roc_auc": metrics["development"].get("roc_auc"),
        "train_brier_score": metrics["development"].get("brier_score"),
        "train_log_loss": metrics["development"].get("log_loss"),
        "constraints_satisfied": constraints_satisfied,
        "delta_balanced_precision": validation_metrics["balanced_precision"] - BASELINE["balanced_precision"],
        "feature_count": len(feature_columns),
        "dropped_feature_count": len(dropped_columns),
        "top_n_features": variant.get("top_n_features"),
        "t_up": t_up,
        "t_down": t_down,
        "duration_seconds": duration_seconds,
    }


def _write_variant_summary(path: Path, report: dict[str, Any]) -> None:
    validation = report["metrics"]["validation"]
    lines = [
        f"# Validation Optimization Variant: {report['variant']['name']}",
        "",
        f"- category: `{report['variant']['category']}`",
        f"- balanced_precision: `{validation['balanced_precision']:.6f}`",
        f"- delta_vs_baseline: `{report['validation_delta_vs_baseline']['balanced_precision']:.6f}`",
        f"- precision_up: `{validation['precision_up']:.6f}`",
        f"- precision_down: `{validation['precision_down']:.6f}`",
        f"- signal_coverage: `{validation['signal_coverage']:.6f}`",
        f"- constraints_satisfied: `{report['constraints_satisfied']}`",
        f"- thresholds: `t_up={report['thresholds']['t_up']:.4f}`, `t_down={report['thresholds']['t_down']:.4f}`",
        f"- config_path: `{report['config_copy']}`",
        f"- report_path: `{report['artifacts']['report']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leaderboard(output_root: Path, records: list[dict[str, Any]], *, experiment_id: str) -> None:
    df = pd.DataFrame.from_records(records)
    ranked = df.sort_values(
        ["constraints_satisfied", "validation_balanced_precision", "validation_signal_coverage"],
        ascending=[False, False, False],
    )
    leaderboard_path = output_root / "leaderboard.csv"
    ranked.to_csv(leaderboard_path, index=False)
    best = ranked.iloc[0].to_dict() if not ranked.empty else None
    summary = {
        "experiment_id": experiment_id,
        "baseline": BASELINE,
        "target_balanced_precision": 0.60,
        "best": best,
        "target_reached": bool(best and best["constraints_satisfied"] and best["validation_balanced_precision"] >= 0.60),
        "leaderboard_path": str(leaderboard_path),
        "record_count": int(len(records)),
        "git": _git_info(),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Validation Optimization Leaderboard",
        "",
        f"- experiment_id: `{experiment_id}`",
        f"- target_balanced_precision: `0.600000`",
        f"- baseline_balanced_precision: `{BASELINE['balanced_precision']:.6f}`",
    ]
    if best:
        lines.extend(
            [
                f"- best_variant: `{best['name']}`",
                f"- best_balanced_precision: `{best['validation_balanced_precision']:.6f}`",
                f"- best_delta: `{best['delta_balanced_precision']:.6f}`",
                f"- best_coverage: `{best['validation_signal_coverage']:.6f}`",
                f"- target_reached: `{summary['target_reached']}`",
            ]
        )
    lines.extend(["", "## Top 10", ""])
    lines.append(
        "| rank | variant | category | balanced_precision | precision_up | precision_down | coverage | UP | DOWN | total | constraints |"
    )
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for rank, row in enumerate(ranked.head(10).to_dict(orient="records"), start=1):
        lines.append(
            "| {rank} | `{name}` | `{category}` | {validation_balanced_precision:.6f} | "
            "{validation_precision_up:.6f} | {validation_precision_down:.6f} | "
            "{validation_signal_coverage:.6f} | {validation_up_signal_count} | "
            "{validation_down_signal_count} | {validation_total_signal_count} | {constraints_satisfied} |".format(
                rank=rank, **row
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_commit(output_root: Path, config_root: Path, script_path: Path, *, experiment_id: str) -> str | None:
    paths = [
        str(output_root),
        str(config_root),
        str(script_path),
    ]
    subprocess.run(["git", "add", *paths], cwd=REPO_ROOT, check=True)
    commit_message = f"experiment: validation balanced precision optimization {experiment_id}"
    result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation balanced_precision optimization experiments.")
    parser.add_argument("--training-frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="20260502_validation_bp_optimization")
    parser.add_argument("--dev-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=15)
    parser.add_argument("--purge-rows", type=int, default=1)
    parser.add_argument(
        "--feature-importance",
        type=Path,
        default=Path("artifacts/data_v2/experiments/20260502_validation_bp_optimization/baseline_replay/feature_importance.csv"),
    )
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--skip-combos", action="store_true")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    base_config = _read_yaml(args.config)
    training_frame = pd.read_parquet(args.training_frame)
    base_feature_columns = infer_feature_columns(training_frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.config_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.config_root / "base.yaml")

    records: list[dict[str, Any]] = []
    variants = _variant_matrix()
    if args.max_variants > 0:
        variants = variants[: args.max_variants]
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] running {variant['name']}", flush=True)
        records.append(
            _run_variant(
                base_config=base_config,
                training_frame=training_frame,
                base_feature_columns=base_feature_columns,
                variant=variant,
                output_root=args.output_root,
                config_root=args.config_root,
                args=args,
            )
        )
        _write_leaderboard(args.output_root, records, experiment_id=args.experiment_id)

    if not args.skip_combos and args.max_variants == 0:
        combos = _combo_variants(records)
        for index, variant in enumerate(combos, start=1):
            print(f"[combo {index}/{len(combos)}] running {variant['name']}", flush=True)
            records.append(
                _run_variant(
                    base_config=base_config,
                    training_frame=training_frame,
                    base_feature_columns=base_feature_columns,
                    variant=variant,
                    output_root=args.output_root,
                    config_root=args.config_root,
                    args=args,
                )
            )
            _write_leaderboard(args.output_root, records, experiment_id=args.experiment_id)

    _write_leaderboard(args.output_root, records, experiment_id=args.experiment_id)
    commit_hash = None
    if args.commit:
        commit_hash = _maybe_commit(args.output_root, args.config_root, Path(__file__), experiment_id=args.experiment_id)
        summary_path = args.output_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["experiment_commit"] = commit_hash
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_root / "summary.json"), "commit": commit_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
