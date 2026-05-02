from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_builder import infer_feature_columns

from scripts.run_validation_optimization_experiments import (
    _read_yaml,
    _run_variant,
    _write_leaderboard,
)


def _model_overrides(
    *,
    leaves: int,
    depth: int,
    min_child: int,
    reg_alpha: float,
    reg_lambda: float,
    learning_rate: float,
    estimators: int,
    scale_pos_weight: float,
    subsample: float = 0.9,
    colsample: float = 0.8,
) -> dict[str, Any]:
    return {
        "model.plugins.lightgbm.num_leaves": leaves,
        "model.plugins.lightgbm.max_depth": depth,
        "model.plugins.lightgbm.min_child_samples": min_child,
        "model.plugins.lightgbm.reg_alpha": reg_alpha,
        "model.plugins.lightgbm.reg_lambda": reg_lambda,
        "model.plugins.lightgbm.learning_rate": learning_rate,
        "model.plugins.lightgbm.n_estimators": estimators,
        "model.plugins.lightgbm.scale_pos_weight": scale_pos_weight,
        "model.plugins.lightgbm.subsample": subsample,
        "model.plugins.lightgbm.colsample_bytree": colsample,
    }


def _weight_overrides(name: str) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {
        "current": {
            "sample_weighting.enabled": True,
            "sample_weighting.min_abs_return": 0.0001,
            "sample_weighting.full_weight_abs_return": 0.0003,
            "sample_weighting.min_weight": 0.35,
        },
        "disabled": {
            "sample_weighting.enabled": False,
            "sample_weighting.min_abs_return": 0.0001,
            "sample_weighting.full_weight_abs_return": 0.0003,
            "sample_weighting.min_weight": 0.35,
        },
        "conservative": {
            "sample_weighting.enabled": True,
            "sample_weighting.min_abs_return": 0.00015,
            "sample_weighting.full_weight_abs_return": 0.0005,
            "sample_weighting.min_weight": 0.25,
        },
        "aggressive": {
            "sample_weighting.enabled": True,
            "sample_weighting.min_abs_return": 0.00005,
            "sample_weighting.full_weight_abs_return": 0.0002,
            "sample_weighting.min_weight": 0.50,
        },
        "high_floor": {
            "sample_weighting.enabled": True,
            "sample_weighting.min_abs_return": 0.0001,
            "sample_weighting.full_weight_abs_return": 0.0003,
            "sample_weighting.min_weight": 0.60,
        },
    }
    return presets[name]


def _variants() -> list[dict[str, Any]]:
    base_shape = dict(
        leaves=20,
        depth=6,
        min_child=600,
        reg_alpha=5.0,
        reg_lambda=20.0,
        learning_rate=0.02,
        estimators=700,
    )
    model_shapes = {
        "base": base_shape,
        "extra_reg": {
            **base_shape,
            "min_child": 800,
            "reg_alpha": 8.0,
            "reg_lambda": 35.0,
        },
        "lower_lr": {
            **base_shape,
            "learning_rate": 0.015,
            "estimators": 1000,
        },
        "less_colsample": {
            **base_shape,
            "colsample": 0.65,
        },
        "more_colsample": {
            **base_shape,
            "colsample": 0.95,
        },
    }
    top_ns = [900, 1000, 1100, 1150, 1200, 1250, 1300, 1400, 1500]
    scales = [1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.70]

    variants: list[dict[str, Any]] = []
    for top_n in top_ns:
        for scale in scales:
            variants.append(
                {
                    "name": f"feature_top{top_n}_spw{str(scale).replace('.', 'p')}",
                    "category": "feature_model",
                    "overrides": _model_overrides(scale_pos_weight=scale, **base_shape),
                    "drop_packs": [],
                    "top_n_features": top_n,
                }
            )

    for shape_name, shape in model_shapes.items():
        if shape_name == "base":
            continue
        for top_n in [1100, 1200, 1300]:
            for scale in [1.45, 1.50, 1.55]:
                variants.append(
                    {
                        "name": f"model_{shape_name}_top{top_n}_spw{str(scale).replace('.', 'p')}",
                        "category": "model",
                        "overrides": _model_overrides(scale_pos_weight=scale, **shape),
                        "drop_packs": [],
                        "top_n_features": top_n,
                    }
                )

    for weight_name in ["disabled", "conservative", "aggressive", "high_floor"]:
        for top_n in [1100, 1200, 1300]:
            for scale in [1.45, 1.50, 1.55]:
                overrides = _model_overrides(scale_pos_weight=scale, **base_shape)
                overrides.update(_weight_overrides(weight_name))
                variants.append(
                    {
                        "name": f"sample_{weight_name}_top{top_n}_spw{str(scale).replace('.', 'p')}",
                        "category": "sample_weighting",
                        "overrides": overrides,
                        "drop_packs": [],
                        "top_n_features": top_n,
                    }
                )

    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run feature/sample/model experiments without threshold-only changes.")
    parser.add_argument("--training-frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--feature-importance", type=Path, required=True)
    parser.add_argument("--dev-days", type=int, default=60)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--purge-rows", type=int, default=1)
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()

    base_config = _read_yaml(args.config)
    training_frame = pd.read_parquet(args.training_frame)
    base_feature_columns = infer_feature_columns(training_frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.config_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.config_root / "base.yaml")

    variants = _variants()
    if args.max_variants:
        variants = variants[: args.max_variants]

    records: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] {variant['name']}", flush=True)
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

    best = max(
        (record for record in records if record["constraints_satisfied"]),
        key=lambda record: record["validation_balanced_precision"],
        default=None,
    )
    (args.output_root / "target_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "target_balanced_precision": 0.65,
                "target_reached": bool(best and best["validation_balanced_precision"] >= 0.65),
                "best": best,
                "record_count": len(records),
                "threshold_note": "Threshold search settings are inherited from the base config.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
