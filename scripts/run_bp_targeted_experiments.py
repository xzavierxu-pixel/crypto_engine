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


def _lgbm_overrides(
    *,
    leaves: int,
    depth: int,
    min_child: int,
    reg_alpha: float,
    reg_lambda: float,
    learning_rate: float,
    estimators: int,
    scale_pos_weight: float,
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
    }


def _variants() -> list[dict[str, Any]]:
    model_shapes = [
        ("tiny_strong", dict(leaves=12, depth=4, min_child=600, reg_alpha=5.0, reg_lambda=20.0, learning_rate=0.03, estimators=400)),
        ("shallow_min150", dict(leaves=12, depth=4, min_child=150, reg_alpha=0.8, reg_lambda=5.0, learning_rate=0.03, estimators=400)),
        ("low_lr_high_l2", dict(leaves=20, depth=6, min_child=600, reg_alpha=5.0, reg_lambda=20.0, learning_rate=0.02, estimators=700)),
        ("tiny_extra_l2", dict(leaves=8, depth=3, min_child=700, reg_alpha=8.0, reg_lambda=40.0, learning_rate=0.025, estimators=600)),
    ]
    feature_sets: list[tuple[str, dict[str, int]]] = [
        ("all", {}),
        ("top800", {"top_n_features": 800}),
        ("top1200", {"top_n_features": 1200}),
        ("top600", {"top_n_features": 600}),
    ]
    scales = [0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    variants: list[dict[str, Any]] = []
    for shape_name, shape in model_shapes:
        for feature_name, feature_extra in feature_sets:
            for scale in scales:
                variants.append(
                    {
                        "name": f"{shape_name}_{feature_name}_spw{str(scale).replace('.', 'p')}",
                        "category": "targeted_scale_pos_weight",
                        "overrides": _lgbm_overrides(scale_pos_weight=scale, **shape),
                        "drop_packs": [],
                        **feature_extra,
                    }
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run targeted BP>=0.65 validation experiments.")
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
    target_summary = {
        "experiment_id": args.experiment_id,
        "target_balanced_precision": 0.65,
        "target_reached": bool(best and best["validation_balanced_precision"] >= 0.65),
        "best": best,
        "record_count": len(records),
    }
    (args.output_root / "target_summary.json").write_text(
        json.dumps(target_summary, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
