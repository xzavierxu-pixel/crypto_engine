from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.run_auc_optimization_experiments import _write_auc_leaderboard
from scripts.experiments.run_validation_optimization_experiments import _read_yaml, _run_variant
from src.data.dataset_builder import infer_feature_columns


def _token(value: float) -> str:
    return str(value).replace(".", "p")


def _variants() -> list[dict]:
    variants: list[dict] = []
    for top_n in [100, 200, 400, 600, 800, 1200, 1500]:
        for c_value in [0.05, 0.1, 0.25, 0.5, 1.0]:
            for weighted in [False, True]:
                variants.append(
                    {
                        "name": f"logistic_top{top_n}_c{_token(c_value)}_w{int(weighted)}",
                        "category": "auc_logistic",
                        "overrides": {
                            "model.active_plugin": "logistic",
                            "model.plugins.logistic.C": c_value,
                            "model.plugins.logistic.max_iter": 2000,
                            "model.plugins.logistic.solver": "lbfgs",
                            "sample_weighting.enabled": weighted,
                        },
                        "drop_packs": [],
                        "top_n_features": top_n,
                    }
                )

    for top_n in [200, 400, 600, 800, 1200]:
        for depth in [4, 6, 8]:
            for learning_rate in [0.02, 0.03, 0.05]:
                variants.append(
                    {
                        "name": f"catboost_top{top_n}_d{depth}_lr{_token(learning_rate)}",
                        "category": "auc_catboost",
                        "overrides": {
                            "model.active_plugin": "catboost",
                            "model.plugins.catboost.iterations": 500,
                            "model.plugins.catboost.learning_rate": learning_rate,
                            "model.plugins.catboost.depth": depth,
                            "model.plugins.catboost.l2_leaf_reg": 8.0,
                            "model.plugins.catboost.loss_function": "Logloss",
                            "model.plugins.catboost.eval_metric": "AUC",
                            "model.plugins.catboost.random_seed": 42,
                            "model.plugins.catboost.verbose": False,
                            "sample_weighting.enabled": True,
                        },
                        "drop_packs": [],
                        "top_n_features": top_n,
                    }
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation AUC experiments across existing model plugins.")
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

    variants = _variants()
    if args.max_variants:
        variants = variants[: args.max_variants]

    records: list[dict] = []
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
        _write_auc_leaderboard(args.output_root, records, experiment_id=args.experiment_id)
    _write_auc_leaderboard(args.output_root, records, experiment_id=args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
