from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.run_bp_feature_sample_model_experiments import _variants as _feature_sample_model_variants
from scripts.experiments.run_bp_targeted_experiments import _variants as _targeted_variants
from scripts.experiments.run_validation_optimization_experiments import (
    _read_yaml,
    _run_variant,
    _variant_matrix,
)
from src.data.dataset_builder import infer_feature_columns


def _variant_key(variant: dict[str, Any]) -> str:
    return json.dumps(
        {
            "overrides": variant.get("overrides", {}),
            "drop_packs": variant.get("drop_packs", []),
            "top_n_features": variant.get("top_n_features"),
            "corr_top_n_features": variant.get("corr_top_n_features"),
        },
        sort_keys=True,
    )


def _variants() -> list[dict[str, Any]]:
    candidates = [
        *_variant_matrix(),
        *_targeted_variants(),
        *_feature_sample_model_variants(),
    ]
    deduped: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(candidates):
        key = _variant_key(variant)
        renamed = dict(variant)
        renamed["name"] = f"auc_{index:03d}_{variant['name']}"
        renamed["category"] = f"auc_{variant['category']}"
        deduped.setdefault(key, renamed)
    return list(deduped.values())


def _write_auc_leaderboard(output_root: Path, records: list[dict[str, Any]], *, experiment_id: str) -> None:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return
    ranked = df.sort_values(
        ["validation_roc_auc", "validation_brier_score", "validation_log_loss"],
        ascending=[False, True, True],
        na_position="last",
    )
    leaderboard_path = output_root / "auc_leaderboard.csv"
    ranked.to_csv(leaderboard_path, index=False)
    best = ranked.iloc[0].to_dict()
    summary = {
        "experiment_id": experiment_id,
        "target_validation_roc_auc": 0.65,
        "target_reached": bool(pd.notna(best.get("validation_roc_auc")) and best["validation_roc_auc"] >= 0.65),
        "best": best,
        "leaderboard_path": str(leaderboard_path),
        "record_count": int(len(records)),
        "objective_note": "Ranked by validation roc_auc only; threshold metrics are diagnostics.",
    }
    (output_root / "auc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# AUC Optimization Leaderboard",
        "",
        f"- experiment_id: `{experiment_id}`",
        "- target_validation_roc_auc: `0.650000`",
        f"- best_variant: `{best['name']}`",
        f"- best_validation_roc_auc: `{best['validation_roc_auc']:.6f}`",
        f"- target_reached: `{summary['target_reached']}`",
        "",
        "## Top 10",
        "",
        "| rank | variant | category | validation_auc | train_auc | brier | log_loss | balanced_precision | coverage | UP | DOWN |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked.head(10).to_dict(orient="records"), start=1):
        lines.append(
            "| {rank} | `{name}` | `{category}` | {validation_roc_auc:.6f} | "
            "{train_roc_auc:.6f} | {validation_brier_score:.6f} | {validation_log_loss:.6f} | "
            "{validation_balanced_precision:.6f} | {validation_signal_coverage:.6f} | "
            "{validation_up_signal_count} | {validation_down_signal_count} |".format(rank=rank, **row)
        )
    (output_root / "auc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation AUC optimization experiments.")
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
        _write_auc_leaderboard(args.output_root, records, experiment_id=args.experiment_id)
    _write_auc_leaderboard(args.output_root, records, experiment_id=args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
