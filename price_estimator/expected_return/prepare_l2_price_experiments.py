from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", type=Path,
        default=ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/config.yaml",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "price_estimator/expected_return/experiments/20260702_l2_common_window_data",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "price_estimator/expected_return/experiments",
    )
    args = parser.parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    sample = pd.read_parquet(
        args.data_root / "A2_classifier_l2/expected_return_train.parquet"
    )
    selected_l2 = sorted(column for column in sample if column.startswith("pm_l2_selected_"))
    variants = {
        "20260702_l2_P0_A0_common_window": ("A0_common_window", False),
        "20260702_l2_A2_direction_l2": ("A2_classifier_l2", False),
        "20260702_l2_A3_price_l2": ("A0_common_window", True),
        "20260702_l2_A4_full": ("A2_classifier_l2", True),
        "20260702_l2_A4_market_direction_full": ("A2_market_l2", True),
    }
    for experiment_id, (data_variant, use_l2) in variants.items():
        config = deepcopy(base)
        experiment_dir = args.output_root / experiment_id
        data_dir = args.data_root / data_variant
        config["experiment_id"] = experiment_id
        config["paths"].update(
            {
                "experiment_dir": str(experiment_dir.relative_to(ROOT)),
                "target_summary_source": str((data_dir / "target_build_summary.json").relative_to(ROOT)),
                "train_dataset": str((data_dir / "expected_return_train.parquet").relative_to(ROOT)),
                "validation_dataset": str((data_dir / "expected_return_validation.parquet").relative_to(ROOT)),
                "models_dir": str((experiment_dir / "models").relative_to(ROOT)),
                "reports_dir": str((experiment_dir / "reports").relative_to(ROOT)),
                "predictions_train": str((experiment_dir / "reports/predictions_train.parquet").relative_to(ROOT)),
                "predictions_calibration": str((experiment_dir / "reports/predictions_calibration.parquet").relative_to(ROOT)),
                "predictions_validation": str((experiment_dir / "reports/predictions_validation.parquet").relative_to(ROOT)),
            }
        )
        base_added = [
            "selected_side", "p_up", "p_side", "direction_confidence",
            "p_bin", "p_side_bucket", "market_time_bucket",
        ]
        config["features"]["added_columns"] = base_added + (selected_l2 if use_l2 else [])
        config["target"]["future_low_target"] = "polymarket_l2_future_four_minute_v1"
        config["target"]["min_ev_grid"] = [
            0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
            0.15, 0.20, 0.30, 0.40, 0.50,
        ]
        config["probability_lineage"] = {
            "source": "calibrated_p_up",
            "raw_p_up_diagnostic_only": True,
            "direction_variant": data_variant,
        }
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    ablations = {
        "20260702_l2_A5_no_book_depth": ("depth", "bid_size", "ask_size", "top_imbalance"),
        "20260702_l2_A6_no_order_flow": ("ofi", "update", "add_", "cancel", "order_flow"),
        "20260702_l2_A7_no_trade_dynamics": ("trade", "last"),
        "20260702_l2_A8_no_cross_side": ("minus_opposite", "complement", "pair_", "difference"),
    }
    full_config = yaml.safe_load(
        (args.output_root / "20260702_l2_A4_full/config.yaml").read_text(encoding="utf-8")
    )
    base_added = full_config["features"]["added_columns"][:7]
    for experiment_id, blocked_tokens in ablations.items():
        config = deepcopy(full_config)
        experiment_dir = args.output_root / experiment_id
        config["experiment_id"] = experiment_id
        config["features"]["added_columns"] = base_added + [
            column for column in selected_l2
            if not any(token in column.lower() for token in blocked_tokens)
        ]
        for key in ("experiment_dir", "models_dir", "reports_dir", "predictions_train", "predictions_calibration", "predictions_validation"):
            old = str(config["paths"][key])
            config["paths"][key] = old.replace("20260702_l2_A4_full", experiment_id)
        config["ablation"] = {"blocked_tokens": list(blocked_tokens)}
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    policy_base_path = (
        ROOT / "price_estimator/expected_return/experiments/"
        "20260625_expected_return_xgb_lowbid_isotonic_no_leak/config.yaml"
    )
    policy = yaml.safe_load(policy_base_path.read_text(encoding="utf-8"))
    policy_id = "20260702_l2_A4_xgb_q_hazard"
    policy_dir = args.output_root / policy_id
    policy["experiment_id"] = policy_id
    policy["baseline"] = {
        "experiment_id": "20260702_l2_A4_full",
        "source": "price_estimator/expected_return/experiments/20260702_l2_A4_full/reports/summary_metrics.json",
        "row_filter": "validation_metrics",
        "validation_sum_pnl": -89.71,
    }
    policy["paths"].update(
        {
            "train_dataset": "price_estimator/expected_return/experiments/20260702_l2_common_window_data/A2_classifier_l2/expected_return_train.parquet",
            "validation_dataset": "price_estimator/expected_return/experiments/20260702_l2_common_window_data/A2_classifier_l2/expected_return_validation.parquet",
            "h2_checkpoint": "price_estimator/expected_return/experiments/20260702_l2_A4_full/models/hazard_survival_cdf.pt",
            "reports_dir": f"price_estimator/expected_return/experiments/{policy_id}/reports",
        }
    )
    policy["features"]["forbidden_columns"] = sorted(
        set(policy["features"]["forbidden_columns"] + ["raw_p_up", "target_prediction"])
    )
    policy["policy_search"].update(
        {
            "bid_min": 0.01,
            "bid_max": 0.85,
            "bid_step": 0.01,
            "min_ev_grid": [value / 100.0 for value in range(-30, 31)],
            "min_q_grid": [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            "bid_offset_steps_grid": [-5, -3, -2, -1, 0, 1],
            "min_order_count": 20,
        }
    )
    policy["metadata"] = {
        "deploy_training_mode": "no_leak_l2_common_window",
        "offline_validation_metric_source": "artifacts/data_v2/experiments/20260702_l2_common_window_direction/A2_classifier_l2/report.json",
        "probability_source": "calibrated_p_up",
        "raw_p_up_diagnostic_only": True,
    }
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "config.yaml").write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    noiso = deepcopy(policy)
    noiso_id = "20260702_l2_A4_xgb_q_hazard_noiso"
    noiso["experiment_id"] = noiso_id
    noiso["model"]["isotonic_q"] = False
    noiso["paths"]["reports_dir"] = (
        f"price_estimator/expected_return/experiments/{noiso_id}/reports"
    )
    noiso_dir = args.output_root / noiso_id
    noiso_dir.mkdir(parents=True, exist_ok=True)
    (noiso_dir / "config.yaml").write_text(
        yaml.safe_dump(noiso, sort_keys=False), encoding="utf-8"
    )
    for isotonic in (False, True):
        market = deepcopy(policy)
        suffix = "iso" if isotonic else "raw"
        market_id = f"20260702_l2_A4_market_mid_q_{suffix}"
        market["experiment_id"] = market_id
        market["model"] = {
            "family": "market_mid",
            "probability_column": "pm_l2_selected_mid",
            "isotonic_q": isotonic,
        }
        market["paths"]["reports_dir"] = (
            f"price_estimator/expected_return/experiments/{market_id}/reports"
        )
        market_dir = args.output_root / market_id
        market_dir.mkdir(parents=True, exist_ok=True)
        (market_dir / "config.yaml").write_text(
            yaml.safe_dump(market, sort_keys=False), encoding="utf-8"
        )
    market_direction = deepcopy(policy)
    market_direction_id = "20260702_l2_market_direction_xgb_q_hazard"
    market_direction["experiment_id"] = market_direction_id
    market_direction["paths"].update(
        {
            "train_dataset": "price_estimator/expected_return/experiments/20260702_l2_common_window_data/A2_market_l2/expected_return_train.parquet",
            "validation_dataset": "price_estimator/expected_return/experiments/20260702_l2_common_window_data/A2_market_l2/expected_return_validation.parquet",
            "h2_checkpoint": "price_estimator/expected_return/experiments/20260702_l2_A4_market_direction_full/models/hazard_survival_cdf.pt",
            "reports_dir": f"price_estimator/expected_return/experiments/{market_direction_id}/reports",
        }
    )
    market_direction["metadata"]["offline_validation_metric_source"] = (
        "artifacts/data_v2/experiments/20260702_l2_common_window_direction/A2_market_l2/report.json"
    )
    market_direction_dir = args.output_root / market_direction_id
    market_direction_dir.mkdir(parents=True, exist_ok=True)
    (market_direction_dir / "config.yaml").write_text(
        yaml.safe_dump(market_direction, sort_keys=False), encoding="utf-8"
    )
    print({"selected_l2_feature_count": len(selected_l2), "experiments": list(variants)})


if __name__ == "__main__":
    main()
