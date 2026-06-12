from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (  # noqa: E402
    OBJECTIVE_METRIC_FIELDS,
    apply_continuation_expert_decision,
    apply_reversal_only_decision,
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260612_catboost_continuation_reversal_stacked_meta_model.yaml")
EXPERIMENT_ID = "20260612_catboost_continuation_reversal_stacked_meta_model"
BASELINE_SELECTION_SCORE = 0.5748509217
BASELINE_UTILITY = 0.2674076058
BASELINE_ACCEPTED_SAMPLE_ACCURACY = 0.6909542934
BASELINE_ACCEPTED_COUNT = 5229.0
BASELINE_COVERAGE = 0.7001874665
LEAKAGE_TOKENS = (
    "target",
    "future_close",
    "abs_return",
    "signed_return",
    "stage1_target",
    "stage2_target",
    "original_btc_direction_target",
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _load_inputs(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = _read_json(experiment_dir / "artifact_manifest.json")
    return (
        pd.read_parquet(experiment_dir / "development_frame.parquet"),
        pd.read_parquet(experiment_dir / "validation_frame.parquet"),
        pd.read_parquet(experiment_dir / "train_predictions.parquet"),
        pd.read_parquet(experiment_dir / "validation_predictions.parquet"),
        list(manifest["feature_columns"]),
    )


def _is_safe_feature(name: str) -> bool:
    lower = name.lower()
    return not any(token in lower for token in LEAKAGE_TOKENS)


def _select_features(all_features: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    selected = [name for name in all_features if _is_safe_feature(name) and any(pattern.search(name) for pattern in compiled)]
    if not selected:
        raise ValueError("feature set selected no features")
    return selected


def _select_top_market_features(all_features: list[str], config: dict[str, Any]) -> list[str]:
    market = config["market_feature_set"]
    importance = pd.read_csv(Path(market["importance_path"]))
    if "feature" not in importance.columns:
        raise ValueError(f"feature importance file has no feature column: {market['importance_path']}")
    allowed = set(_select_features(all_features, list(market["allowlist_patterns"])))
    top_n = int(market["top_n"])
    selected: list[str] = []
    for feature in importance["feature"].astype(str):
        if feature in allowed and feature not in selected:
            selected.append(feature)
        if len(selected) >= top_n:
            break
    if not selected:
        raise ValueError("top market feature selection produced no features")
    return selected


def _regimes(predictions: pd.DataFrame) -> pd.Series:
    missing = {"timestamp", "first_minute_side"}.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing regime columns: {sorted(missing)}")
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    session = pd.cut(timestamps.dt.hour, bins=[-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    fm_side = predictions["first_minute_side"].astype("object")
    if not fm_side.isin(["YES", "NO"]).all():
        bad = sorted(str(value) for value in fm_side.loc[~fm_side.isin(["YES", "NO"])].dropna().unique())
        raise ValueError(f"first_minute_side has unsupported values: {bad}")
    values = [
        "d" + str(day) + "_" + sess + "_fm_" + side.lower()
        for day, sess, side in zip(timestamps.dt.dayofweek, session, fm_side)
    ]
    return pd.Series(values, index=predictions.index)


def _is_reversal(predictions: pd.DataFrame) -> pd.Series:
    resolved_side = pd.Series(np.where(predictions["target"].astype(int) == 1, "YES", "NO"), index=predictions.index)
    fm_side = predictions["first_minute_side"].astype("object")
    return fm_side.isin(["YES", "NO"]) & (fm_side != resolved_side)


def _train_expert_pair(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    train_predictions: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series]:
    continuation_model = CatBoostClassifier(**config["expert_model"])
    continuation_model.fit(train_frame[features], train_frame["target"].astype(int))
    cont = pd.Series(continuation_model.predict_proba(target_frame[features])[:, 1], index=target_frame.index).clip(0.0, 1.0)

    sample_weight = pd.Series(1.0, index=train_frame.index)
    sample_weight.loc[_is_reversal(train_predictions)] = float(config["reversal_expert"]["sample_weight"])
    reversal_model = CatBoostClassifier(**config["expert_model"])
    reversal_model.fit(train_frame[features], train_frame["target"].astype(int), sample_weight=sample_weight)
    rev = pd.Series(reversal_model.predict_proba(target_frame[features])[:, 1], index=target_frame.index).clip(0.0, 1.0)
    return cont, rev


def _chronological_oof_expert_predictions(
    config: dict[str, Any],
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    n = len(frame)
    folds = int(config["oof"]["chronological_folds"])
    min_train_end = max(1, int(n * float(config["oof"]["min_train_fraction_first_fold"])))
    starts = np.linspace(min_train_end, n, folds + 1, dtype=int)
    cont = pd.Series(np.nan, index=frame.index, dtype="float64")
    rev = pd.Series(np.nan, index=frame.index, dtype="float64")
    fold_records: list[dict[str, int]] = []
    for fold_id, (start, end) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        if end <= start:
            continue
        train_idx = frame.index[:start]
        predict_idx = frame.index[start:end]
        fold_cont, fold_rev = _train_expert_pair(
            config,
            frame.loc[train_idx],
            frame.loc[predict_idx],
            predictions.loc[train_idx],
            features,
        )
        cont.loc[predict_idx] = fold_cont
        rev.loc[predict_idx] = fold_rev
        fold_records.append(
            {
                "fold": fold_id,
                "train_rows": int(len(train_idx)),
                "predict_rows": int(len(predict_idx)),
                "train_start_pos": 0,
                "train_end_pos": int(start - 1),
                "predict_start_pos": int(start),
                "predict_end_pos": int(end - 1),
            }
        )
    valid = cont.notna() & rev.notna()
    if not bool(valid.any()):
        raise ValueError("OOF expert prediction generation produced no stacker training rows")
    return cont.loc[valid], rev.loc[valid], {
        "method": "chronological_oof_with_expanding_expert_training",
        "folds_requested": folds,
        "folds_used": len(fold_records),
        "oof_train_rows": int(valid.sum()),
        "dropped_initial_rows": int((~valid).sum()),
        "folds": fold_records,
    }


def _expert_decisions(
    config: dict[str, Any],
    predictions: pd.DataFrame,
    cont_p: pd.Series,
    rev_p: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    thresholds = _read_json(Path(config["continuation_expert"]["thresholds_path"]))
    cont = apply_continuation_expert_decision(cont_p, predictions["first_minute_side"], _regimes(predictions), thresholds)
    rev = apply_reversal_only_decision(
        rev_p,
        predictions["first_minute_side"],
        t_up=float(config["reversal_expert"]["t_up"]),
        t_down=float(config["reversal_expert"]["t_down"]),
    )
    return cont, rev


def _base_stacker_features(
    predictions: pd.DataFrame,
    cont_p: pd.Series,
    rev_p: pd.Series,
    cont_decision: pd.Series,
    rev_decision: pd.Series,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    fm_side = predictions["first_minute_side"].astype("object")
    cont_conf = (cont_p.astype("float64").clip(0.0, 1.0) - 0.5).abs()
    rev_conf = (rev_p.astype("float64").clip(0.0, 1.0) - 0.5).abs()
    cont_accept = cont_decision != "ABSTAIN"
    rev_accept = rev_decision != "ABSTAIN"
    cont_up = cont_decision == "UP"
    cont_down = cont_decision == "DOWN"
    rev_up = rev_decision == "UP"
    rev_down = rev_decision == "DOWN"
    session_id = pd.cut(timestamps.dt.hour, bins=[-1, 7, 15, 23], labels=[0, 1, 2]).astype(int)
    session_name = pd.cut(timestamps.dt.hour, bins=[-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    expert_direction_known = cont_accept & rev_accept
    experts_agree = expert_direction_known & (cont_decision == rev_decision)
    output = pd.DataFrame(index=predictions.index)
    output["continuation_p_up"] = cont_p.astype("float64")
    output["reversal_p_up"] = rev_p.astype("float64")
    output["continuation_accept"] = cont_accept.astype(int)
    output["reversal_accept"] = rev_accept.astype(int)
    output["continuation_confidence"] = cont_conf
    output["reversal_confidence"] = rev_conf
    output["continuation_minus_reversal_p_up"] = output["continuation_p_up"] - output["reversal_p_up"]
    output["continuation_confidence_minus_reversal_confidence"] = cont_conf - rev_conf
    output["max_expert_confidence"] = pd.concat([cont_conf, rev_conf], axis=1).max(axis=1)
    output["min_expert_confidence"] = pd.concat([cont_conf, rev_conf], axis=1).min(axis=1)
    output["both_accept"] = (cont_accept & rev_accept).astype(int)
    output["neither_accept"] = (~cont_accept & ~rev_accept).astype(int)
    output["only_continuation_accept"] = (cont_accept & ~rev_accept).astype(int)
    output["only_reversal_accept"] = (rev_accept & ~cont_accept).astype(int)
    output["experts_agree_direction"] = experts_agree.astype(int)
    output["experts_disagree_direction"] = (expert_direction_known & ~experts_agree).astype(int)
    output["first_minute_side"] = (fm_side == "YES").astype(int)
    output["first_minute_side_yes"] = (fm_side == "YES").astype(int)
    output["first_minute_side_no"] = (fm_side == "NO").astype(int)
    output["continuation_same_side_indicator"] = cont_accept.astype(int)
    output["reversal_opposite_side_indicator"] = rev_accept.astype(int)
    output["cont_decision_is_up"] = cont_up.astype(int)
    output["cont_decision_is_down"] = cont_down.astype(int)
    output["rev_decision_is_up"] = rev_up.astype(int)
    output["rev_decision_is_down"] = rev_down.astype(int)
    output["dayofweek"] = timestamps.dt.dayofweek.astype(int)
    output["hour"] = timestamps.dt.hour.astype(int)
    output["session_id"] = session_id
    output["session_open"] = ((timestamps.dt.hour == 0) | (timestamps.dt.hour == 8) | (timestamps.dt.hour == 16)).astype(int)
    output["session_asia"] = (session_name == "asia").astype(int)
    output["session_europe"] = (session_name == "europe").astype(int)
    output["session_us"] = (session_name == "us").astype(int)
    return output


def _stacker_features(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    cont_p: pd.Series,
    rev_p: pd.Series,
    cont_decision: pd.Series,
    rev_decision: pd.Series,
    market_features: list[str] | None,
) -> pd.DataFrame:
    base = _base_stacker_features(predictions, cont_p, rev_p, cont_decision, rev_decision)
    if market_features:
        market = frame.loc[base.index, market_features].apply(pd.to_numeric, errors="coerce")
        base = pd.concat([base, market], axis=1)
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _threshold_values(search: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    step = float(search["step"])
    up = np.round(np.arange(float(search["t_up_min"]), float(search["t_up_max"]) + step / 2.0, step), 6)
    down = np.round(np.arange(float(search["t_down_min"]), float(search["t_down_max"]) + step / 2.0, step), 6)
    return up, down


def _decisions(p_up: pd.Series, *, t_up: float, t_down: float) -> pd.Series:
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= float(t_up)] = "UP"
    decisions.loc[p_up <= float(t_down)] = "DOWN"
    return decisions


def _search(predictions: pd.DataFrame, p_up: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    min_coverage = float(config["objective"]["min_coverage"])
    for t_up in _threshold_values(config["threshold_search"])[0]:
        for t_down in _threshold_values(config["threshold_search"])[1]:
            if float(t_down) >= float(t_up):
                continue
            decisions = _decisions(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                predictions["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
            row = {"t_up": float(t_up), "t_down": float(t_down), **metrics}
            row["coverage_constraint_satisfied"] = bool(metrics["coverage"] >= min_coverage)
            records.append(row)
            if (
                metrics["coverage"] >= min_coverage
                and metrics["utility"] > 0.0
                and metrics["accepted_sample_accuracy"] > 0.50
            ):
                eligible.append(row)
    if not records:
        raise ValueError("threshold search produced no candidates")
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
            -abs(row["t_up"] - 0.5) - abs(row["t_down"] - 0.5),
        ),
    )
    best["constraint_satisfied"] = bool(eligible)
    best["fallback_reason"] = None if eligible else "no candidate satisfied coverage/utility/accuracy constraints"
    best["objective"] = "selection_score"
    best["hard_constraint"] = "coverage_only"
    decisions = _decisions(p_up, t_up=float(best["t_up"]), t_down=float(best["t_down"]))
    return pd.DataFrame.from_records(records), best, decisions


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _audit(
    predictions: pd.DataFrame,
    p_up: pd.Series,
    decisions: pd.Series,
    cont_p: pd.Series,
    rev_p: pd.Series,
    cont_decision: pd.Series,
    rev_decision: pd.Series,
) -> pd.DataFrame:
    y = predictions["target"].astype(int)
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    accepted = decisions != "ABSTAIN"
    correct = ((decisions == "UP") == (y == 1)) & accepted
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "target": y,
            "first_minute_side": predictions["first_minute_side"].astype("object"),
            "stacked_p_up": p_up.astype("float64"),
            "final_decision": decisions.astype("object"),
            "final_accept": accepted,
            "final_correct": correct,
            "continuation_p_up": cont_p.astype("float64"),
            "continuation_decision": cont_decision.astype("object"),
            "continuation_accept": cont_decision.astype("object") != "ABSTAIN",
            "reversal_p_up": rev_p.astype("float64"),
            "reversal_decision": rev_decision.astype("object"),
            "reversal_accept": rev_decision.astype("object") != "ABSTAIN",
        }
    )


def _diagnostic_slices(audit: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    audit = audit.copy()
    audit["session"] = pd.cut(pd.to_datetime(audit["timestamp"], utc=True).dt.hour, [-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    audit["expert_agreement_bucket"] = np.where(
        audit["continuation_decision"] == audit["reversal_decision"],
        "agree",
        np.where((audit["continuation_accept"]) & (audit["reversal_accept"]), "disagree", "one_or_neither_accept"),
    )
    resolved_side = pd.Series(np.where(audit["target"].astype(int) == 1, "YES", "NO"), index=audit.index)
    audit["actual_regime"] = np.where(audit["first_minute_side"] == resolved_side, "continuation", "reversal")
    for column in ["actual_regime", "first_minute_side", "session", "expert_agreement_bucket"]:
        records = []
        for value, group in audit.groupby(column, dropna=False):
            accepted = group["final_accept"].astype(bool)
            records.append(
                {
                    column: str(value),
                    "sample_count": float(len(group)),
                    "accepted_count": float(accepted.sum()),
                    "coverage": float(accepted.mean()) if len(group) else 0.0,
                    "accepted_accuracy": float(group.loc[accepted, "final_correct"].mean()) if accepted.any() else 0.0,
                }
            )
        output[column] = records
    return output


def _run_variant(
    config: dict[str, Any],
    variant: str,
    model_params: dict[str, Any],
    market_features: list[str] | None,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    oof_cont: pd.Series,
    oof_rev: pd.Series,
    validation_cont: pd.Series,
    validation_rev: pd.Series,
) -> dict[str, Any]:
    stacker_train_predictions = train_predictions.loc[oof_cont.index]
    cont_train_decision, rev_train_decision = _expert_decisions(config, stacker_train_predictions, oof_cont, oof_rev)
    cont_val_decision, rev_val_decision = _expert_decisions(config, validation_predictions, validation_cont, validation_rev)
    train_x = _stacker_features(
        train_frame.loc[oof_cont.index],
        stacker_train_predictions,
        oof_cont,
        oof_rev,
        cont_train_decision,
        rev_train_decision,
        market_features,
    )
    validation_x = _stacker_features(
        validation_frame,
        validation_predictions,
        validation_cont,
        validation_rev,
        cont_val_decision,
        rev_val_decision,
        market_features,
    )
    model = CatBoostClassifier(**model_params)
    model.fit(train_x, stacker_train_predictions["target"].astype(int))
    train_p = pd.Series(model.predict_proba(train_x)[:, 1], index=train_x.index).clip(0.0, 1.0)
    validation_p = pd.Series(model.predict_proba(validation_x)[:, 1], index=validation_x.index).clip(0.0, 1.0)
    frontier, validation_best, validation_decisions = _search(validation_predictions, validation_p, config)
    train_decisions = _decisions(
        train_p,
        t_up=float(validation_best["selected_t_up"]),
        t_down=float(validation_best["selected_t_down"]),
    )
    train_metrics = compute_decision_metrics(
        stacker_train_predictions["target"],
        train_p,
        train_decisions,
        selected_t_up=float(validation_best["selected_t_up"]),
        selected_t_down=float(validation_best["selected_t_down"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(stacker_train_predictions, train_decisions))
    validation_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_p,
        validation_decisions,
        selected_t_up=float(validation_best["selected_t_up"]),
        selected_t_down=float(validation_best["selected_t_down"]),
    )
    validation_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_decisions))
    validation_metrics.update(
        {
            "constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
            "objective": "selection_score",
            "hard_constraint": "coverage_only",
        }
    )
    importance = pd.DataFrame(
        {
            "feature": train_x.columns,
            "importance": model.get_feature_importance(),
            "variant": variant,
        }
    ).sort_values("importance", ascending=False)
    audit = _audit(
        validation_predictions,
        validation_p,
        validation_decisions,
        validation_cont,
        validation_rev,
        cont_val_decision,
        rev_val_decision,
    )
    return {
        "variant": variant,
        "model_params": model_params,
        "market_feature_count": 0 if market_features is None else len(market_features),
        "stacker_feature_list": list(train_x.columns),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_frontier": frontier,
        "feature_importance": importance,
        "validation_audit": audit,
        "diagnostic_slices": _diagnostic_slices(audit),
        "target_met": bool(
            validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])
            and validation_metrics["selection_score"] > BASELINE_SELECTION_SCORE
            and validation_metrics["utility"] > 0.0
            and validation_metrics["accepted_sample_accuracy"] > 0.50
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    expert_features = _select_features(all_features, list(config["expert_feature_set"]["patterns"]))
    top_market_features = _select_top_market_features(all_features, config)
    leakage_features = sorted(name for name in set(expert_features + top_market_features) if not _is_safe_feature(name))
    if leakage_features:
        raise ValueError(f"leakage features selected: {leakage_features}")

    oof_cont, oof_rev, expert_generation = _chronological_oof_expert_predictions(
        config,
        train_frame,
        train_predictions,
        expert_features,
    )
    validation_cont, validation_rev = _train_expert_pair(
        config,
        train_frame,
        validation_frame,
        train_predictions,
        expert_features,
    )

    variants: list[dict[str, Any]] = []
    base_params = dict(config["stacker_model"])
    if config["experiments"].get("run_expert_only", True):
        variants.append(
            _run_variant(
                config,
                "experiment_a_expert_only",
                base_params,
                None,
                train_frame,
                validation_frame,
                train_predictions,
                validation_predictions,
                oof_cont,
                oof_rev,
                validation_cont,
                validation_rev,
            )
        )
    if config["experiments"].get("run_expert_top_market", True):
        variants.append(
            _run_variant(
                config,
                "experiment_b_expert_top50_market",
                base_params,
                top_market_features,
                train_frame,
                validation_frame,
                train_predictions,
                validation_predictions,
                oof_cont,
                oof_rev,
                validation_cont,
                validation_rev,
            )
        )
    if config["experiments"].get("run_regularization_grid", True):
        for depth in config["regularization_grid"]["depth_values"]:
            for l2_leaf_reg in config["regularization_grid"]["l2_leaf_reg_values"]:
                params = dict(base_params)
                params["depth"] = int(depth)
                params["l2_leaf_reg"] = float(l2_leaf_reg)
                variants.append(
                    _run_variant(
                        config,
                        f"experiment_c_grid_depth{int(depth)}_l2{float(l2_leaf_reg):g}",
                        params,
                        top_market_features,
                        train_frame,
                        validation_frame,
                        train_predictions,
                        validation_predictions,
                        oof_cont,
                        oof_rev,
                        validation_cont,
                        validation_rev,
                    )
                )

    best = max(
        variants,
        key=lambda row: (
            row["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"]),
            row["validation_metrics"]["selection_score"],
            row["validation_metrics"]["utility"],
            row["validation_metrics"]["coverage"],
            row["validation_metrics"]["accepted_count"],
        ),
    )
    summary_rows = []
    for row in variants:
        metrics = row["validation_metrics"]
        summary_rows.append(
            {
                "variant": row["variant"],
                "depth": row["model_params"]["depth"],
                "l2_leaf_reg": row["model_params"]["l2_leaf_reg"],
                "feature_count": len(row["stacker_feature_list"]),
                "market_feature_count": row["market_feature_count"],
                "coverage": metrics["coverage"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_count": metrics["accepted_count"],
                "up_prediction_count": metrics["up_prediction_count"],
                "down_prediction_count": metrics["down_prediction_count"],
                "selected_t_up": metrics["selected_t_up"],
                "selected_t_down": metrics["selected_t_down"],
                "coverage_constraint_satisfied": metrics["coverage"] >= float(config["objective"]["min_coverage"]),
                "target_met": row["target_met"],
            }
        )
    threshold_summary_path = output_dir / "threshold_search_summary.csv"
    pd.DataFrame(summary_rows).sort_values(["coverage_constraint_satisfied", "selection_score"], ascending=False).to_csv(
        threshold_summary_path,
        index=False,
    )
    frontier_path = output_dir / "threshold_search_frontier.csv"
    frontier = best["validation_frontier"].copy()
    frontier.insert(0, "variant", best["variant"])
    frontier.to_csv(frontier_path, index=False)
    importance_path = output_dir / "stacked_feature_importance.csv"
    pd.concat([row["feature_importance"] for row in variants], ignore_index=True).to_csv(importance_path, index=False)
    audit_path = output_dir / "validation_audit.csv"
    best["validation_audit"].to_csv(audit_path, index=False)

    validation_metrics = best["validation_metrics"]
    improved_over_baseline = bool(
        validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])
        and validation_metrics["selection_score"] > BASELINE_SELECTION_SCORE
        and validation_metrics["utility"] > 0.0
        and validation_metrics["accepted_sample_accuracy"] > 0.50
    )
    report_path = output_dir / "report.json"
    report = {
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "report_path": str(report_path),
        "primary_metric": "validation selection_score with coverage >= 0.70",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "train_window": _window_summary(train_frame.loc[oof_cont.index]),
        "validation_window": _window_summary(validation_frame),
        "train_metrics": _required_metrics(best["train_metrics"]),
        "validation_metrics": _required_metrics(validation_metrics),
        "signal_coverage": float(validation_metrics["signal_coverage"]),
        "coverage_constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
        "improved_over_accepted_baseline": improved_over_baseline,
        "deploy_training_mode": "not_regenerated",
        "offline_validation_metric_source": "validation threshold-tuned stacked meta-model evaluation only",
        "baseline_comparison": {
            "baseline_experiment_id": "20260520_polymarket_resolved_extended_history_baseline",
            "before_selection_score": BASELINE_SELECTION_SCORE,
            "after_selection_score": float(validation_metrics["selection_score"]),
            "before_utility": BASELINE_UTILITY,
            "after_utility": float(validation_metrics["utility"]),
            "before_accepted_sample_accuracy": BASELINE_ACCEPTED_SAMPLE_ACCURACY,
            "after_accepted_sample_accuracy": float(validation_metrics["accepted_sample_accuracy"]),
            "before_signal_count": BASELINE_ACCEPTED_COUNT,
            "after_signal_count": float(validation_metrics["accepted_count"]),
            "before_coverage": BASELINE_COVERAGE,
            "after_coverage": float(validation_metrics["coverage"]),
            "selection_score_improved_under_coverage_constraint": improved_over_baseline,
        },
        "leakage_feature_check": {
            "blocked_tokens": list(LEAKAGE_TOKENS),
            "leakage_features_selected": leakage_features,
            "passed": len(leakage_features) == 0,
        },
        "label_source": "polymarket_resolved",
        "label_version": "polymarket_resolved_gamma_v1",
        "expert_prediction_generation_method": expert_generation,
        "expert_feature_count": len(expert_features),
        "market_feature_source": config["market_feature_set"]["importance_path"],
        "top_market_features": top_market_features,
        "best_variant": best["variant"],
        "stacker_feature_list": best["stacker_feature_list"],
        "variant_summary": summary_rows,
        "diagnostic_slices": best["diagnostic_slices"],
        "output_files": {
            "threshold_search_summary": str(threshold_summary_path),
            "threshold_search_frontier": str(frontier_path),
            "stacked_feature_importance": str(importance_path),
            "validation_audit": str(audit_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation + reversal stacked meta-model experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "best_variant": report["best_variant"],
                "coverage_constraint_satisfied": report["coverage_constraint_satisfied"],
                "improved_over_accepted_baseline": report["improved_over_accepted_baseline"],
                "coverage": metrics["coverage"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_count": metrics["accepted_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
