from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


OBJECTIVE_METRIC_FIELDS = [
    "sample_count",
    "coverage",
    "precision_up",
    "precision_down",
    "balanced_precision",
    "all_sample_accuracy",
    "accepted_sample_accuracy",
    "share_up_predictions",
    "share_down_predictions",
    "selected_t_up",
    "selected_t_down",
    "accepted_count",
    "up_prediction_count",
    "down_prediction_count",
    "roc_auc",
    "brier_score",
    "log_loss",
    "utility",
    "downside_risk",
    "selection_score",
    "up_signal_count",
    "down_signal_count",
    "total_signal_count",
    "signal_coverage",
    "overall_signal_accuracy",
]


def _opposite_side(side: pd.Series) -> pd.Series:
    return side.map({"YES": "NO", "NO": "YES"})


def apply_continuation_expert_decision(
    p_up: pd.Series,
    first_minute_side: pd.Series,
    regimes: pd.Series,
    thresholds: dict[str, float],
) -> pd.Series:
    """Apply same-side-only continuation thresholds by regime."""
    probability = p_up.astype("float64").clip(0.0, 1.0)
    fm_side = first_minute_side.astype("object")
    decisions = pd.Series("ABSTAIN", index=probability.index, dtype="object")
    for regime, threshold in thresholds.items():
        mask = regimes == regime
        decisions.loc[mask & (fm_side == "YES") & (probability >= float(threshold))] = "UP"
        decisions.loc[mask & (fm_side == "NO") & (probability <= float(threshold))] = "DOWN"
    return decisions


def apply_reversal_only_decision(
    p_up: pd.Series,
    first_minute_side: pd.Series,
    *,
    t_up: float,
    t_down: float,
) -> pd.Series:
    """Apply opposite-side-only reversal thresholds."""
    probability = p_up.astype("float64").clip(0.0, 1.0)
    fm_side = first_minute_side.astype("object")
    decisions = pd.Series("ABSTAIN", index=probability.index, dtype="object")
    decisions.loc[(fm_side == "YES") & (probability <= float(t_down))] = "DOWN"
    decisions.loc[(fm_side == "NO") & (probability >= float(t_up))] = "UP"
    return decisions


def route_conflict_margin_hybrid(
    continuation_p_up: pd.Series,
    continuation_decision: pd.Series,
    reversal_p_up: pd.Series,
    reversal_decision: pd.Series,
    *,
    conflict_margin: float,
) -> pd.DataFrame:
    """Route between disagreeing continuation/reversal experts with a confidence margin."""
    cont_decision = continuation_decision.astype("object")
    rev_decision = reversal_decision.astype("object")
    cont_accept = cont_decision != "ABSTAIN"
    rev_accept = rev_decision != "ABSTAIN"
    cont_conf = (continuation_p_up.astype("float64").clip(0.0, 1.0) - 0.5).abs()
    rev_conf = (reversal_p_up.astype("float64").clip(0.0, 1.0) - 0.5).abs()

    final_decision = pd.Series("ABSTAIN", index=cont_decision.index, dtype="object")
    used_expert = pd.Series("abstain", index=cont_decision.index, dtype="object")
    routing_reason = pd.Series("both_abstain", index=cont_decision.index, dtype="object")

    cont_only = cont_accept & ~rev_accept
    rev_only = rev_accept & ~cont_accept
    both = cont_accept & rev_accept
    cont_win = both & ((cont_conf - rev_conf) >= float(conflict_margin))
    rev_win = both & ((rev_conf - cont_conf) >= float(conflict_margin))
    conflict_abstain = both & ~(cont_win | rev_win)

    final_decision.loc[cont_only | cont_win] = cont_decision.loc[cont_only | cont_win]
    used_expert.loc[cont_only | cont_win] = "continuation"
    routing_reason.loc[cont_only] = "cont_only"
    routing_reason.loc[cont_win] = "conflict_cont_win"

    final_decision.loc[rev_only | rev_win] = rev_decision.loc[rev_only | rev_win]
    used_expert.loc[rev_only | rev_win] = "reversal"
    routing_reason.loc[rev_only] = "rev_only"
    routing_reason.loc[rev_win] = "conflict_rev_win"
    routing_reason.loc[conflict_abstain] = "conflict_abstain"

    return pd.DataFrame(
        {
            "final_decision": final_decision,
            "used_expert": used_expert,
            "routing_reason": routing_reason,
            "continuation_accept": cont_accept,
            "reversal_accept": rev_accept,
            "continuation_confidence": cont_conf,
            "reversal_confidence": rev_conf,
        }
    )


def route_continuation_first_reversal_fallback(
    continuation_decision: pd.Series,
    reversal_decision: pd.Series,
) -> pd.DataFrame:
    """Use reversal decisions only where the continuation expert abstains."""
    cont_decision = continuation_decision.astype("object")
    rev_decision = reversal_decision.astype("object")
    if len(cont_decision) != len(rev_decision):
        raise ValueError("continuation_decision and reversal_decision must have the same length.")

    cont_accept = cont_decision != "ABSTAIN"
    rev_accept = rev_decision != "ABSTAIN"
    fallback_accept = ~cont_accept & rev_accept

    final_decision = pd.Series("ABSTAIN", index=cont_decision.index, dtype="object")
    final_source = pd.Series("abstain", index=cont_decision.index, dtype="object")
    final_decision.loc[cont_accept] = cont_decision.loc[cont_accept]
    final_source.loc[cont_accept] = "continuation"
    final_decision.loc[fallback_accept] = rev_decision.loc[fallback_accept]
    final_source.loc[fallback_accept] = "reversal_fallback"

    return pd.DataFrame(
        {
            "final_decision": final_decision,
            "final_source": final_source,
            "continuation_accept": cont_accept,
            "reversal_accept": rev_accept,
            "reversal_fallback_accept": fallback_accept,
        }
    )


def p_follow_from_direction_probability(predictions: pd.DataFrame) -> pd.Series:
    """Recover p_follow from the follow artifact's final-direction p_up output."""
    if "p_up" not in predictions.columns or "first_minute_side" not in predictions.columns:
        raise ValueError("follow predictions require p_up and first_minute_side columns.")
    p_up = pd.to_numeric(predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    fm_side = predictions["first_minute_side"].astype("object")
    p_follow = p_up.where(fm_side == "YES", 1.0 - p_up)
    return p_follow.astype("float64").clip(0.0, 1.0)


def apply_four_bucket_abstain_gate(
    frame: pd.DataFrame,
    *,
    p_follow_cutoff: float,
    base_band: float,
    baseline_decision_column: str = "base_decision",
    p_base_column: str = "p_base",
    p_follow_column: str = "p_follow",
) -> pd.DataFrame:
    required = {baseline_decision_column, p_base_column, p_follow_column, "first_minute_side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"hybrid gate frame is missing required columns: {missing}")

    output = frame.copy()
    p_base = pd.to_numeric(output[p_base_column], errors="coerce").astype("float64").clip(0.0, 1.0)
    p_follow = pd.to_numeric(output[p_follow_column], errors="coerce").astype("float64").clip(0.0, 1.0)
    fm_side = output["first_minute_side"].astype("object")
    base_decision = output[baseline_decision_column].astype("object")
    base_side = base_decision.replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
    follow_side = fm_side.where(p_follow >= 0.5, _opposite_side(fm_side))

    base_accepted = base_decision != "ABSTAIN"
    base_continuation = base_accepted & (base_side == fm_side)
    base_reversal = base_accepted & (base_side != fm_side)
    follow_continuation = follow_side == fm_side
    follow_matches_base = follow_side == base_side
    weak_base_band = (p_base - 0.5).abs() <= float(base_band)

    keep = pd.Series(False, index=output.index, dtype="bool")
    keep_continuation = base_continuation & ~(~follow_continuation & (p_follow <= p_follow_cutoff) & weak_base_band)
    keep_reversal = base_reversal & follow_matches_base & (p_follow <= p_follow_cutoff)
    keep.loc[keep_continuation | keep_reversal] = True

    hybrid_decision = pd.Series("ABSTAIN", index=output.index, dtype="object")
    hybrid_decision.loc[keep] = base_decision.loc[keep]
    output["follow_side"] = follow_side
    output["hybrid_decision"] = hybrid_decision
    output["hybrid_accepted"] = keep
    output["hybrid_predicted_side"] = hybrid_decision.replace({"UP": "YES", "DOWN": "NO"})
    output["hybrid_bucket"] = assign_four_buckets(base_side=base_side, fm_side=fm_side, follow_side=follow_side)
    output["p_follow_cutoff"] = float(p_follow_cutoff)
    output["base_band"] = float(base_band)
    return output


def assign_four_buckets(*, base_side: pd.Series, fm_side: pd.Series, follow_side: pd.Series) -> pd.Series:
    bucket = pd.Series("unaccepted_or_unknown", index=base_side.index, dtype="object")
    known = base_side.isin(["YES", "NO"]) & fm_side.isin(["YES", "NO"]) & follow_side.isin(["YES", "NO"])
    base_eq_fm = base_side == fm_side
    follow_eq_fm = follow_side == fm_side
    bucket.loc[known & base_eq_fm & follow_eq_fm] = "bucket_1_base_fm_follow_fm"
    bucket.loc[known & base_eq_fm & ~follow_eq_fm] = "bucket_2_base_fm_follow_reversal"
    bucket.loc[known & ~base_eq_fm & (follow_side == base_side)] = "bucket_3_base_reversal_follow_base"
    bucket.loc[known & ~base_eq_fm & (follow_side != base_side)] = "bucket_4_base_reversal_follow_fm"
    return bucket


def compute_decision_metrics(
    y_true: pd.Series,
    p_up: pd.Series,
    decisions: pd.Series,
    *,
    selected_t_up: float,
    selected_t_down: float,
) -> dict[str, float]:
    if not (len(y_true) == len(p_up) == len(decisions)):
        raise ValueError("y_true, p_up, and decisions must have the same length.")

    y = y_true.astype(int)
    probability = p_up.astype("float64").clip(0.0, 1.0)
    decision = decisions.astype("object")
    accepted = decision != "ABSTAIN"
    up_mask = decision == "UP"
    down_mask = decision == "DOWN"
    accepted_count = int(accepted.sum())
    up_count = int(up_mask.sum())
    down_count = int(down_mask.sum())
    accepted_correct = ((decision.loc[accepted] == "UP") == (y.loc[accepted] == 1)) if accepted_count else pd.Series(dtype="bool")
    hard_predictions = (probability >= 0.5).astype(int)

    coverage = float(accepted_count / len(y)) if len(y) else 0.0
    accepted_sample_accuracy = float(accepted_correct.mean()) if accepted_count else 0.0
    utility = float(coverage * (2.0 * accepted_sample_accuracy - 1.0))
    downside_risk = float(math.sqrt(max(coverage * (1.0 - accepted_sample_accuracy), 0.0)))
    if downside_risk > 0.0:
        selection_score = float(utility / downside_risk)
    elif utility > 0.0:
        selection_score = float("inf")
    elif utility < 0.0:
        selection_score = float("-inf")
    else:
        selection_score = 0.0

    metrics = {
        "sample_count": float(len(y)),
        "coverage": coverage,
        "precision_up": float((y.loc[up_mask] == 1).mean()) if up_count else 0.0,
        "precision_down": float((y.loc[down_mask] == 0).mean()) if down_count else 0.0,
        "all_sample_accuracy": float(accuracy_score(y, hard_predictions)) if len(y) else 0.0,
        "accepted_sample_accuracy": accepted_sample_accuracy,
        "share_up_predictions": float(up_count / accepted_count) if accepted_count else 0.0,
        "share_down_predictions": float(down_count / accepted_count) if accepted_count else 0.0,
        "selected_t_up": float(selected_t_up),
        "selected_t_down": float(selected_t_down),
        "accepted_count": float(accepted_count),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "utility": utility,
        "downside_risk": downside_risk,
        "selection_score": selection_score,
    }
    metrics["balanced_precision"] = float((metrics["precision_up"] + metrics["precision_down"]) / 2.0)
    metrics["roc_auc"] = float(roc_auc_score(y, probability)) if y.nunique() == 2 else 0.0
    metrics["brier_score"] = float(brier_score_loss(y, probability)) if len(y) else 0.0
    metrics["log_loss"] = float(log_loss(y, pd.concat([1.0 - probability, probability], axis=1), labels=[0, 1])) if len(y) else 0.0
    metrics.update(
        {
            "up_signal_count": metrics["up_prediction_count"],
            "down_signal_count": metrics["down_prediction_count"],
            "total_signal_count": metrics["accepted_count"],
            "signal_coverage": metrics["coverage"],
            "overall_signal_accuracy": metrics["accepted_sample_accuracy"],
        }
    )
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def compute_reversal_continuation_metrics(frame: pd.DataFrame, decisions: pd.Series) -> dict[str, float]:
    required = {"target", "first_minute_side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reversal metrics frame is missing required columns: {missing}")
    y = frame["target"].astype(int)
    fm_side = frame["first_minute_side"].astype("object")
    resolved_side = pd.Series(np.where(y == 1, "YES", "NO"), index=frame.index)
    known = fm_side.isin(["YES", "NO"])
    continuation = known & (fm_side == resolved_side)
    reversal = known & ~continuation
    accepted = decisions != "ABSTAIN"
    predicted_up = decisions == "UP"
    correct = ((predicted_up == (y == 1)) & accepted)

    def payload(prefix: str, mask: pd.Series) -> dict[str, float]:
        sample_count = int(mask.sum())
        accepted_mask = mask & accepted
        accepted_count = int(accepted_mask.sum())
        return {
            f"{prefix}_sample_count": float(sample_count),
            f"{prefix}_coverage": float(accepted_count / sample_count) if sample_count else 0.0,
            f"{prefix}_accepted_accuracy": float(correct.loc[accepted_mask].mean()) if accepted_count else 0.0,
            f"{prefix}_accepted_count": float(accepted_count),
        }

    return {**payload("continuation", continuation), **payload("reversal", reversal)}


def summarize_buckets(frame: pd.DataFrame, decisions: pd.Series, buckets: pd.Series) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    y = frame["target"].astype(int)
    accepted = decisions != "ABSTAIN"
    predicted_up = decisions == "UP"
    correct = (predicted_up == (y == 1)) & accepted
    reversal = frame["post_first_minute_reversal"].astype(bool) if "post_first_minute_reversal" in frame else pd.Series(False, index=frame.index)
    for name in sorted(str(value) for value in buckets.dropna().unique()):
        mask = buckets == name
        count = int(mask.sum())
        accepted_mask = mask & accepted
        accepted_count = int(accepted_mask.sum())
        records.append(
            {
                "bucket": name,
                "count": float(count),
                "accepted_count": float(accepted_count),
                "accuracy": float(correct.loc[accepted_mask].mean()) if accepted_count else 0.0,
                "reversal_share": float(reversal.loc[mask].mean()) if count else 0.0,
            }
        )
    return records


def search_four_bucket_gate(
    frame: pd.DataFrame,
    *,
    p_follow_cutoffs: Iterable[float],
    base_bands: Iterable[float],
    min_coverage: float,
) -> tuple[pd.DataFrame, dict[str, float | bool | str | None]]:
    records = []
    eligible = []
    for cutoff in p_follow_cutoffs:
        for band in base_bands:
            gated = apply_four_bucket_abstain_gate(frame, p_follow_cutoff=float(cutoff), base_band=float(band))
            metrics = compute_decision_metrics(
                gated["target"],
                gated["p_base"],
                gated["hybrid_decision"],
                selected_t_up=float(cutoff),
                selected_t_down=float(band),
            )
            row = {"p_follow_cutoff": float(cutoff), "base_band": float(band), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["accepted_sample_accuracy"] > 0.50 and metrics["utility"] > 0.0:
                eligible.append(row)

    frontier = pd.DataFrame.from_records(records)
    if not records:
        raise ValueError("hybrid gate search produced no candidates.")
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
            -abs(row["p_follow_cutoff"] - 0.5) - abs(row["base_band"]),
        ),
    )
    return frontier, {
        **best,
        "constraint_satisfied": bool(eligible),
        "fallback_reason": None if eligible else "no gate candidate satisfied coverage/accuracy/utility constraints",
        "objective": "selection_score",
        "hard_constraint": "coverage_only",
    }
