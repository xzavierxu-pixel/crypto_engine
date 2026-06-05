from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _side_from_thresholds(p_up: float, t_up: float, t_down: float) -> str | None:
    if p_up >= t_up:
        return "YES"
    if p_up <= t_down:
        return "NO"
    return None


def _selection_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    sample_count = len(rows)
    accepted = [row for row in rows if row.get("side") in {"YES", "NO"}]
    accepted_count = len(accepted)
    up_count = sum(1 for row in accepted if row["side"] == "YES")
    down_count = sum(1 for row in accepted if row["side"] == "NO")
    correct_count = sum(1 for row in accepted if row.get("correct"))
    coverage = accepted_count / sample_count if sample_count else 0.0
    accepted_accuracy = correct_count / accepted_count if accepted_count else 0.0
    utility = coverage * (2.0 * accepted_accuracy - 1.0)
    downside_risk = math.sqrt(max(coverage * (1.0 - accepted_accuracy), 0.0))
    selection_score = utility / downside_risk if downside_risk > 0.0 else 0.0
    return {
        "sample_count": float(sample_count),
        "coverage": float(coverage),
        "accepted_sample_accuracy": float(accepted_accuracy),
        "accepted_count": float(accepted_count),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "share_up_predictions": float(up_count / accepted_count) if accepted_count else 0.0,
        "share_down_predictions": float(down_count / accepted_count) if accepted_count else 0.0,
        "wins": float(correct_count),
        "losses": float(accepted_count - correct_count),
        "utility": float(utility),
        "downside_risk": float(downside_risk),
        "selection_score": float(selection_score),
    }


def _bucket_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key)), []).append(row)
    return {name: _selection_metrics(bucket_rows) for name, bucket_rows in sorted(buckets.items())}


def _load_first_minute_sides(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    frame = pd.read_parquet(path, columns=["timestamp", "open", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["first_minute_return"] = (frame["close"].astype(float) / frame["open"].astype(float)) - 1.0
    frame["first_minute_side"] = frame["first_minute_return"].map(lambda value: "YES" if value >= 0.0 else "NO")
    return {
        ts.isoformat(): {
            "first_minute_return": float(ret),
            "first_minute_side": str(side),
        }
        for ts, ret, side in zip(
            frame["timestamp"],
            frame["first_minute_return"],
            frame["first_minute_side"],
            strict=True,
        )
    }


def _attach_reversal_trend(rows: list[dict[str, Any]], first_minute_sides: dict[str, dict[str, Any]]) -> bool:
    if not first_minute_sides:
        return False
    available = 0
    for row in rows:
        t0 = pd.Timestamp(row["t0"]).tz_convert("UTC").isoformat()
        first_minute = first_minute_sides.get(t0)
        if first_minute is None:
            continue
        actual_side = row.get("actual_side")
        first_minute_side = first_minute["first_minute_side"]
        row["first_minute_return"] = first_minute["first_minute_return"]
        row["first_minute_side"] = first_minute_side
        row["trend_following"] = bool(first_minute_side == actual_side)
        row["post_first_minute_reversal"] = bool(first_minute_side != actual_side)
        available += 1
    return available == len(rows)


def _reversal_trend_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    trend_rows = [row for row in rows if row.get("trend_following") is True]
    reversal_rows = [row for row in rows if row.get("post_first_minute_reversal") is True]
    accepted = [row for row in rows if row.get("accepted")]
    total_losses = sum(1 for row in accepted if not row.get("correct"))
    trend_losses = sum(1 for row in trend_rows if row.get("accepted") and not row.get("correct"))
    reversal_losses = sum(1 for row in reversal_rows if row.get("accepted") and not row.get("correct"))
    trend_metrics = _selection_metrics(trend_rows)
    reversal_metrics = _selection_metrics(reversal_rows)
    return {
        "continuation_sample_count": trend_metrics["sample_count"],
        "continuation_coverage": trend_metrics["coverage"],
        "continuation_accepted_accuracy": trend_metrics["accepted_sample_accuracy"],
        "continuation_accepted_count": trend_metrics["accepted_count"],
        "reversal_sample_count": reversal_metrics["sample_count"],
        "reversal_coverage": reversal_metrics["coverage"],
        "reversal_accepted_accuracy": reversal_metrics["accepted_sample_accuracy"],
        "reversal_accepted_count": reversal_metrics["accepted_count"],
        "reversal_loss_contribution": float(reversal_losses / total_losses) if total_losses else 0.0,
        "trend_following_loss_contribution": float(trend_losses / total_losses) if total_losses else 0.0,
    }


def _find_combo(payload: dict[str, Any], combo_name: str) -> dict[str, Any]:
    for combo in payload.get("combos", []):
        if combo.get("name") == combo_name:
            return combo
    raise KeyError(f"Replay combo not found: {combo_name}")


def build_replay_summary(
    replay_payload: dict[str, Any],
    *,
    combo_name: str,
    run_id: str,
    t_up: float,
    t_down: float,
    baseline_report_path: str | None = None,
    kline_1m_path: str | None = None,
) -> dict[str, Any]:
    combo = _find_combo(replay_payload, combo_name)
    rows: list[dict[str, Any]] = []
    for row in combo.get("rows", []):
        p_up = float(row["p_up"])
        side = _side_from_thresholds(p_up, t_up, t_down)
        actual_side = row.get("actual_side")
        rows.append(
            {
                **row,
                "run_id": run_id,
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "t_up": float(t_up),
                "t_down": float(t_down),
                "side": side,
                "accepted": side is not None,
                "correct": bool(side == actual_side) if side is not None else False,
            }
        )
    first_minute_sides = _load_first_minute_sides(kline_1m_path)
    reversal_trend_available = _attach_reversal_trend(rows, first_minute_sides)
    reversal_trend = _reversal_trend_metrics(rows) if reversal_trend_available else {}
    return {
        "run_id": run_id,
        "source_replay_window_start_utc": replay_payload.get("window_start_utc"),
        "source_replay_window_end_utc": replay_payload.get("window_end_utc"),
        "source_combo": combo_name,
        "baseline_report_path": baseline_report_path,
        "selected_t_up": float(t_up),
        "selected_t_down": float(t_down),
        "metrics": _selection_metrics(rows),
        "by_confidence_bucket": _bucket_summary(rows, "confidence_bucket"),
        "source_artifact": replay_payload.get("current_artifact") if combo_name.startswith("current_") else replay_payload.get("old_artifact"),
        "reversal_trend_metrics_available": reversal_trend_available,
        "reversal_trend_metrics": reversal_trend,
        "reversal_trend_metrics_note": None
        if reversal_trend_available
        else (
            "Source replay rows do not contain first-minute return/side and no local 1m kline path was provided. "
            "Use validation report reversal_trend_slices for baseline reversal diagnostics."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an existing replay window with explicit thresholds.")
    parser.add_argument("--input", required=True, help="Existing replay_old_vs_current JSON.")
    parser.add_argument("--combo", default="current_model_current_thresholds")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--t-up", required=True, type=float)
    parser.add_argument("--t-down", required=True, type=float)
    parser.add_argument("--baseline-report-path")
    parser.add_argument("--kline-1m-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = build_replay_summary(
        payload,
        combo_name=args.combo,
        run_id=args.run_id,
        t_up=args.t_up,
        t_down=args.t_down,
        baseline_report_path=args.baseline_report_path,
        kline_1m_path=args.kline_1m_path,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
