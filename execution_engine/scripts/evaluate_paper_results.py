from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

OutcomeFetcher = Callable[[str], str | None]


@dataclass(frozen=True)
class PaperPrediction:
    summary_path: str
    signal_t0: datetime
    market_start: datetime
    slug: str
    p_up: float
    side: str | None
    actual_side: str | None
    feature_timestamp: str | None
    used_actual_market_slug: bool

    @property
    def predicted(self) -> bool:
        return self.side in {"YES", "NO"}

    @property
    def resolved(self) -> bool:
        return self.actual_side in {"YES", "NO"}

    @property
    def correct(self) -> bool | None:
        if not self.predicted or not self.resolved:
            return None
        return self.side == self.actual_side


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def market_start_from_slug(slug: str) -> datetime:
    return datetime.fromtimestamp(int(slug.rsplit("-", 1)[-1]), tz=UTC)


def build_btc_5m_slug(t0: datetime, *, offset_windows: int = 0) -> tuple[str, datetime, datetime]:
    ts = t0.astimezone(UTC).replace(second=0, microsecond=0)
    minute = ts.minute - (ts.minute % 5)
    window_start = ts.replace(minute=minute) + timedelta(minutes=5 * offset_windows)
    window_end = window_start + timedelta(minutes=5)
    return f"btc-updown-5m-{int(window_start.timestamp())}", window_start, window_end


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_side_from_thresholds(p_up: float, t_up: float, t_down: float) -> str | None:
    if p_up >= t_up:
        return "YES"
    if p_up <= t_down:
        return "NO"
    return None


class GammaOutcomeClient:
    def __init__(
        self,
        *,
        base_url: str = "https://gamma-api.polymarket.com",
        cache_path: str | Path | None = None,
        timeout_seconds: float = 15.0,
        sleep_seconds: float = 0.05,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = load_json(self.cache_path)

    def __call__(self, slug: str) -> str | None:
        cached = self.cache.get(slug)
        if not cached or not cached.get("resolved"):
            self.cache[slug] = self._fetch(slug)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(self.cache, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        outcome = self.cache[slug].get("outcome")
        return str(outcome) if outcome else None

    def _fetch(self, slug: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/markets",
            params={"slug": slug, "closed": "true"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return {"slug": slug, "resolved": False, "outcome": None, "reason": "not_found"}

        market = payload[0]
        outcomes = _json_list(market.get("outcomes"))
        prices = [float(value) for value in _json_list(market.get("outcomePrices"))]
        if not outcomes or len(outcomes) != len(prices):
            return {"slug": slug, "resolved": False, "outcome": None, "reason": "missing_outcome_prices"}

        winner_index = max(range(len(prices)), key=lambda index: prices[index])
        raw_outcome = str(outcomes[winner_index]).upper()
        if raw_outcome in {"UP", "YES"}:
            outcome = "YES"
        elif raw_outcome in {"DOWN", "NO"}:
            outcome = "NO"
        else:
            outcome = raw_outcome
        return {
            "slug": slug,
            "resolved": market.get("umaResolutionStatus") == "resolved" or market.get("closed") is True,
            "outcome": outcome,
            "outcomes": outcomes,
            "outcomePrices": prices,
            "closed": market.get("closed"),
            "umaResolutionStatus": market.get("umaResolutionStatus"),
            "closedTime": market.get("closedTime"),
        }


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def load_predictions(
    summary_dir: str | Path,
    *,
    outcome_fetcher: OutcomeFetcher,
    replay_thresholds: tuple[float, float] | None = None,
    use_actual_market_slug: bool = True,
) -> list[PaperPrediction]:
    predictions: list[PaperPrediction] = []
    for path in sorted(Path(summary_dir).glob("*.json")):
        payload = load_json(path)
        signal = payload.get("signal") or {}
        if signal.get("t0") is None or signal.get("p_up") is None:
            continue
        signal_t0 = parse_timestamp(signal["t0"])
        inferred_slug, inferred_start, _ = build_btc_5m_slug(signal_t0)
        actual_slug = (payload.get("market") or {}).get("slug")
        slug = actual_slug if use_actual_market_slug and actual_slug else inferred_slug
        market_start = market_start_from_slug(slug)

        if replay_thresholds is None:
            decision = payload.get("decision") or {}
            side = decision.get("side") if decision.get("should_trade") else None
        else:
            side = infer_side_from_thresholds(float(signal["p_up"]), replay_thresholds[0], replay_thresholds[1])

        predictions.append(
            PaperPrediction(
                summary_path=str(path),
                signal_t0=signal_t0,
                market_start=market_start if actual_slug else inferred_start,
                slug=slug,
                p_up=float(signal["p_up"]),
                side=side,
                actual_side=outcome_fetcher(slug),
                feature_timestamp=signal.get("feature_timestamp"),
                used_actual_market_slug=bool(actual_slug and use_actual_market_slug),
            )
        )
    return predictions


def summarize_predictions(
    predictions: list[PaperPrediction],
    *,
    min_hourly_predictions: int = 6,
    min_hourly_correct: int = 4,
    min_available_per_hour: int = 10,
) -> dict[str, Any]:
    resolved = [prediction for prediction in predictions if prediction.resolved]
    predicted = [prediction for prediction in resolved if prediction.predicted]
    correct_count = sum(prediction.correct is True for prediction in predicted)

    hourly: dict[datetime, dict[str, int | bool]] = defaultdict(
        lambda: {"available": 0, "predicted": 0, "correct": 0, "resolved": 0, "goal_passed": False}
    )
    for prediction in predictions:
        hour = prediction.market_start.replace(minute=0, second=0, microsecond=0)
        hourly[hour]["available"] += 1
        hourly[hour]["resolved"] += int(prediction.resolved)
        hourly[hour]["predicted"] += int(prediction.predicted)
        hourly[hour]["correct"] += int(prediction.correct is True)

    complete_hourly = {
        hour: stats
        for hour, stats in sorted(hourly.items())
        if int(stats["available"]) >= min_available_per_hour
    }
    for stats in complete_hourly.values():
        stats["goal_passed"] = (
            int(stats["predicted"]) >= min_hourly_predictions
            and int(stats["correct"]) >= min_hourly_correct
        )

    mismatches = [
        prediction
        for prediction in predictions
        if prediction.used_actual_market_slug and prediction.signal_t0 != prediction.market_start
    ]
    return {
        "sample_count": len(predictions),
        "resolved_count": len(resolved),
        "accepted_count": len(predicted),
        "correct_count": correct_count,
        "coverage": len(predicted) / len(resolved) if resolved else 0.0,
        "accepted_sample_accuracy": correct_count / len(predicted) if predicted else 0.0,
        "share_up_predictions": sum(prediction.side == "YES" for prediction in predicted) / len(predicted)
        if predicted
        else 0.0,
        "share_down_predictions": sum(prediction.side == "NO" for prediction in predicted) / len(predicted)
        if predicted
        else 0.0,
        "hourly_goal": {
            "min_predictions": min_hourly_predictions,
            "min_correct": min_hourly_correct,
            "min_available_per_hour": min_available_per_hour,
            "complete_hour_count": len(complete_hourly),
            "passed_hour_count": sum(bool(stats["goal_passed"]) for stats in complete_hourly.values()),
            "hourly": {hour.isoformat(): stats for hour, stats in complete_hourly.items()},
        },
        "signal_market_start_mismatch_count": len(mismatches),
        "signal_market_start_mismatches": [asdict(prediction) for prediction in mismatches[:20]],
    }


def threshold_search(
    summary_dir: str | Path,
    *,
    outcome_fetcher: OutcomeFetcher,
    t_up_min: float,
    t_up_max: float,
    t_down_min: float,
    t_down_max: float,
    step: float,
    min_hourly_predictions: int,
    min_hourly_correct: int,
    min_available_per_hour: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scale = 1_000_000
    for t_up_i in range(round(t_up_min * scale), round(t_up_max * scale) + 1, round(step * scale)):
        for t_down_i in range(round(t_down_min * scale), round(t_down_max * scale) + 1, round(step * scale)):
            t_up = t_up_i / scale
            t_down = t_down_i / scale
            if t_down >= t_up:
                continue
            predictions = load_predictions(
                summary_dir,
                outcome_fetcher=outcome_fetcher,
                replay_thresholds=(t_up, t_down),
                use_actual_market_slug=False,
            )
            summary = summarize_predictions(
                predictions,
                min_hourly_predictions=min_hourly_predictions,
                min_hourly_correct=min_hourly_correct,
                min_available_per_hour=min_available_per_hour,
            )
            results.append(
                {
                    "selected_t_up": t_up,
                    "selected_t_down": t_down,
                    "accepted_count": summary["accepted_count"],
                    "correct_count": summary["correct_count"],
                    "coverage": summary["coverage"],
                    "accepted_sample_accuracy": summary["accepted_sample_accuracy"],
                    "passed_hour_count": summary["hourly_goal"]["passed_hour_count"],
                    "complete_hour_count": summary["hourly_goal"]["complete_hour_count"],
                    "hourly": summary["hourly_goal"]["hourly"],
                }
            )
    results.sort(
        key=lambda item: (
            item["passed_hour_count"],
            min((stats["correct"] for stats in item["hourly"].values()), default=0),
            min((stats["predicted"] for stats in item["hourly"].values()), default=0),
            item["accepted_sample_accuracy"],
            item["correct_count"],
        ),
        reverse=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate execution_engine paper/live summaries against resolved markets.")
    parser.add_argument("--summary-dir", default="artifacts/logs/execution_engine/summaries")
    parser.add_argument("--gamma-base-url", default="https://gamma-api.polymarket.com")
    parser.add_argument("--cache-path", default="artifacts/state/execution_engine/outcome_cache.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--min-hourly-predictions", type=int, default=6)
    parser.add_argument("--min-hourly-correct", type=int, default=4)
    parser.add_argument("--min-available-per-hour", type=int, default=10)
    parser.add_argument("--threshold-search", action="store_true")
    parser.add_argument("--t-up-min", type=float, default=0.50)
    parser.add_argument("--t-up-max", type=float, default=0.65)
    parser.add_argument("--t-down-min", type=float, default=0.35)
    parser.add_argument("--t-down-max", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()

    outcome_client = GammaOutcomeClient(base_url=args.gamma_base_url, cache_path=args.cache_path)
    predictions = load_predictions(args.summary_dir, outcome_fetcher=outcome_client)
    report = summarize_predictions(
        predictions,
        min_hourly_predictions=args.min_hourly_predictions,
        min_hourly_correct=args.min_hourly_correct,
        min_available_per_hour=args.min_available_per_hour,
    )
    if args.threshold_search:
        report["threshold_search"] = threshold_search(
            args.summary_dir,
            outcome_fetcher=outcome_client,
            t_up_min=args.t_up_min,
            t_up_max=args.t_up_max,
            t_down_min=args.t_down_min,
            t_down_max=args.t_down_max,
            step=args.step,
            min_hourly_predictions=args.min_hourly_predictions,
            min_hourly_correct=args.min_hourly_correct,
            min_available_per_hour=args.min_available_per_hour,
        )[:20]

    encoded = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
