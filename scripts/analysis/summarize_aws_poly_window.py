from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATCHED_PATH = ROOT / "artifacts/aws_poly_since_20260607_analysis.json"
SUMMARY_DIR = ROOT / "artifacts/artifacts/logs/execution_engine/summaries"

UPDATES = [
    ("U01", "2026-06-07T06:46:08Z", "切换 regime_reversal_rank_blend 并转 live"),
    ("U02", "2026-06-07T12:42:47Z", "恢复/确认 live profitable baseline"),
    ("U03", "2026-06-07T12:46:05Z", "首单改为 best-bid/cap 定价并加入 edge 配置"),
    ("U04", "2026-06-07T13:05:29Z", "best_bid_offset -0.8 修正为 -0.08"),
    ("U05", "2026-06-07T13:26:21Z", "首单 cap 0.55→0.60，随后到 0.65/offset 0"),
    ("U06", "2026-06-09T01:32:23Z", "启用 execution edge：min_edge 0.06/notional 4"),
    ("U07", "2026-06-09T09:46:33Z", "最大订单名义额 4→10"),
    ("U08", "2026-06-09T12:00:11Z", "min_edge 0.06→0"),
    ("U09", "2026-06-09T14:11:47Z", "首单 size 10→5"),
    ("U10", "2026-06-09T17:31:06Z", "最大订单名义额 10→5"),
    ("U11", "2026-06-12T14:58:40Z", "CatBoost 日历阈值 + q80 价格估计"),
]


def f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bucket(value: float) -> str:
    lo = min(9, max(0, int(value * 10))) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def main() -> None:
    report = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))
    actual_by_slug = {row["slug"]: row.get("actual_side") for row in report["cycles"]}
    fills_by_order: dict[str, list[dict]] = defaultdict(list)
    for row in report["orders"]:
        fills_by_order[row["order_id"]].append(row)

    orders = []
    for path in sorted(SUMMARY_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        signal = payload.get("signal") or {}
        market = payload.get("market") or {}
        p_up = f(signal.get("p_up"))
        confidence = max(p_up, 1 - p_up)
        actual = actual_by_slug.get(market.get("slug"))
        responses = payload.get("responses") or []
        for i, order in enumerate(payload.get("orders") or []):
            response = ((responses[i].get("response") or {}) if i < len(responses) else {})
            order_id = response.get("orderID") or response.get("order_id") or response.get("id")
            success = response.get("success") is True and bool(order_id)
            price, size, side = f(order.get("price")), f(order.get("size")), order.get("side")
            fill_rows = fills_by_order.get(order_id, [])
            matched_size = sum(f(x.get("matched_size")) for x in fill_rows)
            cost = sum(f(x.get("notional")) for x in fill_rows)
            pnl = sum(f(x.get("pnl")) for x in fill_rows)
            correct = side == actual if actual in {"YES", "NO"} else None
            orders.append({
                "order_id": order_id, "side": side, "actual": actual, "success": success,
                "t0": signal.get("t0"),
                "price": price, "size": size, "confidence": confidence,
                "matched_size": matched_size, "cost": cost, "pnl": pnl, "correct": correct,
            })

    def metrics(rows: list[dict]) -> dict:
        successful = [x for x in rows if x["success"]]
        filled = [x for x in successful if x["matched_size"] > 0]
        requested = sum(x["size"] for x in successful)
        matched = sum(x["matched_size"] for x in filled)
        wins = sum(x["correct"] is True for x in filled)
        replay_pnl = sum(x["size"] * (1 - x["price"]) if x["correct"] else -x["size"] * x["price"] for x in successful if x["correct"] is not None)
        cost = sum(x["cost"] for x in filled)
        pnl = sum(x["pnl"] for x in filled)
        return {
            "orders": len(rows), "submitted": len(successful), "submit_rate": len(successful) / len(rows) if rows else 0,
            "filled_orders": len(filled), "order_fill_rate": len(filled) / len(successful) if successful else 0,
            "requested_shares": requested, "matched_shares": matched, "size_fill_rate": matched / requested if requested else 0,
            "winning_filled_orders": wins, "filled_order_win_rate": wins / len(filled) if filled else 0,
            "avg_limit_price": sum(x["price"] for x in successful) / len(successful) if successful else 0,
            "avg_fill_price": cost / matched if matched else 0, "matched_cost": cost, "pnl": pnl,
            "roi": pnl / cost if cost else 0, "submitted_replay_pnl": replay_pnl,
        }

    def grouped(field: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in orders:
            groups[bucket(row[field])].append(row)
        return {key: metrics(value) for key, value in sorted(groups.items())}

    update_metrics = []
    boundaries = [(key, datetime.fromisoformat(ts.replace("Z", "+00:00")), change) for key, ts, change in UPDATES]
    end = datetime.fromisoformat(report["window_end_utc"])
    for index, (key, start, change) in enumerate(boundaries):
        stop = boundaries[index + 1][1] if index + 1 < len(boundaries) else end
        rows = [x for x in orders if x["t0"] and start <= datetime.fromisoformat(x["t0"].replace("Z", "+00:00")) < stop]
        value = metrics(rows)
        filled = [x for x in rows if x["success"] and x["matched_size"] > 0]
        value.update({
            "update": key, "start_utc": start.isoformat(), "end_utc": stop.isoformat(),
            "duration_hours": (stop - start).total_seconds() / 3600, "change": change,
            "pnl_per_hour": value["pnl"] / ((stop - start).total_seconds() / 3600),
            "mean_filled_order_roi": sum(x["pnl"] / x["cost"] for x in filled if x["cost"] > 0) / len(filled) if filled else 0,
        })
        update_metrics.append(value)

    output = {
        "window_start_utc": report["window_start_utc"], "window_end_utc": report["window_end_utc"],
        "cycles": len(report["cycles"]), "overall": metrics(orders),
        "by_limit_price": grouped("price"), "by_model_confidence": grouped("confidence"),
        "by_update": update_metrics,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
