import numpy as np

from price_estimator.safe_lowest_price_gap.backtest_experiment_pnl import realized_order_pnl


def test_realized_order_pnl_uses_conservative_wrong_fill_rule() -> None:
    pnl, filled, printed = realized_order_pnl(
        correct=np.array([True, True, False, False]),
        chosen_low=np.array([0.20, 0.30, np.nan, 0.50]),
        bid=np.array([0.25, 0.25, 0.10, 0.00]),
    )

    assert printed.tolist() == [True, False, False, False]
    assert filled.tolist() == [True, False, True, False]
    assert pnl.tolist() == [0.75, 0.0, -0.10, 0.0]


def test_realized_order_pnl_at_p_side_price() -> None:
    pnl, filled, _ = realized_order_pnl(
        correct=np.array([True, False]),
        chosen_low=np.array([0.40, 0.01]),
        bid=np.array([0.60, 0.55]),
    )

    assert filled.tolist() == [True, True]
    assert pnl.tolist() == [0.40, -0.55]
