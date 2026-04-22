from __future__ import annotations

from scripts.download_derivatives_public_data import (
    _normalize_basis_records,
    _normalize_deribit_vol_rows,
    _normalize_funding_records,
    _normalize_oi_records,
)


def test_normalize_funding_records_maps_binance_payload_to_repo_schema() -> None:
    frame = _normalize_funding_records(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": 1711929600000,
                "fundingRate": "0.0001",
            }
        ]
    )

    assert frame.loc[0, "funding_rate"] == 0.0001
    assert str(frame.loc[0, "timestamp"]) == "2024-04-01 00:00:00+00:00"
    assert frame.loc[0, "source_version"] == "binance_fapi_funding_v1"


def test_normalize_basis_records_maps_binance_basis_payload_to_repo_schema() -> None:
    frame = _normalize_basis_records(
        [
            {
                "pair": "BTCUSDT",
                "timestamp": 1711929900000,
                "futuresPrice": "70100.0",
                "indexPrice": "70080.0",
                "basisRate": "0.000285",
            }
        ]
    )

    assert frame.loc[0, "mark_price"] == 70100.0
    assert frame.loc[0, "index_price"] == 70080.0
    assert frame.loc[0, "premium_index"] == 0.000285


def test_normalize_oi_records_maps_binance_open_interest_payload_to_repo_schema() -> None:
    frame = _normalize_oi_records(
        [
            {
                "symbol": "BTCUSDT",
                "timestamp": 1711929900000,
                "sumOpenInterest": "91234.5",
                "sumOpenInterestValue": "6400000000.0",
            }
        ]
    )

    assert frame.loc[0, "open_interest"] == 91234.5
    assert frame.loc[0, "oi_notional"] == 6400000000.0


def test_normalize_deribit_vol_rows_uses_close_as_decimal_iv_proxy() -> None:
    frame = _normalize_deribit_vol_rows(
        [
            [1711929600000, 52.0, 53.0, 51.0, 52.5],
        ]
    )

    assert frame.loc[0, "atm_iv_near"] == 0.525
    assert frame.loc[0, "exchange"] == "deribit"
    assert frame.loc[0, "source_version"] == "deribit_volatility_index_v1"
