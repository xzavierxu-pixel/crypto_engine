from __future__ import annotations

import pandas as pd

from src.core.constants import DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.validation import normalize_ohlcv_frame


def _combine_metadata_columns(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_values = primary.astype("object")
    fallback_values = fallback.astype("object")
    return primary_values.where(primary_values.notna(), fallback_values)


def merge_derivatives_frames(
    funding_frame: pd.DataFrame | None = None,
    basis_frame: pd.DataFrame | None = None,
    oi_frame: pd.DataFrame | None = None,
    options_frame: pd.DataFrame | None = None,
    book_ticker_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    if funding_frame is not None and not funding_frame.empty:
        funding = funding_frame.copy()
        if "source_version" in funding.columns:
            funding = funding.rename(columns={"source_version": "funding_source_version"})
        merged = funding

    if basis_frame is not None and not basis_frame.empty:
        basis = basis_frame.copy()
        if "source_version" in basis.columns:
            basis = basis.rename(columns={"source_version": "basis_source_version"})
        if merged is None:
            merged = basis
        else:
            merged = merged.merge(
                basis,
                on=DEFAULT_TIMESTAMP_COLUMN,
                how="outer",
                suffixes=("_funding", "_basis"),
            )
            for column in ("exchange", "symbol"):
                funding_column = f"{column}_funding"
                basis_column = f"{column}_basis"
                if funding_column in merged.columns or basis_column in merged.columns:
                    funding_values = (
                        merged[funding_column]
                        if funding_column in merged.columns
                        else pd.Series(pd.NA, index=merged.index, dtype="object")
                    )
                    basis_values = (
                        merged[basis_column]
                        if basis_column in merged.columns
                        else pd.Series(pd.NA, index=merged.index, dtype="object")
                    )
                    merged[column] = _combine_metadata_columns(funding_values, basis_values)
                    merged = merged.drop(
                        columns=[name for name in (funding_column, basis_column) if name in merged.columns]
                    )
            if DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN in merged.columns:
                merged[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN] = pd.to_datetime(
                    merged[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN],
                    utc=True,
                )

    if oi_frame is not None and not oi_frame.empty:
        oi = oi_frame.copy()
        if "source_version" in oi.columns:
            oi = oi.rename(columns={"source_version": "oi_source_version"})
        if merged is None:
            merged = oi
        else:
            merged = merged.merge(
                oi,
                on=DEFAULT_TIMESTAMP_COLUMN,
                how="outer",
                suffixes=("", "_oi"),
            )
            for column in ("exchange", "symbol"):
                oi_column = f"{column}_oi"
                if oi_column in merged.columns:
                    base_values = (
                        merged[column]
                        if column in merged.columns
                        else pd.Series(pd.NA, index=merged.index, dtype="object")
                    )
                    merged[column] = _combine_metadata_columns(base_values, merged[oi_column])
                    merged = merged.drop(columns=[oi_column])

    if options_frame is not None and not options_frame.empty:
        options = options_frame.copy()
        if "source_version" in options.columns:
            options = options.rename(columns={"source_version": "options_source_version"})
        if merged is None:
            merged = options
        else:
            merged = merged.merge(
                options,
                on=DEFAULT_TIMESTAMP_COLUMN,
                how="outer",
                suffixes=("", "_options"),
            )
            for column in ("exchange", "symbol"):
                options_column = f"{column}_options"
                if options_column in merged.columns:
                    base_values = (
                        merged[column]
                        if column in merged.columns
                        else pd.Series(pd.NA, index=merged.index, dtype="object")
                    )
                    merged[column] = _combine_metadata_columns(base_values, merged[options_column])
                    merged = merged.drop(columns=[options_column])

    if book_ticker_frame is not None and not book_ticker_frame.empty:
        book_ticker = book_ticker_frame.copy()
        if "source_version" in book_ticker.columns:
            book_ticker = book_ticker.rename(columns={"source_version": "book_ticker_source_version"})
        if merged is None:
            merged = book_ticker
        else:
            merged = merged.merge(
                book_ticker,
                on=DEFAULT_TIMESTAMP_COLUMN,
                how="outer",
                suffixes=("", "_book_ticker"),
            )
            for column in ("exchange", "symbol"):
                book_ticker_column = f"{column}_book_ticker"
                if book_ticker_column in merged.columns:
                    base_values = (
                        merged[column]
                        if column in merged.columns
                        else pd.Series(pd.NA, index=merged.index, dtype="object")
                    )
                    merged[column] = _combine_metadata_columns(base_values, merged[book_ticker_column])
                    merged = merged.drop(columns=[book_ticker_column])

    if merged is None:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN])

    merged[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(merged[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    merged = merged.sort_values(DEFAULT_TIMESTAMP_COLUMN).drop_duplicates(
        subset=[DEFAULT_TIMESTAMP_COLUMN],
        keep="last",
    )
    fill_columns = [column for column in merged.columns if column != DEFAULT_TIMESTAMP_COLUMN]
    if fill_columns:
        merged[fill_columns] = merged[fill_columns].ffill()
    return merged.reset_index(drop=True)


def align_derivatives_to_spot(
    spot_frame: pd.DataFrame,
    derivatives_frame: pd.DataFrame,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    normalized_spot = normalize_ohlcv_frame(spot_frame, timestamp_column=timestamp_column, require_volume=False)
    if derivatives_frame.empty:
        return normalized_spot

    derivatives = derivatives_frame.copy()
    derivatives[timestamp_column] = pd.to_datetime(derivatives[timestamp_column], utc=True)
    if DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN in derivatives.columns:
        derivatives[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN] = pd.to_datetime(
            derivatives[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN],
            utc=True,
        )
    derivatives = derivatives.sort_values(timestamp_column).drop_duplicates(
        subset=[timestamp_column],
        keep="last",
    )

    aligned = pd.merge_asof(
        normalized_spot.sort_values(timestamp_column),
        derivatives.sort_values(timestamp_column),
        on=timestamp_column,
        direction="backward",
    )
    return aligned.reset_index(drop=True)
