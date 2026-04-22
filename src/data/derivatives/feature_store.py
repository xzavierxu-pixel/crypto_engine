from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Settings
from src.data.binance_public.derivatives_archive import (
    load_archive_basis_frame,
    load_archive_book_ticker_frame,
    load_archive_funding_frame,
    load_archive_oi_frame,
    load_archive_options_frame,
    resolve_um_symbol,
)
from src.data.derivatives.aligner import align_derivatives_to_spot, merge_derivatives_frames
from src.data.derivatives.basis_loader import load_basis_frame
from src.data.derivatives.book_ticker_loader import load_book_ticker_frame
from src.data.derivatives.funding_loader import load_funding_frame
from src.data.derivatives.oi_loader import load_oi_frame
from src.data.derivatives.options_loader import load_options_frame


RAW_DERIVATIVES_RENAME_MAP = {
    "funding_rate": "raw_funding_rate",
    "funding_effective_time": "raw_funding_effective_time",
    "mark_price": "raw_mark_price",
    "index_price": "raw_index_price",
    "premium_index": "raw_premium_index",
    "exchange": "derivatives_exchange",
    "symbol": "derivatives_symbol",
    "funding_source_version": "raw_funding_source_version",
    "basis_source_version": "raw_basis_source_version",
    "open_interest": "raw_open_interest",
    "oi_notional": "raw_oi_notional",
    "oi_source_version": "raw_oi_source_version",
    "atm_iv_near": "raw_atm_iv_near",
    "iv_term_slope": "raw_iv_term_slope",
    "options_source_version": "raw_options_source_version",
    "bid_price": "raw_bid_price",
    "bid_qty": "raw_bid_qty",
    "ask_price": "raw_ask_price",
    "ask_qty": "raw_ask_qty",
    "book_ticker_source_version": "raw_book_ticker_source_version",
    "source_version": "raw_source_version",
}

DERIVATIVES_PATH_MODES = {"latest", "archive"}


def _normalize_path_mode(path_mode: str | None) -> str:
    normalized = (path_mode or "latest").strip().lower()
    if normalized not in DERIVATIVES_PATH_MODES:
        supported = ", ".join(sorted(DERIVATIVES_PATH_MODES))
        raise ValueError(f"Unsupported derivatives path mode '{path_mode}'. Expected one of: {supported}.")
    return normalized


def _resolve_source_path(
    *,
    enabled: bool,
    source_name: str,
    latest_path: str | Path | None,
    archive_path: str | Path | None,
    override_path: str | Path | None,
    path_mode: str,
) -> str | None:
    if override_path is not None:
        return str(override_path)
    if not enabled:
        return None
    if path_mode == "archive":
        if archive_path:
            return str(archive_path)
        raise ValueError(
            f"Derivatives {source_name} is enabled with path_mode='archive' but no archive_path is configured."
        )
    return str(latest_path) if latest_path else None


def resolve_derivatives_paths(
    settings: Settings,
    *,
    funding_path: str | Path | None = None,
    basis_path: str | Path | None = None,
    oi_path: str | Path | None = None,
    options_path: str | Path | None = None,
    book_ticker_path: str | Path | None = None,
    path_mode: str | None = None,
) -> dict[str, Any]:
    resolved_path_mode = _normalize_path_mode(path_mode or settings.derivatives.path_mode)
    top_level_enabled = settings.derivatives.enabled
    return {
        "path_mode": resolved_path_mode,
        "funding_path": _resolve_source_path(
            enabled=top_level_enabled and settings.derivatives.funding.enabled,
            source_name="funding",
            latest_path=settings.derivatives.funding.path,
            archive_path=settings.derivatives.funding.archive_path,
            override_path=funding_path,
            path_mode=resolved_path_mode,
        ),
        "basis_path": _resolve_source_path(
            enabled=top_level_enabled and settings.derivatives.basis.enabled,
            source_name="basis",
            latest_path=settings.derivatives.basis.path,
            archive_path=settings.derivatives.basis.archive_path,
            override_path=basis_path,
            path_mode=resolved_path_mode,
        ),
        "oi_path": _resolve_source_path(
            enabled=top_level_enabled and settings.derivatives.oi.enabled,
            source_name="oi",
            latest_path=settings.derivatives.oi.path,
            archive_path=settings.derivatives.oi.archive_path,
            override_path=oi_path,
            path_mode=resolved_path_mode,
        ),
        "options_path": _resolve_source_path(
            enabled=top_level_enabled and settings.derivatives.options.enabled,
            source_name="options",
            latest_path=settings.derivatives.options.path,
            archive_path=settings.derivatives.options.archive_path,
            override_path=options_path,
            path_mode=resolved_path_mode,
        ),
        "book_ticker_path": _resolve_source_path(
            enabled=top_level_enabled and settings.derivatives.book_ticker.enabled,
            source_name="book_ticker",
            latest_path=settings.derivatives.book_ticker.path,
            archive_path=settings.derivatives.book_ticker.archive_path,
            override_path=book_ticker_path,
            path_mode=resolved_path_mode,
        ),
    }


class DerivativesFeatureStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_raw_frame(
        self,
        funding_frame: pd.DataFrame | None = None,
        basis_frame: pd.DataFrame | None = None,
        oi_frame: pd.DataFrame | None = None,
        options_frame: pd.DataFrame | None = None,
        book_ticker_frame: pd.DataFrame | None = None,
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if derivatives_frame is not None:
            return derivatives_frame.copy().rename(columns=RAW_DERIVATIVES_RENAME_MAP)

        if (
            funding_frame is None
            and basis_frame is None
            and oi_frame is None
            and options_frame is None
            and book_ticker_frame is None
        ):
            loaded = load_derivatives_frame_from_settings(self.settings)
            if loaded is None:
                return pd.DataFrame()
            return loaded.rename(columns=RAW_DERIVATIVES_RENAME_MAP)

        resolved_funding = funding_frame
        resolved_basis = basis_frame
        resolved_oi = oi_frame
        resolved_options = options_frame
        resolved_book_ticker = book_ticker_frame

        if resolved_funding is None and self.settings.derivatives.funding.enabled:
            path = self.settings.derivatives.funding.path
            if not path:
                raise ValueError("Derivatives funding is enabled but no funding path is configured.")
            resolved_funding = load_funding_frame(Path(path))

        if resolved_basis is None and self.settings.derivatives.basis.enabled:
            path = self.settings.derivatives.basis.path
            if not path:
                raise ValueError("Derivatives basis is enabled but no basis path is configured.")
            resolved_basis = load_basis_frame(Path(path))

        if resolved_oi is None and self.settings.derivatives.oi.enabled:
            path = self.settings.derivatives.oi.path
            if not path:
                raise ValueError("Derivatives OI is enabled but no OI path is configured.")
            resolved_oi = load_oi_frame(Path(path))

        if resolved_options is None and self.settings.derivatives.options.enabled:
            path = self.settings.derivatives.options.path
            if not path:
                raise ValueError("Derivatives options is enabled but no options path is configured.")
            resolved_options = load_options_frame(Path(path))

        if resolved_book_ticker is None and self.settings.derivatives.book_ticker.enabled:
            path = self.settings.derivatives.book_ticker.path
            if not path:
                raise ValueError("Derivatives book ticker is enabled but no book ticker path is configured.")
            resolved_book_ticker = load_book_ticker_frame(Path(path))

        return merge_derivatives_frames(
            funding_frame=resolved_funding,
            basis_frame=resolved_basis,
            oi_frame=resolved_oi,
            options_frame=resolved_options,
            book_ticker_frame=resolved_book_ticker,
        ).rename(columns=RAW_DERIVATIVES_RENAME_MAP)

    def attach_to_spot(
        self,
        spot_frame: pd.DataFrame,
        funding_frame: pd.DataFrame | None = None,
        basis_frame: pd.DataFrame | None = None,
        oi_frame: pd.DataFrame | None = None,
        options_frame: pd.DataFrame | None = None,
        book_ticker_frame: pd.DataFrame | None = None,
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raw_frame = self.load_raw_frame(
            funding_frame=funding_frame,
            basis_frame=basis_frame,
            oi_frame=oi_frame,
            options_frame=options_frame,
            book_ticker_frame=book_ticker_frame,
            derivatives_frame=derivatives_frame,
        )
        return align_derivatives_to_spot(spot_frame, raw_frame)


def _load_archive_frame(
    *,
    settings: Settings,
    source_name: str,
    archive_path: str | Path | None,
) -> pd.DataFrame | None:
    if not archive_path:
        return None

    symbol = resolve_um_symbol(settings)
    resolved_archive_path = Path(archive_path)
    if source_name == "funding":
        return load_archive_funding_frame(resolved_archive_path, symbol=symbol)
    if source_name == "basis":
        return load_archive_basis_frame(resolved_archive_path, symbol=symbol)
    if source_name == "oi":
        return load_archive_oi_frame(resolved_archive_path, symbol=symbol)
    if source_name == "options":
        options_symbol = "BTCBVOLUSDT"
        configured_symbols = settings.data_backfill.option.symbols.get("BVOLIndex", [])
        if configured_symbols:
            options_symbol = configured_symbols[0]
        return load_archive_options_frame(resolved_archive_path, symbol=options_symbol)
    if source_name == "book_ticker":
        return load_archive_book_ticker_frame(resolved_archive_path, symbol=symbol)
    return None


def load_derivatives_frame_from_paths(
    funding_path: str | Path | None = None,
    basis_path: str | Path | None = None,
    oi_path: str | Path | None = None,
    options_path: str | Path | None = None,
    book_ticker_path: str | Path | None = None,
) -> pd.DataFrame | None:
    resolved_funding = load_funding_frame(Path(funding_path)) if funding_path else None
    resolved_basis = load_basis_frame(Path(basis_path)) if basis_path else None
    resolved_oi = load_oi_frame(Path(oi_path)) if oi_path else None
    resolved_options = load_options_frame(Path(options_path)) if options_path else None
    resolved_book_ticker = load_book_ticker_frame(Path(book_ticker_path)) if book_ticker_path else None
    if (
        resolved_funding is None
        and resolved_basis is None
        and resolved_oi is None
        and resolved_options is None
        and resolved_book_ticker is None
    ):
        return None
    return merge_derivatives_frames(
        funding_frame=resolved_funding,
        basis_frame=resolved_basis,
        oi_frame=resolved_oi,
        options_frame=resolved_options,
        book_ticker_frame=resolved_book_ticker,
    )


def load_derivatives_frame_from_settings(
    settings: Settings,
    *,
    funding_path: str | Path | None = None,
    basis_path: str | Path | None = None,
    oi_path: str | Path | None = None,
    options_path: str | Path | None = None,
    book_ticker_path: str | Path | None = None,
    path_mode: str | None = None,
) -> pd.DataFrame | None:
    resolved = resolve_derivatives_paths(
        settings,
        funding_path=funding_path,
        basis_path=basis_path,
        oi_path=oi_path,
        options_path=options_path,
        book_ticker_path=book_ticker_path,
        path_mode=path_mode,
    )
    if resolved["path_mode"] == "archive":
        archive_funding = _load_archive_frame(
            settings=settings,
            source_name="funding",
            archive_path=resolved["funding_path"],
        )
        archive_basis = _load_archive_frame(
            settings=settings,
            source_name="basis",
            archive_path=resolved["basis_path"],
        )
        archive_oi = _load_archive_frame(
            settings=settings,
            source_name="oi",
            archive_path=resolved["oi_path"],
        )
        archive_options = _load_archive_frame(
            settings=settings,
            source_name="options",
            archive_path=resolved["options_path"],
        )
        archive_book_ticker = _load_archive_frame(
            settings=settings,
            source_name="book_ticker",
            archive_path=resolved["book_ticker_path"],
        )
        if (
            archive_funding is None
            and archive_basis is None
            and archive_oi is None
            and archive_options is None
            and archive_book_ticker is None
        ):
            return None
        return merge_derivatives_frames(
            funding_frame=archive_funding,
            basis_frame=archive_basis,
            oi_frame=archive_oi,
            options_frame=archive_options,
            book_ticker_frame=archive_book_ticker,
        )

    return load_derivatives_frame_from_paths(
        funding_path=resolved["funding_path"],
        basis_path=resolved["basis_path"],
        oi_path=resolved["oi_path"],
        options_path=resolved["options_path"],
        book_ticker_path=resolved["book_ticker_path"],
    )
