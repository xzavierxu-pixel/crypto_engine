from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import Settings
from src.data.derivatives.basis_loader import normalize_basis_frame
from src.data.derivatives.book_ticker_loader import normalize_book_ticker_frame
from src.data.derivatives.funding_loader import normalize_funding_frame
from src.data.derivatives.oi_loader import normalize_oi_frame
from src.data.derivatives.options_loader import normalize_options_frame


def _normalize_symbol_hint(raw_symbol: str) -> str:
    before_settlement = raw_symbol.split(":", 1)[0]
    return "".join(character for character in before_settlement if character.isalnum())


def resolve_um_symbol(settings: Settings) -> str:
    configured_symbols = settings.data_backfill.futures_um.symbols
    if configured_symbols:
        return configured_symbols[0]

    if settings.derivatives.symbol_perp:
        return _normalize_symbol_hint(settings.derivatives.symbol_perp)
    if settings.market.pair:
        return _normalize_symbol_hint(settings.market.pair)
    raise ValueError("Unable to resolve Binance UM symbol for archive derivatives inputs.")


def _resolve_normalized_root(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved.parent.parent.parent
    return resolved


def _load_normalized_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Normalized derivatives archive file not found: {path}")
    return pd.read_parquet(path)


def _downsample_to_minute_last(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    downsampled = frame.copy()
    downsampled["timestamp"] = pd.to_datetime(downsampled["timestamp"], utc=True).dt.floor("1min")
    downsampled = downsampled.sort_values("timestamp").groupby("timestamp", as_index=False).last()
    return downsampled.reset_index(drop=True)


def _load_optional_normalized_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _normalize_bvol_options_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "index_value" in normalized.columns and "atm_iv_near" not in normalized.columns:
        normalized = normalized.rename(columns={"index_value": "atm_iv_near"})
        atm_iv = pd.to_numeric(normalized["atm_iv_near"], errors="coerce")
        if atm_iv.dropna().median() > 5.0:
            normalized["atm_iv_near"] = atm_iv / 100.0
    normalized = _downsample_to_minute_last(normalized)
    normalized["exchange"] = normalized.get("exchange", "binance")
    if "iv_term_slope" not in normalized.columns:
        normalized["iv_term_slope"] = 0.0
    return normalize_options_frame(normalized)


def _normalize_eoh_summary_options_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "atm_iv_near", "iv_term_slope", "exchange", "symbol", "source_version"])

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).dt.floor("1h")
    normalized["mark_iv"] = pd.to_numeric(normalized["mark_iv"], errors="coerce")
    normalized["delta"] = pd.to_numeric(normalized.get("delta"), errors="coerce")
    normalized["openinterest_usdt"] = pd.to_numeric(normalized.get("openinterest_usdt"), errors="coerce").fillna(0.0)
    if "expiry" not in normalized.columns and "strike" in normalized.columns:
        normalized["expiry"] = normalized["strike"].astype(str).str.extract(r"(\d{6})", expand=False)

    normalized = normalized.dropna(subset=["timestamp", "mark_iv"])
    if normalized.empty:
        return pd.DataFrame(columns=["timestamp", "atm_iv_near", "iv_term_slope", "exchange", "symbol", "source_version"])

    near_atm = normalized.copy()
    near_atm["delta_distance"] = (near_atm["delta"].abs() - 0.5).abs()
    near_atm = near_atm.sort_values(
        ["timestamp", "delta_distance", "openinterest_usdt"],
        ascending=[True, True, False],
    )
    atm = near_atm.groupby("timestamp", as_index=False).first()[["timestamp", "mark_iv"]].rename(
        columns={"mark_iv": "atm_iv_near"}
    )

    expiry_iv = (
        normalized.dropna(subset=["expiry"])
        .groupby(["timestamp", "expiry"], as_index=False)["mark_iv"]
        .median()
        .sort_values(["timestamp", "expiry"])
    )
    slopes: list[dict[str, object]] = []
    for timestamp, group in expiry_iv.groupby("timestamp", sort=True):
        if len(group) < 2:
            slope = 0.0
        else:
            slope = float(group["mark_iv"].iloc[-1] - group["mark_iv"].iloc[0])
        slopes.append({"timestamp": timestamp, "iv_term_slope": slope})
    slope_frame = pd.DataFrame(slopes)
    result = atm.merge(slope_frame, on="timestamp", how="left")
    result["iv_term_slope"] = result["iv_term_slope"].fillna(0.0)
    result["exchange"] = "binance"
    result["symbol"] = "BTCUSDT"
    result["source_version"] = "binance_public_eoh_summary_options_v1"
    return normalize_options_frame(result)


def _merge_options_sources(primary: pd.DataFrame | None, fallback: pd.DataFrame | None) -> pd.DataFrame:
    frames = [frame for frame in (primary, fallback) if frame is not None and not frame.empty]
    if not frames:
        raise FileNotFoundError("No normalized options archive files were found.")
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    source_rank = merged["source_version"].astype("string").str.contains("eoh_summary", na=False).astype(int)
    merged = merged.assign(_source_rank=source_rank).sort_values(["timestamp", "_source_rank"])
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").drop(columns=["_source_rank"])
    return normalize_options_frame(merged)


def load_archive_funding_frame(
    normalized_root: str | Path,
    *,
    symbol: str,
) -> pd.DataFrame:
    root = _resolve_normalized_root(normalized_root)
    path = root / "futures_um" / "fundingRate" / f"{symbol}.parquet"
    frame = _load_normalized_frame(path).copy()
    if "last_funding_rate" in frame.columns and "funding_rate" not in frame.columns:
        frame = frame.rename(columns={"last_funding_rate": "funding_rate"})
    frame["exchange"] = frame.get("exchange", "binance")
    return normalize_funding_frame(frame)


def _extract_basis_component(
    normalized_root: Path,
    *,
    symbol: str,
    data_type: str,
    value_column: str,
) -> tuple[pd.DataFrame, str | None] | tuple[None, None]:
    path = normalized_root / "futures_um" / data_type / f"{symbol}-1m.parquet"
    if not path.exists():
        return None, None

    frame = pd.read_parquet(path)
    component = frame.loc[:, ["timestamp", "close"]].rename(columns={"close": value_column})
    source_version = None
    if "source_version" in frame.columns and not frame.empty:
        source_version = str(frame["source_version"].dropna().iloc[-1])
    return component, source_version


def load_archive_basis_frame(
    normalized_root: str | Path,
    *,
    symbol: str,
) -> pd.DataFrame:
    root = _resolve_normalized_root(normalized_root)
    components: list[pd.DataFrame] = []
    source_versions: list[str] = []

    for data_type, value_column in (
        ("markPriceKlines", "mark_price"),
        ("indexPriceKlines", "index_price"),
        ("premiumIndexKlines", "premium_index"),
    ):
        component, source_version = _extract_basis_component(
            root,
            symbol=symbol,
            data_type=data_type,
            value_column=value_column,
        )
        if component is not None:
            components.append(component)
        if source_version is not None:
            source_versions.append(source_version)

    if not components:
        raise FileNotFoundError(
            f"No normalized mark/index/premium archive files were found under {root} for {symbol}."
        )

    merged = components[0]
    for component in components[1:]:
        merged = merged.merge(component, on="timestamp", how="outer", sort=True)

    merged["exchange"] = "binance"
    merged["symbol"] = symbol
    if source_versions:
        merged["source_version"] = source_versions[-1]

    return normalize_basis_frame(merged)


def load_archive_oi_frame(
    normalized_root: str | Path,
    *,
    symbol: str,
) -> pd.DataFrame:
    root = _resolve_normalized_root(normalized_root)
    path = root / "futures_um" / "metrics" / f"{symbol}.parquet"
    frame = _load_normalized_frame(path).copy()
    rename_map: dict[str, str] = {}
    if "sum_open_interest" in frame.columns:
        rename_map["sum_open_interest"] = "open_interest"
    if "sum_open_interest_value" in frame.columns:
        rename_map["sum_open_interest_value"] = "oi_notional"
    frame = frame.rename(columns=rename_map)
    frame["exchange"] = frame.get("exchange", "binance")
    return normalize_oi_frame(frame)


def load_archive_options_frame(
    normalized_root: str | Path,
    *,
    symbol: str,
) -> pd.DataFrame:
    root = _resolve_normalized_root(normalized_root)
    bvol_path = root / "option" / "BVOLIndex" / f"{symbol}.parquet"
    eoh_path = root / "option" / "EOHSummary" / "BTCUSDT.parquet"
    bvol_frame = _load_optional_normalized_frame(bvol_path)
    eoh_frame = _load_optional_normalized_frame(eoh_path)
    normalized_bvol = _normalize_bvol_options_frame(bvol_frame) if bvol_frame is not None else None
    normalized_eoh = _normalize_eoh_summary_options_frame(eoh_frame) if eoh_frame is not None else None
    return _merge_options_sources(normalized_eoh, normalized_bvol)


def load_archive_book_ticker_frame(
    normalized_root: str | Path,
    *,
    symbol: str,
) -> pd.DataFrame:
    root = _resolve_normalized_root(normalized_root)
    path = root / "futures_um" / "bookTicker" / f"{symbol}.parquet"
    frame = _load_normalized_frame(path).copy()
    frame["exchange"] = frame.get("exchange", "binance")
    return normalize_book_ticker_frame(frame)
