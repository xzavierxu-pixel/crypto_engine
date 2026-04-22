from __future__ import annotations

from src.features.asymmetry import AsymmetryFeaturePack
from src.features.base import FeaturePack
from src.features.candle_structure import CandleStructureFeaturePack
from src.features.compression_breakout import CompressionBreakoutFeaturePack
from src.features.derivatives_basis import DerivativesBasisFeaturePack
from src.features.derivatives_book_ticker import DerivativesBookTickerFeaturePack
from src.features.derivatives_funding import DerivativesFundingFeaturePack
from src.features.derivatives_oi import DerivativesOIFeaturePack
from src.features.derivatives_options import DerivativesOptionsFeaturePack
from src.features.flow_proxy import FlowProxyFeaturePack
from src.features.htf_context import HTFContextFeaturePack
from src.features.lagged import LaggedFeaturePack
from src.features.market_quality import MarketQualityFeaturePack
from src.features.momentum import MomentumFeaturePack
from src.features.momentum_acceleration import MomentumAccelerationFeaturePack
from src.features.path_structure import PathStructureFeaturePack
from src.features.regime import RegimeFeaturePack
from src.features.time_features import TimeFeaturePack
from src.features.intra_5m_structure import Intra5mStructureFeaturePack
from src.features.volume import VolumeFeaturePack
from src.features.volatility import VolatilityFeaturePack


FEATURE_PACKS: dict[str, FeaturePack] = {
    "momentum": MomentumFeaturePack(),
    "momentum_acceleration": MomentumAccelerationFeaturePack(),
    "volatility": VolatilityFeaturePack(),
    "path_structure": PathStructureFeaturePack(),
    "regime": RegimeFeaturePack(),
    "volume": VolumeFeaturePack(),
    "candle_structure": CandleStructureFeaturePack(),
    "market_quality": MarketQualityFeaturePack(),
    "htf_context": HTFContextFeaturePack(),
    "compression_breakout": CompressionBreakoutFeaturePack(),
    "asymmetry": AsymmetryFeaturePack(),
    "flow_proxy": FlowProxyFeaturePack(),
    "intra_5m_structure": Intra5mStructureFeaturePack(),
    "lagged": LaggedFeaturePack(),
    "time": TimeFeaturePack(),
    "derivatives_funding": DerivativesFundingFeaturePack(),
    "derivatives_basis": DerivativesBasisFeaturePack(),
    "derivatives_book_ticker": DerivativesBookTickerFeaturePack(),
    "derivatives_oi": DerivativesOIFeaturePack(),
    "derivatives_options": DerivativesOptionsFeaturePack(),
}


def get_feature_pack(name: str) -> FeaturePack:
    try:
        return FEATURE_PACKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown feature pack '{name}'.") from exc
