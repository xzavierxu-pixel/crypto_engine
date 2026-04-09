from __future__ import annotations

from src.calibration.base import CalibrationPlugin
from src.calibration.isotonic import IsotonicCalibration
from src.calibration.none import NoCalibration
from src.calibration.platt import PlattScalingCalibration
from src.core.config import Settings


CALIBRATION_PLUGINS: dict[str, type[CalibrationPlugin]] = {
    "none": NoCalibration,
    "isotonic": IsotonicCalibration,
    "platt": PlattScalingCalibration,
}


def create_calibration_plugin(settings: Settings, plugin_name: str | None = None) -> CalibrationPlugin:
    target = plugin_name or settings.calibration.active_plugin
    try:
        plugin_cls = CALIBRATION_PLUGINS[target]
    except KeyError as exc:
        raise KeyError(f"Unknown calibration plugin '{target}'.") from exc
    return plugin_cls()


def load_calibration_plugin(plugin_name: str, path: str) -> CalibrationPlugin:
    try:
        plugin_cls = CALIBRATION_PLUGINS[plugin_name]
    except KeyError as exc:
        raise KeyError(f"Unknown calibration plugin '{plugin_name}'.") from exc
    return plugin_cls.load(path)
