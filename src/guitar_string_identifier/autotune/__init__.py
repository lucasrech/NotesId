"""Módulo de Auto Tuning (configuração automática por processamento clássico de sinais)."""

from guitar_string_identifier.autotune.adaptive_params import AdaptiveParams
from guitar_string_identifier.autotune.device_probe import DeviceProbe
from guitar_string_identifier.autotune.hum_detector import HumDetector
from guitar_string_identifier.autotune.level_analyzer import LevelAnalyzer
from guitar_string_identifier.autotune.noise_floor import NoiseFloorEstimator
from guitar_string_identifier.autotune.session import AutoTuneSession
from guitar_string_identifier.autotune.threshold_calibrator import ThresholdCalibrator
from guitar_string_identifier.autotune.tuning_detector import TuningDetector
from guitar_string_identifier.autotune.weight_estimator import WeightEstimator

__all__ = [
    "AdaptiveParams",
    "AutoTuneSession",
    "DeviceProbe",
    "HumDetector",
    "LevelAnalyzer",
    "NoiseFloorEstimator",
    "ThresholdCalibrator",
    "TuningDetector",
    "WeightEstimator",
]
