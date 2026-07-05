"""Predictor Engine. Turns ``HardwareProfile`` + intent into a ranked recommendation.

Public entry points: ``recommend``, ``validate_entries``.
"""

from __future__ import annotations

from shadowbench.predictor.discovery import recommend
from shadowbench.predictor.models import Prediction, Recommendation, RuntimeFlags
from shadowbench.predictor.validate_catalog import find_catalog_path, validate_entries

__all__ = [
    "Prediction",
    "Recommendation",
    "RuntimeFlags",
    "find_catalog_path",
    "recommend",
    "validate_entries",
]
