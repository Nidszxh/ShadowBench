"""Module 2 — Predictor Engine.

Turns a :class:`~shadowbench.profiler.models.HardwareProfile` + user intent into a ranked model recommendation
with predicted tokens/sec and exact runtime flags. Branches on model topology (Dense vs. MoE) per
``DATAFLOW.md §1``.

Public entry point: :func:`~shadowbench.predictor.discovery.recommend`.
"""

from __future__ import annotations

from shadowbench.predictor.discovery import recommend
from shadowbench.predictor.models import Prediction, Recommendation, RuntimeFlags

__all__ = ["Prediction", "Recommendation", "RuntimeFlags", "recommend"]
