"""Swappable, per-vendor GPU backends behind a common interface."""

from __future__ import annotations

from shadowbench.profiler.gpu.base import GpuBackend, detect_gpu

__all__ = ["GpuBackend", "detect_gpu"]
