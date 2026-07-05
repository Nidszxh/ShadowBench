"""Detect CPU/GPU/RAM/VRAM and measure real bandwidth/compute throughput.

Public entry point: :func:`~shadowbench.profiler.detect.profile_hardware`.
"""

from __future__ import annotations

from shadowbench.profiler.detect import profile_hardware
from shadowbench.profiler.models import BandwidthResult, GpuInfo, HardwareProfile, SystemInfo

__all__ = [
    "BandwidthResult",
    "GpuInfo",
    "HardwareProfile",
    "SystemInfo",
    "profile_hardware",
]
