"""Module 1 — Hardware Profiler.

Detects CPU/GPU/RAM/VRAM and runs a bounded stress kernel to measure *real* PCIe bandwidth and compute
throughput, producing a :class:`~shadowbench.profiler.models.HardwareProfile`.

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
