"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from shadowbench.profiler.models import BandwidthResult, GpuInfo, HardwareProfile, SystemInfo

ProfileBuilder = Callable[..., HardwareProfile]


@pytest.fixture
def make_profile() -> ProfileBuilder:
    """Factory for a HardwareProfile with sensible defaults, overridable per test."""

    def _build(
        *,
        vram_mb: int = 24576,
        ram_mb: int = 32768,
        pcie_gbps: float = 12.0,
        with_gpu: bool = True,
    ) -> HardwareProfile:
        gpu = (
            GpuInfo(
                vendor="nvidia",
                name="Test GPU",
                vram_total_mb=vram_mb,
                vram_free_mb=vram_mb,
            )
            if with_gpu
            else None
        )
        return HardwareProfile(
            system=SystemInfo(
                cpu_name="Test CPU",
                physical_cores=8,
                logical_cores=16,
                ram_total_mb=ram_mb,
                ram_available_mb=ram_mb,
            ),
            gpu=gpu,
            bandwidth=BandwidthResult(
                host_to_device_gbps=pcie_gbps,
                device_compute_tflops=20.0,
                duration_s=3.0,
            ),
        )

    return _build
