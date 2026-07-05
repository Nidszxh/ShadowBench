"""Profiler orchestrator: assemble a complete :class:`HardwareProfile`."""

from __future__ import annotations

from shadowbench.common.logging import get_logger
from shadowbench.profiler.bandwidth import measure_ram_bandwidth, run_bandwidth_test
from shadowbench.profiler.gpu.base import detect_gpu
from shadowbench.profiler.models import BandwidthResult, HardwareProfile
from shadowbench.profiler.system import detect_system

logger = get_logger(__name__)


def profile_hardware(
    *,
    run_stress_test: bool = True,
    budget_seconds: float = 3.0,
) -> HardwareProfile:
    """Detect GPU and system memory, optionally running the bandwidth stress kernel.

    RAM bandwidth is always measured; only the GEMM loop is skipped when ``run_stress_test`` is False.
    """
    system = detect_system()
    gpu = detect_gpu()

    if run_stress_test:
        bandwidth = run_bandwidth_test(budget_seconds)
    else:
        ram_gbps = measure_ram_bandwidth()
        bandwidth = BandwidthResult(
            cpu_matmul_gbps=0.0,
            device_compute_tflops=0.0,
            duration_s=0.0,
            system_ram_gbps=ram_gbps,
        )

    profile = HardwareProfile(system=system, gpu=gpu, bandwidth=bandwidth)
    logger.info(
        "Profiled %s / %d MB RAM / GPU=%s",
        system.cpu_name,
        system.ram_total_mb,
        gpu.name if gpu else "none",
    )
    return profile
