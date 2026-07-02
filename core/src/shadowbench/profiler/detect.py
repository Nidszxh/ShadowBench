"""Profiler orchestrator: assemble a complete :class:`HardwareProfile`."""

from __future__ import annotations

from shadowbench.common.logging import get_logger
from shadowbench.profiler.bandwidth import run_bandwidth_test
from shadowbench.profiler.gpu.base import detect_gpu
from shadowbench.profiler.models import HardwareProfile
from shadowbench.profiler.system import detect_system

logger = get_logger(__name__)


def profile_hardware(
    *,
    run_stress_test: bool = True,
    budget_seconds: float = 3.0,
) -> HardwareProfile:
    """Detect GPU + system memory and (optionally) run the bandwidth stress kernel.

    Args:
        run_stress_test: When False, skip the timed GEMM loop (useful for fast, side-effect-free calls).
        budget_seconds: Wall-clock cap passed to the bandwidth kernel.
    """
    system = detect_system()
    gpu = detect_gpu()
    bandwidth = run_bandwidth_test(budget_seconds) if run_stress_test else None

    profile = HardwareProfile(system=system, gpu=gpu, bandwidth=bandwidth)
    logger.info(
        "Profiled %s / %d MB RAM / GPU=%s",
        system.cpu_name,
        system.ram_total_mb,
        gpu.name if gpu else "none",
    )
    return profile
