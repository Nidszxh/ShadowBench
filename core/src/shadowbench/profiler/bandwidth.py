"""Bounded RAM bandwidth measurement and GEMM stress kernel.

Two tiers: a quick ~0.1 s RAM read (:func:`measure_ram_bandwidth`) and a ~3 s GEMM loop
(:func:`run_bandwidth_test`).
"""

from __future__ import annotations

import time

import numpy as np

from shadowbench.common.logging import get_logger
from shadowbench.profiler.models import BandwidthResult

logger = get_logger(__name__)

#: Square matrix dimension for the GEMM stress kernel (FP32, ~48 MB working set).
_MATRIX_N = 2048
#: Size of the buffer (MB) for the system RAM bandwidth measurement.
_RAM_BW_BUFFER_MB = 512
#: Default fallback when measurement fails.
_DEFAULT_RAM_GBPS = 30.0


def measure_ram_bandwidth(buffer_mb: int = _RAM_BW_BUFFER_MB) -> float:
    """Estimate system RAM sequential-read bandwidth (GB/s) via a timed numpy sum."""
    try:
        size = buffer_mb * 1024 * 1024 // 8  # uint64 elements
        buf = np.empty(size, dtype=np.uint64)
        buf[:] = 1  # force physical page allocation
        t0 = time.perf_counter()
        _ = int(buf.sum())
        elapsed = time.perf_counter() - t0
        if elapsed <= 0:
            return _DEFAULT_RAM_GBPS
        gbps = buf.nbytes / elapsed / 1e9
        return round(gbps, 1)
    except Exception:
        logger.debug("RAM bandwidth measurement failed, using default %.1f GB/s", _DEFAULT_RAM_GBPS)
        return _DEFAULT_RAM_GBPS


def run_bandwidth_test(budget_seconds: float = 3.0) -> BandwidthResult:
    """Run a time-boxed GEMM loop returning compute (TFLOPS) and memory bandwidth (GB/s).

    Also populates ``system_ram_gbps`` via :func:`measure_ram_bandwidth`.
    """
    a = np.random.rand(_MATRIX_N, _MATRIX_N).astype(np.float32)
    b = np.random.rand(_MATRIX_N, _MATRIX_N).astype(np.float32)

    flops_per_iter = 2.0 * (_MATRIX_N**3)
    # A, B read + C write.
    bytes_per_iter = 3.0 * (_MATRIX_N**2) * np.dtype(np.float32).itemsize

    iterations = 0
    start = time.perf_counter()
    while True:
        c = a @ b
        # Touch the result so the optimizer can't elide the work.
        _ = float(c[0, 0])
        iterations += 1
        if time.perf_counter() - start >= budget_seconds:
            break
    elapsed = time.perf_counter() - start

    tflops = (flops_per_iter * iterations) / elapsed / 1e12
    gbps = (bytes_per_iter * iterations) / elapsed / 1e9
    logger.debug(
        "bandwidth: %d iters in %.2fs → %.1f TFLOPS, %.1f GB/s", iterations, elapsed, tflops, gbps
    )

    ram_gbps = measure_ram_bandwidth()

    return BandwidthResult(
        cpu_matmul_gbps=round(gbps, 2),
        device_compute_tflops=round(tflops, 3),
        duration_s=round(elapsed, 2),
        system_ram_gbps=ram_gbps,
    )
