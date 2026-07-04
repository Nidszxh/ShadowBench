"""Bounded compute / memory-bandwidth stress kernel.

Measures *observed* throughput rather than trusting spec sheets — the key input to the VRAM-spillover
prediction in ``DATAFLOW.md §1.3``. The kernel is hard-capped in wall-clock time so profiling stays
"non-invasive" (see ``Settings.bandwidth_test_seconds``).

Phase 1 (P1.2) will add a true host↔device transfer path (CUDA/Metal) to measure real PCIe GB/s. This CPU/numpy
implementation is the portable baseline and establishes the timing-budget contract that path must honor.
"""

from __future__ import annotations

import time

import numpy as np

from shadowbench.common.logging import get_logger
from shadowbench.profiler.models import BandwidthResult

logger = get_logger(__name__)

#: Square matrix dimension for the GEMM stress kernel. FP32 → ~0.5 GB working set at N=4096.
_MATRIX_N = 2048
#: Size of the buffer (MB) for the system RAM bandwidth measurement.
_RAM_BW_BUFFER_MB = 512
#: Default fallback when measurement fails.
_DEFAULT_RAM_GBPS = 30.0


def _measure_ram_bandwidth(buffer_mb: int = _RAM_BW_BUFFER_MB) -> float:
    """Estimate system RAM sequential-read bandwidth (GB/s) via timed memcpy.

    Uses ``memoryview`` cast to ``uint64`` and a simple sum, which compiles down to a tight
    SIMD-optimised loop in numpy.  This is a good proxy for the CPU-side read pattern during
    expert-offload inference.
    """
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
    """Run a time-boxed GEMM loop and derive compute (TFLOPS) + memory bandwidth (GB/s).

    Args:
        budget_seconds: Hard wall-clock cap. The loop stops after the first iteration that crosses it, so the
            total runtime is bounded regardless of machine speed.
    """
    a = np.random.rand(_MATRIX_N, _MATRIX_N).astype(np.float32)
    b = np.random.rand(_MATRIX_N, _MATRIX_N).astype(np.float32)

    # One matmul = 2 * N^3 floating-point ops.
    flops_per_iter = 2.0 * (_MATRIX_N**3)
    # Bytes moved per matmul: read A, read B, write C.
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

    ram_gbps = _measure_ram_bandwidth()

    return BandwidthResult(
        cpu_matmul_gbps=round(gbps, 2),
        device_compute_tflops=round(tflops, 3),
        duration_s=round(elapsed, 2),
        system_ram_gbps=ram_gbps,
    )
