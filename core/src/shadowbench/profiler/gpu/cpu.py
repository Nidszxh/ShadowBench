"""CPU-only fallback.

Not a :class:`GpuBackend` — when :func:`~shadowbench.profiler.gpu.base.detect_gpu` returns ``None``, the
Predictor treats the machine as CPU-only and estimates throughput from system-memory bandwidth alone.
This module exists as the documented home for any CPU-specific capability probing (e.g. AVX-512 detection).
"""

from __future__ import annotations

import platform


def cpu_supports_avx512() -> bool:
    """Best-effort AVX-512 detection (affects CPU inference throughput).

    Placeholder — Phase 1 will read ``/proc/cpuinfo`` on Linux and ``sysctl`` on macOS/Windows equivalents.
    """
    # TODO(Phase 1): real flag detection per platform.
    return False


def cpu_label() -> str:
    return platform.processor() or platform.machine() or "Unknown CPU"
