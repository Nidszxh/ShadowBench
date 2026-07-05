"""GPU backend interface with priority-ordered auto-detection (returns None on CPU-only machines)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from shadowbench.common.logging import get_logger
from shadowbench.profiler.models import GpuInfo

logger = get_logger(__name__)


class GpuBackend(ABC):
    """Abstract per-vendor GPU accessor."""

    #: Lower runs first. NVIDIA/Intel/Apple/AMD before the CPU fallback.
    priority: int = 100

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can talk to hardware on the current machine."""

    @abstractmethod
    def probe(self) -> GpuInfo:
        """Return a populated :class:`GpuInfo`."""


def _ordered_backends() -> list[GpuBackend]:
    # Imported lazily so a missing optional dep (e.g. pynvml) never breaks import of the package.
    from shadowbench.profiler.gpu.amd import AmdBackend
    from shadowbench.profiler.gpu.apple import AppleBackend
    from shadowbench.profiler.gpu.intel import IntelBackend
    from shadowbench.profiler.gpu.nvidia import NvidiaBackend

    backends: list[GpuBackend] = [
        NvidiaBackend(),
        IntelBackend(),
        AppleBackend(),
        AmdBackend(),
    ]
    return sorted(backends, key=lambda b: b.priority)


def detect_gpu() -> GpuInfo | None:
    """Return the first available GPU, or ``None`` for a CPU-only machine."""
    for backend in _ordered_backends():
        try:
            if backend.is_available():
                return backend.probe()
        except Exception:
            logger.debug(
                "GPU backend %s failed, trying next", type(backend).__name__, exc_info=True
            )
    logger.info("No GPU backend available; using CPU-only profile")
    return None
