"""AMD GPU backend via ROCm SMI.

Stub for Phase 1 completion — the interface is fixed so the implementation can land without touching callers.
Prefer parsing ``rocm-smi --showmeminfo vram --json`` where available; fall back to ``pyamdgpuinfo`` on Linux.
"""

from __future__ import annotations

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo


class AmdBackend(GpuBackend):
    priority = 30

    def is_available(self) -> bool:
        # TODO(Phase 1): detect a working rocm-smi / pyamdgpuinfo before claiming availability.
        return False

    def probe(self) -> GpuInfo:  # pragma: no cover - not yet implemented
        raise NotImplementedError("AMD ROCm probing lands in Phase 1 (P1.1).")
