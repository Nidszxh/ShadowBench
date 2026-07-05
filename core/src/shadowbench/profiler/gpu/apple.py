"""Apple Silicon backend: unified memory, VRAM estimated as fraction of system RAM from system_profiler."""

from __future__ import annotations

import json
import platform
import subprocess

import psutil

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo

#: Fraction of unified memory the OS will practically let the GPU use for weights.
_UNIFIED_MEMORY_GPU_FRACTION = 0.75


class AppleBackend(GpuBackend):
    priority = 20

    def is_available(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def probe(self) -> GpuInfo:
        name = self._chip_name()
        total_ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
        usable_vram_mb = int(total_ram_mb * _UNIFIED_MEMORY_GPU_FRACTION)
        available_mb = int(psutil.virtual_memory().available // (1024 * 1024))
        return GpuInfo(
            vendor="apple",
            name=name,
            vram_total_mb=usable_vram_mb,
            vram_free_mb=min(available_mb, usable_vram_mb),
        )

    @staticmethod
    def _chip_name() -> str:
        try:
            out = subprocess.run(
                ["system_profiler", "-json", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            data = json.loads(out.stdout)
            return str(data["SPHardwareDataType"][0].get("chip_type", "Apple Silicon"))
        except Exception:
            return "Apple Silicon"
