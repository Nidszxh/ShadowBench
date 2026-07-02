"""NVIDIA GPU backend via NVML (``nvidia-ml-py`` → ``pynvml``).

Install with the ``nvidia`` extra: ``uv sync --extra nvidia``.
"""

from __future__ import annotations

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo


class NvidiaBackend(GpuBackend):
    priority = 10

    def is_available(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            available = pynvml.nvmlDeviceGetCount() > 0
            pynvml.nvmlShutdown()
            return bool(available)
        except Exception:
            return False

    def probe(self) -> GpuInfo:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            return GpuInfo(
                vendor="nvidia",
                name=name,
                vram_total_mb=int(mem.total // (1024 * 1024)),
                vram_free_mb=int(mem.free // (1024 * 1024)),
                temperature_c=temp,
                driver_version=driver,
            )
        finally:
            pynvml.nvmlShutdown()
