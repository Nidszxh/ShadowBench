"""NVIDIA GPU backend: tries pynvml first, then falls back to nvidia-smi."""

from __future__ import annotations

import shutil
import subprocess

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo


class NvidiaBackend(GpuBackend):
    priority = 10
    _initialized = False
    _use_nvidia_smi = False

    def is_available(self) -> bool:
        if self._initialized:
            return True
        # Try pynvml first (highest fidelity)
        try:
            import pynvml

            pynvml.nvmlInit()
            available = pynvml.nvmlDeviceGetCount() > 0
            if available:
                self._initialized = True
                self._use_nvidia_smi = False
            else:
                pynvml.nvmlShutdown()
            return bool(available)
        except Exception:
            pass
        # Fallback to nvidia-smi (ships with driver)
        if not shutil.which("nvidia-smi"):
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            count = int(result.stdout.strip())
            if count > 0:
                self._initialized = True
                self._use_nvidia_smi = True
                return True
        except Exception:
            pass
        return False

    def probe(self) -> GpuInfo:
        if self._use_nvidia_smi:
            return self._probe_via_smi()
        return self._probe_via_nvml()

    def _probe_via_nvml(self) -> GpuInfo:
        """Populate :class:`GpuInfo` via the NVML library (highest-fidelity path)."""
        import pynvml

        if not self._initialized:
            pynvml.nvmlInit()
            self._initialized = True
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

    @staticmethod
    def _probe_via_smi() -> GpuInfo:
        """Populate :class:`GpuInfo` via ``nvidia-smi`` CLI (zero extra dependencies)."""
        csv = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        parts = [p.strip() for p in csv.split(", ")]
        if len(parts) >= 3:
            name = parts[0]
            total = _parse_smi_mb(parts[1])
            free = _parse_smi_mb(parts[2])
        else:
            name = "NVIDIA GPU"
            total = 0
            free = 0
        driver = parts[3] if len(parts) >= 4 else None

        temp = None
        try:
            t_out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            temp = float(t_out.stdout.strip())
        except Exception:
            pass

        return GpuInfo(
            vendor="nvidia",
            name=name,
            vram_total_mb=total,
            vram_free_mb=free,
            temperature_c=temp,
            driver_version=driver,
        )


def _parse_smi_mb(value: str) -> int:
    """Parse e.g. ``"8188 MiB"`` → 8188."""
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return 0
