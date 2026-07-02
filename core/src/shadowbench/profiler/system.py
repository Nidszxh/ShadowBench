"""CPU / system-memory detection via psutil."""

from __future__ import annotations

import platform

import psutil

from shadowbench.profiler.gpu.cpu import cpu_label
from shadowbench.profiler.models import SystemInfo


def detect_system() -> SystemInfo:
    """Return a snapshot of CPU cores and system memory."""
    vm = psutil.virtual_memory()
    return SystemInfo(
        cpu_name=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_mb=int(vm.total // (1024 * 1024)),
        ram_available_mb=int(vm.available // (1024 * 1024)),
    )


def _cpu_name() -> str:
    # platform.processor() is often empty on Linux; fall back to the machine arch label.
    return platform.processor() or cpu_label()
