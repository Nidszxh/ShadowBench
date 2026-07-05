"""Intel GPU backend: detects integrated/discrete GPUs via sysfs (Linux only)."""

from __future__ import annotations

import os
import platform
from typing import NamedTuple

import psutil

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo

_INTEL_VENDOR_ID = "0x8086"
_SHARED_MEMORY_GPU_FRACTION = 0.50
_DRM_PATH = "/sys/class/drm"


class _CardEntry(NamedTuple):
    name: str | None
    device_id: str | None


class IntelBackend(GpuBackend):
    priority = 15

    def is_available(self) -> bool:
        if platform.system() != "Linux":
            return False
        if not os.path.isdir(_DRM_PATH):
            return False
        return any(_iter_intel_cards())

    def probe(self) -> GpuInfo:
        name = "Intel Graphics"
        device_id = None
        for entry in _iter_intel_cards():
            if entry.name:
                name = entry.name
            if entry.device_id:
                device_id = entry.device_id

        total_ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
        usable_vram_mb = int(total_ram_mb * _SHARED_MEMORY_GPU_FRACTION)
        available_mb = int(psutil.virtual_memory().available // (1024 * 1024))

        if device_id:
            name = f"{name} ({device_id})"

        return GpuInfo(
            vendor="intel",
            name=name,
            vram_total_mb=usable_vram_mb,
            vram_free_mb=min(available_mb, usable_vram_mb),
        )


def _iter_intel_cards() -> list[_CardEntry]:
    """Return entries for Intel DRM cards found in sysfs."""
    results: list[_CardEntry] = []
    try:
        for entry in os.listdir(_DRM_PATH):
            card_dir = os.path.join(_DRM_PATH, entry)
            vendor_path = os.path.join(card_dir, "device/vendor")
            if not entry.startswith("card") or not os.path.isfile(vendor_path):
                continue
            with open(vendor_path) as f:
                vendor = f.read().strip()
            if vendor != _INTEL_VENDOR_ID:
                continue

            name = _read_intel_name(card_dir)
            device_id = _read_intel_device_id(card_dir)
            results.append(_CardEntry(name=name, device_id=device_id))
    except OSError:
        pass
    return results


def _read_intel_name(card_dir: str) -> str | None:
    """Try to extract a human-readable GPU name from sysfs."""
    for name_path in [
        "device/label",
        "device/product_name",
        "device/product",
    ]:
        path = os.path.join(card_dir, name_path)
        try:
            with open(path) as f:
                val = f.read().strip()
            if val:
                return val
        except OSError:
            continue
    return None


def _read_intel_device_id(card_dir: str) -> str | None:
    """Read the PCI device ID (e.g. ``a78b``)."""
    path = os.path.join(card_dir, "device/device")
    try:
        with open(path) as f:
            val = f.read().strip()
        return val.removeprefix("0x") if val else None
    except OSError:
        return None
