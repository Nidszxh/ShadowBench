"""AMD GPU backend: tries rocm-smi first, then falls back to sysfs vendor detection."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys

import psutil

from shadowbench.profiler.gpu.base import GpuBackend
from shadowbench.profiler.models import GpuInfo

_AMD_VENDOR_ID = "0x1002"
_DRM_PATH = "/sys/class/drm"


class AmdBackend(GpuBackend):
    priority = 30
    _use_rocm_smi = False

    def is_available(self) -> bool:
        if platform.system() != "Linux":
            return False
        # Prefer rocm-smi
        if shutil.which("rocm-smi"):
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                data = json.loads(result.stdout)
                if isinstance(data, dict) and any(k.startswith("card") for k in data):
                    self._use_rocm_smi = True
                    return True
            except Exception:
                pass
        # Fallback to sysfs vendor check
        if os.path.isdir(_DRM_PATH):
            try:
                for entry in os.listdir(_DRM_PATH):
                    if not entry.startswith("card"):
                        continue
                    vendor_path = os.path.join(_DRM_PATH, entry, "device/vendor")
                    if os.path.isfile(vendor_path):
                        with open(vendor_path) as f:
                            if f.read().strip() == _AMD_VENDOR_ID:
                                return True
            except OSError:
                pass
        return False

    def probe(self) -> GpuInfo:
        if self._use_rocm_smi:
            return self._probe_via_rocm_smi()
        return self._probe_via_sysfs()

    @staticmethod
    def _probe_via_rocm_smi() -> GpuInfo:
        """Populate :class:`GpuInfo` via ``rocm-smi --json`` (ROCm stack)."""
        mem = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        cards = json.loads(mem.stdout)
        first_card = next(k for k in cards if k.startswith("card"))
        card = cards[first_card]
        total = int(card.get("VRAM Total Memory (MB)", card.get("VRAM Total Memory (Gb)", 0)))
        free = int(
            card.get("VRAM Total Free Memory (MB)", card.get("VRAM Total Free Memory (Gb)", 0))
        )
        # GB→MB
        if "VRAM Total Memory (Gb)" in card:
            total = int(total * 1024)
        if "VRAM Total Free Memory (Gb)" in card:
            free = int(free * 1024)

        name = first_card
        try:
            info = subprocess.run(
                ["rocm-smi", "--showproductinfo", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            info_data = json.loads(info.stdout)
            if first_card in info_data:
                name = str(info_data[first_card].get("Card series", name))
        except Exception:
            pass

        return GpuInfo(
            vendor="amd",
            name=name,
            vram_total_mb=total,
            vram_free_mb=free,
        )

    @staticmethod
    def _probe_via_sysfs() -> GpuInfo:
        """Populate :class:`GpuInfo` from sysfs (fallback, VRAM estimate only)."""
        name = "AMD GPU"
        for entry in os.listdir(_DRM_PATH):
            if not entry.startswith("card"):
                continue
            vendor_path = os.path.join(_DRM_PATH, entry, "device/vendor")
            if not os.path.isfile(vendor_path):
                continue
            with open(vendor_path) as f:
                if f.read().strip() != _AMD_VENDOR_ID:
                    continue
            device_id = None
            dev_path = os.path.join(_DRM_PATH, entry, "device/device")
            if os.path.isfile(dev_path):
                with open(dev_path) as f:
                    device_id = f.read().strip().removeprefix("0x")
            if device_id:
                name = f"AMD GPU ({device_id})"
            # sysfs fallback: estimate VRAM from system RAM (Unix-only)
            if sys.platform != "win32":
                total_ram_mb = int(
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
                )
            else:
                total_ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
            break
        return GpuInfo(
            vendor="amd",
            name=name,
            vram_total_mb=total_ram_mb // 4,
            vram_free_mb=total_ram_mb // 4,
        )
