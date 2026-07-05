"""Tests for AMD GPU backend (profiler/gpu/amd.py)."""

from __future__ import annotations

import os
import platform
import sys
from unittest.mock import patch

from shadowbench.profiler.gpu.amd import AmdBackend


def test_amd_not_available_on_non_linux() -> None:
    with patch.object(platform, "system", return_value="Windows"):
        backend = AmdBackend()
        assert not backend.is_available()


def test_amd_not_available_no_rocm_no_drm() -> None:
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch("shadowbench.profiler.gpu.amd.shutil.which", return_value=None),
        patch.object(os.path, "isdir", return_value=False),
    ):
        backend = AmdBackend()
        assert not backend.is_available()


def test_amd_sysfs_fallback_with_device_id(tmp_path) -> None:
    drm = tmp_path / "sys" / "class" / "drm"
    card = drm / "card0"
    card.mkdir(parents=True)
    (card / "device").mkdir()
    (card / "device" / "vendor").write_text("0x1002\n")
    (card / "device" / "device").write_text("0x67df\n")
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(sys, "platform", "linux"),
        patch("shadowbench.profiler.gpu.amd.shutil.which", return_value=None),
        patch("shadowbench.profiler.gpu.amd._DRM_PATH", str(drm)),
        patch.object(os, "sysconf", side_effect=[4096, 4194304], create=True),
    ):
        backend = AmdBackend()
        assert backend.is_available()
        info = backend.probe()
        assert info.vendor == "amd"
        assert "67df" in info.name
        # page_size * n_pages / (1024*1024) = 4096 * 4194304 / 1048576 = 16384 MB
        # // 4 = 4096
        assert info.vram_total_mb == 4096


def test_amd_sysfs_fallback_no_device_id(tmp_path) -> None:
    drm = tmp_path / "sys" / "class" / "drm"
    card = drm / "card0"
    card.mkdir(parents=True)
    (card / "device").mkdir()
    (card / "device" / "vendor").write_text("0x1002\n")
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(sys, "platform", "linux"),
        patch("shadowbench.profiler.gpu.amd.shutil.which", return_value=None),
        patch("shadowbench.profiler.gpu.amd._DRM_PATH", str(drm)),
        patch.object(os, "sysconf", side_effect=[4096, 2097152], create=True),
    ):
        backend = AmdBackend()
        assert backend.is_available()
        info = backend.probe()
        assert info.vendor == "amd"
        assert "AMD GPU" in info.name
        assert info.vram_total_mb == 2048
