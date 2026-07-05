"""Tests for Intel GPU backend (profiler/gpu/intel.py)."""

from __future__ import annotations

import os
import platform
from unittest.mock import MagicMock, patch

from shadowbench.profiler.gpu.intel import IntelBackend, _iter_intel_cards


def test_intel_not_available_on_non_linux() -> None:
    with patch.object(platform, "system", return_value="Windows"):
        backend = IntelBackend()
        assert not backend.is_available()


def test_intel_not_available_without_drm() -> None:
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(os.path, "isdir", return_value=False),
    ):
        backend = IntelBackend()
        assert not backend.is_available()


def test_intel_not_available_no_intel_cards() -> None:
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(os.path, "isdir", return_value=True),
        patch("shadowbench.profiler.gpu.intel._iter_intel_cards", return_value=[]),
    ):
        backend = IntelBackend()
        assert not backend.is_available()


def test_intel_iter_cards_os_error() -> None:
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(os, "listdir", side_effect=OSError("permission denied")),
    ):
        assert _iter_intel_cards() == []


def test_intel_iter_cards_empty_drm() -> None:
    with (
        patch.object(os.path, "isdir", return_value=True),
        patch.object(os, "listdir", return_value=["card0"]),
        patch.object(os.path, "isfile", return_value=False),
    ):
        cards = _iter_intel_cards()
        assert cards == []


def test_intel_probe_no_device_id(monkeypatch) -> None:
    monkeypatch.setattr("shadowbench.profiler.gpu.intel._iter_intel_cards", lambda: [])
    monkeypatch.setattr(
        "shadowbench.profiler.gpu.intel.psutil.virtual_memory",
        lambda: MagicMock(total=16 * 1024**3, available=8 * 1024**3),
    )
    backend = IntelBackend()
    info = backend.probe()
    assert info.vendor == "intel"
    assert info.vram_total_mb == 16 * 1024 // 2
    assert info.vram_free_mb == min(8 * 1024, 16 * 1024 // 2)
