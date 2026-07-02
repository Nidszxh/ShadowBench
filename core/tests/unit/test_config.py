"""Settings resolution + platform paths."""

from __future__ import annotations

from pathlib import Path

from shadowbench.common.config import Settings, load_settings


def test_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SHADOWBENCH_BANDWIDTH_SECONDS", "1.5")
    monkeypatch.setenv("SHADOWBENCH_PROXY_PORT", "9999")
    settings = load_settings()
    assert settings.bandwidth_test_seconds == 1.5
    assert settings.proxy_port == 9999


def test_derived_paths(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.models_dir == tmp_path / "models"
    assert settings.db_path == tmp_path / "shadowbench.sqlite"
    settings.ensure_dirs()
    assert settings.models_dir.is_dir()
