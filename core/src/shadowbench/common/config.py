"""Application settings and on-disk paths.

Uses :mod:`platformdirs` so data/cache locations are correct on Windows, macOS, and Linux. Every field can be
overridden with a ``SHADOWBENCH_*`` environment variable, which keeps the sidecar configurable by the Tauri
host without a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

_APP_NAME = "ShadowBench"


def _default_data_dir() -> Path:
    return Path(os.environ.get("SHADOWBENCH_DATA_DIR", user_data_dir(_APP_NAME)))


@dataclass(slots=True)
class Settings:
    """Runtime configuration resolved from environment + platform defaults."""

    data_dir: Path = field(default_factory=_default_data_dir)
    #: Duration budget for the PCIe/compute stress kernel. Hard-capped for "non-invasive" profiling.
    bandwidth_test_seconds: float = 3.0
    #: Local proxy bind port for the OpenAI-compatible endpoint (Phase 4).
    proxy_port: int = 8080

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "shadowbench.sqlite"

    def ensure_dirs(self) -> None:
        """Create the data directories if they don't yet exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a :class:`Settings` from the environment."""
    settings = Settings()
    if seconds := os.environ.get("SHADOWBENCH_BANDWIDTH_SECONDS"):
        settings.bandwidth_test_seconds = float(seconds)
    if port := os.environ.get("SHADOWBENCH_PROXY_PORT"):
        settings.proxy_port = int(port)
    return settings
