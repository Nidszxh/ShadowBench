"""Resumable, checksum-verified GGUF downloader.  [Phase 3 — P3.3]

Chunked streaming with HTTP range-resume so an interrupted download continues instead of restarting, plus
post-download checksum verification. Progress is reported back to the UI over IPC.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

#: Called with (bytes_done, bytes_total) so the Tauri UI can render a progress bar.
ProgressCallback = Callable[[int, int], None]


def download_model(  # pragma: no cover - Phase 3
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download ``url`` to ``dest`` with resume + checksum verification. Returns the final path."""
    raise NotImplementedError("Chunked resumable downloader lands in Phase 3 (P3.3).")
