"""CPU-specific capability probes (AVX-512, etc.) when no GPU is detected."""

from __future__ import annotations

import platform


def cpu_supports_avx512() -> bool:
    """Best-effort AVX-512 detection (placeholder; always returns False)."""
    # TODO(Phase 1): real flag detection per platform.
    return False


def cpu_label() -> str:
    return platform.processor() or platform.machine() or "Unknown CPU"
