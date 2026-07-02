"""OpenAI-compatible local proxy.  [Phase 4 — P4.3]

Exposes ``POST /v1/chat/completions`` at ``localhost:{proxy_port}`` so any OpenAI-format client works against
ShadowBench unmodified. The proxy consults the router to run locally or forward to a faster peer, then streams
tokens back with identical framing either way.
"""

from __future__ import annotations

from typing import Any


def create_app() -> Any:  # pragma: no cover - Phase 4  # returns a FastAPI app, typed in impl
    """Build the FastAPI application exposing the OpenAI-compatible endpoints."""
    raise NotImplementedError("Local proxy server lands in Phase 4 (P4.3).")
