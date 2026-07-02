"""Logging setup.

A single :func:`configure_logging` entry point wires up Rich-formatted logs for the CLI. The IPC server
(Phase 3) will swap the handler for structured JSON so the Tauri frontend can parse log lines.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (e.g. ``get_logger(__name__)``)."""
    return logging.getLogger(name)
