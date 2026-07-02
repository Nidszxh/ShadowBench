"""ShadowBench core sidecar.

Crowd-sourced, peer-to-peer benchmarking and local inference pooling for open-source AI models.
See ``PROJECT_STRUCTURE.md`` for the package layout.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION", "__version__"]

#: Application version. Kept in sync with ``pyproject.toml`` at release time.
__version__ = "0.0.1"

#: Shadow Pool wire-protocol version, tracked independently of ``__version__``.
#: Peers on mismatched protocol versions must fail discovery gracefully (never crash).
PROTOCOL_VERSION = 1
