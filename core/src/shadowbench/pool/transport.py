"""TLS WebSocket transport bridge.  [Phase 4 — P4.4]

Streams tokens between nodes using consistent chunked-JSON framing regardless of whether the executing engine
is local or a remote peer. A QUIC/WebRTC datagram fallback is deferred (see MILESTONES Phase 4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator


async def stream_from_peer(  # pragma: no cover - Phase 4
    host: str,
    port: int,
    payload: dict[str, object],
) -> AsyncIterator[bytes]:
    """Open a ``wss://`` channel to a peer and yield token frames as they arrive."""
    raise NotImplementedError("Peer streaming transport lands in Phase 4 (P4.4).")
    yield b""  # pragma: no cover - makes this an async generator for type checkers
