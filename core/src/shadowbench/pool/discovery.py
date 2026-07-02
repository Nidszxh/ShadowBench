"""mDNS/Zeroconf peer discovery.  [Phase 4 — P4.1]

Advertises this node's anonymized hardware summary + available models on the LAN and maintains a peer table
with TTL-based expiry so stale/disconnected peers drop out automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Peer:
    """A discovered pool member."""

    node_id: str
    host: str
    port: int
    gpu_name: str | None
    vram_total_mb: int
    available_models: list[str] = field(default_factory=list)
    #: Advertised compute weight; lowered when a provider throttles (ARCHITECTURE.md §5).
    compute_weight: float = 1.0


class PeerRegistry:
    """In-memory peer table populated by the mDNS listener. Thread-safe in the real impl."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._peers: dict[str, Peer] = {}

    def upsert(self, peer: Peer) -> None:
        self._peers[peer.node_id] = peer

    def active(self) -> list[Peer]:
        return list(self._peers.values())


def start_discovery(registry: PeerRegistry) -> None:  # pragma: no cover - Phase 4
    """Begin advertising this node and listening for peers via Zeroconf."""
    raise NotImplementedError("mDNS discovery lands in Phase 4 (P4.1).")
