"""Routing + failover: decide local vs. peer execution.  [Phase 4 — P4.3 / Phase 5 — P5.2]

Compares estimated local completion time against each peer (using the Predictor + advertised peer state) and
routes to the fastest. On a mid-stream peer drop, hot-swaps to the next-best peer without surfacing an
exception to the caller.
"""

from __future__ import annotations

from shadowbench.pool.discovery import Peer, PeerRegistry


def choose_target(  # pragma: no cover - Phase 4
    model_id: str,
    local_tps: float,
    registry: PeerRegistry,
) -> Peer | None:
    """Return the peer that should run the request, or ``None`` to execute locally."""
    raise NotImplementedError("Routing decision lands in Phase 4 (P4.3).")
