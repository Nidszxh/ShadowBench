"""Pool security: local TLS certs, peer pairing, and request sandboxing.  [Phase 4 — P4.2/P4.5]

Enforces the ``SECURITY.md`` threat model. The pairing step (PIN/QR confirmation) is deliberately *not*
silent trust-on-first-use — a peer must be explicitly accepted before it can submit work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NodeIdentity:
    """This node's self-signed TLS identity, generated at startup."""

    node_id: str
    cert_pem: bytes
    key_pem: bytes
    fingerprint: str


def generate_identity() -> NodeIdentity:  # pragma: no cover - Phase 4
    """Generate an ad-hoc self-signed TLS identity (cryptography lib)."""
    raise NotImplementedError("TLS identity generation lands in Phase 4 (P4.2).")


def sanitize_request(payload: dict[str, object]) -> dict[str, object]:
    """Strip an inbound peer request down to prompt-only fields.

    Providing nodes must expose **no** filesystem/env/network access to a request. This allowlist is the
    enforcement point and is fuzz-tested before P2P ships (P4.5).
    """
    allowed = {"model", "messages", "prompt", "temperature", "top_p", "max_tokens", "stream"}
    return {k: v for k, v in payload.items() if k in allowed}
