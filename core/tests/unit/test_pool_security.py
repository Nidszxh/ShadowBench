"""Security-critical: the pool request allowlist and peer registry (SECURITY.md)."""

from __future__ import annotations

from shadowbench.pool.discovery import Peer, PeerRegistry
from shadowbench.pool.security import sanitize_request


def test_sanitize_strips_non_prompt_fields() -> None:
    dangerous = {
        "model": "llama",
        "messages": [{"role": "user", "content": "hi"}],
        "system_command": "rm -rf /",  # must be dropped
        "file_path": "/etc/passwd",  # must be dropped
        "env": {"SECRET": "x"},  # must be dropped
    }
    clean = sanitize_request(dangerous)
    assert set(clean) == {"model", "messages"}


def test_sanitize_keeps_only_allowlisted_generation_params() -> None:
    payload = {"prompt": "hi", "temperature": 0.7, "max_tokens": 128, "stream": True, "evil": 1}
    clean = sanitize_request(payload)
    assert "evil" not in clean
    assert clean["temperature"] == 0.7


def test_peer_registry_upsert_and_active() -> None:
    registry = PeerRegistry()
    peer = Peer(node_id="a", host="192.168.1.5", port=9000, gpu_name="RTX 4060", vram_total_mb=8192)
    registry.upsert(peer)
    registry.upsert(peer)  # idempotent by node_id
    assert len(registry.active()) == 1
