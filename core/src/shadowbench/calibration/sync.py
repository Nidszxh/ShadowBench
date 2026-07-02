"""Opt-in, privacy-verified telemetry sync.  [Phase 5 — P5.3]

Batches anonymized benchmark rows and uploads them to the public dataset **only** when the user has explicitly
opted in (default-off). A test asserts payloads carry no hostname/IP/user identifier before any upload — see
``HardwareProfile.anonymized``.
"""

from __future__ import annotations

from shadowbench.profiler.models import HardwareProfile

#: Fields that must never appear in a sync payload.
FORBIDDEN_FIELDS = frozenset({"hostname", "ip", "ip_address", "mac", "user", "username", "email"})


def build_sync_payload(
    profile: HardwareProfile, runs: list[dict[str, object]]
) -> dict[str, object]:
    """Assemble a PII-free payload from an anonymized profile + benchmark rows."""
    payload: dict[str, object] = {"hardware": profile.anonymized(), "runs": runs}
    _assert_no_pii(payload)
    return payload


def _assert_no_pii(payload: dict[str, object]) -> None:
    """Defense-in-depth: refuse to build a payload containing forbidden keys at any depth."""

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in FORBIDDEN_FIELDS:
                    raise ValueError(f"Refusing to sync payload containing PII field '{key}'")
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)


def upload(payload: dict[str, object]) -> None:  # pragma: no cover - Phase 5
    """Upload an already-verified payload to the public dataset endpoint."""
    raise NotImplementedError("Telemetry upload lands in Phase 5 (P5.3).")
