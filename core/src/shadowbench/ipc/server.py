"""JSON-RPC-over-stdio IPC server.  [Phase 3 — P3.2]

The Tauri Rust core spawns this sidecar and speaks a small, versioned command set. The CLI and the desktop app
call the *same* underlying functions — this module only marshals requests to them.

Contract (see ``schemas/ipc/ipc.schema.json``):
    profile_system()                      -> HardwareProfile
    analyze_requirement(task, profile)    -> Recommendation
    run_model(config)                     -> EngineHandle
"""

from __future__ import annotations

#: Bumped when the IPC command set changes incompatibly; the frontend negotiates against it.
IPC_PROTOCOL_VERSION = 1


def serve_stdio() -> None:  # pragma: no cover - Phase 3
    """Read JSON-RPC requests from stdin, dispatch to modules, write responses to stdout."""
    raise NotImplementedError("IPC stdio server lands in Phase 3 (P3.2).")
