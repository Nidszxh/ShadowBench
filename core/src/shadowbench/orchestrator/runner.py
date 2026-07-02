"""Local inference-engine process management.  [Phase 3 — P3.4]

Spawns and supervises llama.cpp / Ollama with the Config Coach's flags, capturing stdout/stderr so crashes
surface as clean errors in the UI rather than silent failures.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowbench.predictor.models import RuntimeFlags


@dataclass(slots=True)
class EngineHandle:
    """A running inference process."""

    pid: int
    endpoint: str  # e.g. http://localhost:8080/v1


def launch_engine(  # pragma: no cover - Phase 3
    model_path: str,
    flags: RuntimeFlags,
    *,
    port: int = 8080,
) -> EngineHandle:
    """Start the local engine with ``flags`` and return a handle once it is serving."""
    raise NotImplementedError("Engine process orchestration lands in Phase 3 (P3.4).")
