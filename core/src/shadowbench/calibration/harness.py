"""Ground-truth benchmark harness.  [Phase 2 — P2.4]

Wraps ``llama-bench`` (or an Ollama run) to measure *real* tokens/sec, so every run can contribute a data
point to ``datasets/golden.jsonl``. This measured value is the calibration target for the Predictor formulas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GroundTruth:
    model_id: str
    quantization: str
    context_length: int
    measured_tps: float


def measure_tps(  # pragma: no cover - Phase 2
    model_path: str,
    *,
    context_length: int = 4096,
) -> GroundTruth:
    """Run a short real benchmark and return measured tokens/sec."""
    raise NotImplementedError("llama-bench harness lands in Phase 2 (P2.4).")
