"""GGUF header parsing.

Reads model topology straight from a ``.gguf`` file's metadata (layer count, head config, expert count) instead
of hardcoding per-model facts — the staleness trap called out in ``IMPLEMENTATION_PLAN.md`` P1.4.

Phase 1 (P1.3) implements the binary header reader (or wraps the optional ``gguf`` package). The dataclass below
is the stable contract the Predictor's memory formulas consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shadowbench.common.types import ModelTopology


@dataclass(slots=True)
class GgufMetadata:
    """Architecture facts extracted from a GGUF header."""

    name: str
    topology: ModelTopology
    n_params_billions: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    # MoE-only fields (None for dense models).
    n_experts: int | None = None
    n_experts_active: int | None = None


def read_gguf_metadata(path: str | Path) -> GgufMetadata:
    """Parse a GGUF file header into :class:`GgufMetadata`.

    Not yet implemented — Phase 1 (P1.3). Implementation notes:
      * GGUF is little-endian: magic ``GGUF``, version, tensor count, then a typed key/value metadata block.
      * ``*.expert_count`` / ``*.expert_used_count`` keys distinguish MoE from dense.
    """
    raise NotImplementedError("GGUF header parsing lands in Phase 1 (P1.3).")
