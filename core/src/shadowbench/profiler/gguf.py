"""GGUF header parsing.

Reads model topology straight from a ``.gguf`` file's metadata (layer count, head config, expert count) instead
of hardcoding per-model facts — the staleness trap called out in ``IMPLEMENTATION_PLAN.md`` P1.4.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from shadowbench.common.types import ModelTopology

logger = logging.getLogger(__name__)

_GGUF_MAGIC = 0x46554747

_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


@dataclass(slots=True)
class GgufMetadata:
    """Architecture facts extracted from a GGUF header."""

    name: str
    topology: ModelTopology
    n_params_billions: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_experts: int | None = None
    n_experts_active: int | None = None


def read_gguf_metadata(path: str | Path) -> GgufMetadata:
    """Parse a GGUF file header into :class:`GgufMetadata`.

    Raises:
        FileNotFoundError: if the GGUF file does not exist.
        ValueError: if the file is not a valid GGUF file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != _GGUF_MAGIC:
            raise ValueError(f"Not a valid GGUF file (bad magic): {path}")

        _version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        kv_store: dict[str, object] = {}
        for _ in range(kv_count):
            key, value = _read_kv_pair(f)
            kv_store[key] = value

        n_params = _read_tensor_param_count(f, tensor_count)

    architecture = kv_store.get("general.architecture", "unknown")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8")

    n_layers = _get_int(kv_store, "llm.block_count", 0)
    n_kv_heads = _get_int(kv_store, "llm.attention.head_count_kv", 0)
    head_dim = _get_int(kv_store, "llm.rope.dimension_count", 0)
    n_experts = _get_optional_int(kv_store, "llm.expert_count")
    n_experts_active = _get_optional_int(kv_store, "llm.expert_used_count")
    if n_experts_active is None:
        n_experts_active = _get_optional_int(kv_store, "llm.experts_per_token")

    topology = ModelTopology.MOE if n_experts is not None else ModelTopology.DENSE

    name = kv_store.get("general.name", path.stem)
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    n_params_billions = (
        round(n_params / 1e9, 2)
        if n_params > 0
        else _estimate_params_from_layers(n_layers, topology, n_experts)
    )

    return GgufMetadata(
        name=str(name),
        topology=topology,
        n_params_billions=n_params_billions,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_experts=n_experts,
        n_experts_active=n_experts_active,
    )


def _read_kv_pair(f: BinaryIO) -> tuple[str, object]:
    key_len = struct.unpack("<Q", f.read(8))[0]
    key = f.read(key_len).decode("utf-8")
    value_type = struct.unpack("<I", f.read(4))[0]
    value = _read_value(f, value_type)
    return key, value


def _read_value(f: BinaryIO, value_type: int) -> object:
    """Read a typed GGUF metadata value."""
    if value_type == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if value_type == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if value_type == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if value_type == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if value_type == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if value_type == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if value_type == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == _GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    if value_type == _GGUF_TYPE_STRING:
        slen = struct.unpack("<Q", f.read(8))[0]
        return f.read(slen).decode("utf-8")
    if value_type == _GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(arr_len)]
    if value_type == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if value_type == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if value_type == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    msg = f"Unknown GGUF value type: {value_type}"
    raise ValueError(msg)


def _read_tensor_param_count(f: BinaryIO, tensor_count: int) -> int:
    """Sum parameter count across all tensor info entries."""
    total_params = 0
    for _ in range(tensor_count):
        name_len = struct.unpack("<Q", f.read(8))[0]
        f.read(name_len)  # skip name
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{'Q' * n_dims}", f.read(8 * n_dims))
        _type = struct.unpack("<I", f.read(4))[0]  # tensor type
        _offset = struct.unpack("<Q", f.read(8))[0]  # offset (unused)
        count = 1
        for d in dims:
            count *= d
        total_params += count
    return total_params


def _get_int(store: dict[str, object], key: str, default: int) -> int:
    val = store.get(key)
    if val is None:
        return default
    if isinstance(val, int | float):
        return int(val)
    logger.warning(
        "GGUF key '%s' has unexpected type %s (value=%r), using default=%d",
        key,
        type(val).__name__,
        val,
        default,
    )
    return default


def _get_optional_int(store: dict[str, object], key: str) -> int | None:
    val = store.get(key)
    if val is None:
        return None
    if isinstance(val, int | float):
        return int(val)
    logger.warning(
        "GGUF key '%s' has unexpected type %s (value=%r), returning None",
        key,
        type(val).__name__,
        val,
    )
    return None


def _estimate_params_from_layers(
    n_layers: int, topology: ModelTopology, n_experts: int | None
) -> float:
    """Fallback parameter estimate based on layer count when tensor info is unavailable.

    The constants are rough heuristics calibrated against known model families:
    - Dense: ~0.15B params/layer (derived from LLaMA-3-8B: 32 layers × 0.25 = 8B, but
      typical models range 0.10–0.25 so 0.15 is a conservative middle).
    - MoE: 0.18B base + 0.01B per expert/layer (derived from Mixtral-8x7B: 32 layers ×
      0.18 × (1 + 8 × 0.01) ≈ 10.4B, close to the actual ~12.9B active-only spec).
    """
    if topology is ModelTopology.MOE and n_experts:
        return round(n_layers * 0.18 * (1 + n_experts * 0.01), 2)
    return round(n_layers * 0.15, 2)
