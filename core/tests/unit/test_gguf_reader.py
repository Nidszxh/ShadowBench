"""GGUF header parser tests.

Uses synthetic binary GGUF data to validate parsing without requiring real model files.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from shadowbench.common.types import ModelTopology
from shadowbench.profiler.gguf import (
    _GGUF_TYPE_BOOL,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT8,
    _GGUF_TYPE_UINT32,
    _GGUF_TYPE_UINT64,
    read_gguf_metadata,
)


def _build_gguf(
    *,
    version: int = 3,
    tensor_count: int = 2,
    kv_pairs: list[tuple[str, int, object]] | None = None,
    tensor_dims: list[list[int]] | None = None,
) -> bytes:
    """Build a minimal valid GGUF binary blob.

    Args:
        version: GGUF format version.
        tensor_count: Number of tensor info entries.
        kv_pairs: List of (key, value_type, value) tuples.
        tensor_dims: List of dimension lists, one per tensor.
    """
    buf = bytearray()
    # Header
    buf += struct.pack("<I", 0x46554747)  # magic
    buf += struct.pack("<I", version)  # version
    buf += struct.pack("<Q", tensor_count)  # tensor count
    # KV count placeholder
    kv_count = len(kv_pairs) if kv_pairs else 0
    buf += struct.pack("<Q", kv_count)
    # KV pairs
    for key, vtype, value in kv_pairs or []:
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes))
        buf += key_bytes
        buf += struct.pack("<I", vtype)
        buf += _pack_value(vtype, value)
    # Tensor info (minimal placeholder)
    dims_list = tensor_dims or [[4, 4], [4]]
    for dims in dims_list[:tensor_count]:
        name = b"test_tensor"
        buf += struct.pack("<Q", len(name))
        buf += name
        buf += struct.pack("<I", len(dims))  # n_dims
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<I", 0)  # type (float32)
        buf += struct.pack("<Q", 0)  # offset
    return bytes(buf)


def _pack_value(vtype: int, value: object) -> bytes:
    """Pack a GGUF value by type."""
    if vtype == _GGUF_TYPE_UINT32:
        return struct.pack("<I", value)
    if vtype == _GGUF_TYPE_STRING:
        val = value.encode("utf-8") if isinstance(value, str) else value
        return struct.pack("<Q", len(val)) + val
    if vtype == _GGUF_TYPE_FLOAT32:
        return struct.pack("<f", value)
    if vtype == _GGUF_TYPE_BOOL:
        return struct.pack("<?", value)
    if vtype == _GGUF_TYPE_UINT64:
        return struct.pack("<Q", value)
    if vtype == _GGUF_TYPE_UINT8:
        return struct.pack("<B", value)
    msg = f"Unsupported type in test builder: {vtype}"
    raise ValueError(msg)


def _write_gguf_file(
    tmp_path: Path,
    *,
    version: int = 3,
    tensor_count: int = 2,
    kv_pairs: list[tuple[str, int, object]] | None = None,
    tensor_dims: list[list[int]] | None = None,
    name: str = "test.gguf",
) -> Path:
    path = tmp_path / name
    data = _build_gguf(
        version=version,
        tensor_count=tensor_count,
        kv_pairs=kv_pairs,
        tensor_dims=tensor_dims,
    )
    path.write_bytes(data)
    return path


def test_read_dense_gguf(tmp_path: Path) -> None:
    path = _write_gguf_file(
        tmp_path,
        tensor_count=3,
        kv_pairs=[
            ("general.architecture", _GGUF_TYPE_STRING, "llama"),
            ("general.name", _GGUF_TYPE_STRING, "Test Model"),
            ("llm.block_count", _GGUF_TYPE_UINT32, 32),
            ("llm.attention.head_count_kv", _GGUF_TYPE_UINT32, 8),
            ("llm.rope.dimension_count", _GGUF_TYPE_UINT32, 128),
        ],
        tensor_dims=[[4096, 4096], [4096, 4096], [4096]],
    )
    meta = read_gguf_metadata(path)
    assert meta.name == "Test Model"
    assert meta.topology == ModelTopology.DENSE
    assert meta.n_layers == 32
    assert meta.n_kv_heads == 8
    assert meta.head_dim == 128
    assert meta.n_experts is None
    assert meta.n_experts_active is None
    # tensor_count=3 with dims [4096x4096, 4096x4096, 4096] = 16,777,216 + 16,777,216 + 4096 = ~33.6M params
    # But we don't round to billions (under 1B), so it stays at a small float
    assert meta.n_params_billions > 0


def test_read_moe_gguf(tmp_path: Path) -> None:
    path = _write_gguf_file(
        tmp_path,
        tensor_count=2,
        kv_pairs=[
            ("general.architecture", _GGUF_TYPE_STRING, "deepseek2"),
            ("general.name", _GGUF_TYPE_STRING, "MoE Model"),
            ("llm.block_count", _GGUF_TYPE_UINT32, 48),
            ("llm.attention.head_count_kv", _GGUF_TYPE_UINT32, 8),
            ("llm.rope.dimension_count", _GGUF_TYPE_UINT32, 128),
            ("llm.expert_count", _GGUF_TYPE_UINT32, 64),
            ("llm.expert_used_count", _GGUF_TYPE_UINT32, 6),
        ],
        tensor_dims=[[1024, 1024], [1024]],
    )
    meta = read_gguf_metadata(path)
    assert meta.name == "MoE Model"
    assert meta.topology == ModelTopology.MOE
    assert meta.n_experts == 64
    assert meta.n_experts_active == 6
    assert meta.n_layers == 48


def test_read_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.gguf"
    with pytest.raises(FileNotFoundError):
        read_gguf_metadata(missing)


def test_read_bad_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTG")
    with pytest.raises(ValueError, match="bad magic"):
        read_gguf_metadata(path)


def test_fallback_name_from_stem(tmp_path: Path) -> None:
    path = _write_gguf_file(
        tmp_path,
        tensor_count=1,
        kv_pairs=[
            ("general.architecture", _GGUF_TYPE_STRING, "llama"),
            ("llm.block_count", _GGUF_TYPE_UINT32, 16),
            ("llm.attention.head_count_kv", _GGUF_TYPE_UINT32, 4),
            ("llm.rope.dimension_count", _GGUF_TYPE_UINT32, 64),
        ],
        tensor_dims=[[1024, 1024]],
        name="Meta-Llama-3-8B.gguf",
    )
    meta = read_gguf_metadata(path)
    assert meta.name == "Meta-Llama-3-8B"
    assert meta.topology == ModelTopology.DENSE


def test_uint32_metadata_values(tmp_path: Path) -> None:
    path = _write_gguf_file(
        tmp_path,
        tensor_count=1,
        kv_pairs=[
            ("general.architecture", _GGUF_TYPE_STRING, "llama"),
            ("llm.block_count", _GGUF_TYPE_UINT64, 64),  # uint64
            ("llm.attention.head_count_kv", _GGUF_TYPE_UINT32, 16),  # uint32
            ("llm.rope.dimension_count", _GGUF_TYPE_UINT32, 256),
        ],
        tensor_dims=[[256, 256]],
        name="big_model.gguf",
    )
    meta = read_gguf_metadata(path)
    assert meta.n_layers == 64
    assert meta.n_kv_heads == 16
    assert meta.head_dim == 256


def test_float32_metadata_value(tmp_path: Path) -> None:
    path = _write_gguf_file(
        tmp_path,
        tensor_count=1,
        kv_pairs=[
            ("general.architecture", _GGUF_TYPE_STRING, "llama"),
            ("general.file_type", _GGUF_TYPE_FLOAT32, 7.0),
            ("llm.block_count", _GGUF_TYPE_UINT32, 8),
            ("llm.attention.head_count_kv", _GGUF_TYPE_UINT32, 4),
            ("llm.rope.dimension_count", _GGUF_TYPE_UINT32, 64),
        ],
        tensor_dims=[[64, 64]],
        name="float_test.gguf",
    )
    meta = read_gguf_metadata(path)
    assert meta.topology == ModelTopology.DENSE
