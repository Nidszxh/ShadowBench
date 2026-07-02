"""Memory formulas pinned against hand-computed reference values (DATAFLOW §1.1–1.2)."""

from __future__ import annotations

import pytest

from shadowbench.common.types import Quantization
from shadowbench.predictor import memory


def test_dense_weight_gb_q8_8b() -> None:
    # 8B @ Q8_0 (8.5 bpw): 8e9 * 8.5/8 bytes = 8.5 GB, * 1.2 headroom = 10.2 GB.
    gb = memory.dense_weight_gb(8.0, Quantization.Q8_0)
    assert gb == pytest.approx(10.2, rel=1e-3)


def test_dense_weight_scales_with_quant() -> None:
    q4 = memory.dense_weight_gb(8.0, Quantization.Q4_K_M)
    q8 = memory.dense_weight_gb(8.0, Quantization.Q8_0)
    assert q4 < q8


def test_kv_cache_gb_reference() -> None:
    # 2 * 32 layers * 8 kv-heads * 128 head-dim * 2 bytes * 4096 ctx = 536,870,912 bytes ≈ 0.537 GB.
    gb = memory.kv_cache_gb(n_layers=32, n_kv_heads=8, head_dim=128, context_length=4096)
    assert gb == pytest.approx(0.5369, rel=1e-3)


def test_kv_cache_scales_linearly_with_context() -> None:
    base = memory.kv_cache_gb(32, 8, 128, 4096)
    double = memory.kv_cache_gb(32, 8, 128, 8192)
    assert double == pytest.approx(2 * base, rel=1e-6)
