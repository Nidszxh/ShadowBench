"""Memory-footprint formulas (``DATAFLOW.md §1.1–1.2``). Pure functions — no hardware or I/O."""

from __future__ import annotations

from shadowbench.common.types import KVCacheQuantization, Quantization


def dense_weight_gb(n_params_billions: float, quant: Quantization) -> float:
    """Weight-only footprint in GB (``DATAFLOW.md §1.1``)."""
    bytes_total = n_params_billions * 1e9 * (quant.effective_bpw / 8.0)
    return bytes_total / 1e9


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_length: int,
    kv_quant: KVCacheQuantization = KVCacheQuantization.FP16,
) -> float:
    """KV-cache size in GB (``DATAFLOW.md §1.2``). Uses ``n_kv_heads`` for correct GQA/MQA support."""
    bpe = kv_quant.bytes_per_elem
    bytes_total = 2 * n_layers * n_kv_heads * head_dim * bpe * context_length
    return bytes_total / 1e9
