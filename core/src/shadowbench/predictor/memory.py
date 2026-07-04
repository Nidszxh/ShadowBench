"""Memory-footprint formulas (``DATAFLOW.md §1.1–1.2``).

Pure functions: numbers in, numbers out. No hardware or I/O — so they're testable on any machine and are the
first thing pinned against the golden dataset.
"""

from __future__ import annotations

from shadowbench.common.types import KVCacheQuantization, Quantization


def dense_weight_gb(n_params_billions: float, quant: Quantization) -> float:
    """Weight-only footprint in GB (``DATAFLOW.md §1.1``).

    ``params_B × 1e9 × effective_bits / 8 / 1e9`` — no KV headroom (that is computed separately by
    :func:`kv_cache_gb`).
    """
    bytes_total = n_params_billions * 1e9 * (quant.effective_bpw / 8.0)
    return bytes_total / 1e9


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_length: int,
    kv_quant: KVCacheQuantization = KVCacheQuantization.FP16,
) -> float:
    """Precise KV-cache size in GB (``DATAFLOW.md §1.2``).

    ``M_KV = 2 × n_layers × n_kv_heads × d_head × b_precision × c_context`` (the leading 2 covers K and V).
    Using ``n_kv_heads`` (not attention heads) makes this correct for GQA/MQA models.
    """
    bpe = kv_quant.bytes_per_elem
    bytes_total = 2 * n_layers * n_kv_heads * head_dim * bpe * context_length
    return bytes_total / 1e9
