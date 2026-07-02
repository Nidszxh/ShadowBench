"""Memory-footprint formulas (``DATAFLOW.md §1.1–1.2``).

Pure functions: numbers in, numbers out. No hardware or I/O — so they're testable on any machine and are the
first thing pinned against the golden dataset.
"""

from __future__ import annotations

from shadowbench.common.types import Quantization

#: KV-cache overhead multiplier on raw weight size (context-window headroom), DATAFLOW §1.1.
_KV_HEADROOM = 1.2
#: Bytes per stored KV element at FP16.
_KV_BYTES_FP16 = 2


def dense_weight_gb(n_params_billions: float, quant: Quantization) -> float:
    """First-order weight footprint in GB (``DATAFLOW.md §1.1``).

    ``(params_B × effective_bits / 8) × 1.2`` — the 1.2 reserves headroom for KV/context overhead.
    """
    bytes_total = n_params_billions * 1e9 * (quant.effective_bpw / 8.0)
    return bytes_total / 1e9 * _KV_HEADROOM


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_length: int,
    bytes_per_elem: int = _KV_BYTES_FP16,
) -> float:
    """Precise KV-cache size in GB (``DATAFLOW.md §1.2``).

    ``M_KV = 2 × n_layers × n_kv_heads × d_head × b_precision × c_context`` (the leading 2 covers K and V).
    Using ``n_kv_heads`` (not attention heads) makes this correct for GQA/MQA models.
    """
    bytes_total = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem * context_length
    return bytes_total / 1e9


def total_footprint_gb(
    n_params_billions: float,
    quant: Quantization,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_length: int,
) -> float:
    """Convenience: weights + KV cache for a dense model."""
    return dense_weight_gb(n_params_billions, quant) + kv_cache_gb(
        n_layers, n_kv_heads, head_dim, context_length
    )
