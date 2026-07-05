"""Config Coach — translates predicted bottlenecks into runtime flags.

Implements flag heuristics (``DATAFLOW.md §1.5–1.6``): expert offload, ubatch size, parallelism, KV cache quant.
"""

from __future__ import annotations

from shadowbench.common.types import KVCacheQuantization
from shadowbench.predictor.models import ModelSpec, RuntimeFlags

#: Below this VRAM we force single-slot to protect weight memory.
_LOW_VRAM_GB = 12.0
#: Below this system RAM bandwidth we widen the micro-batch to hide CPU-offload latency.
_SLOW_RAM_GBPS = 30.0
#: Context length beyond which flash attention is recommended.
_FLASH_ATTN_MIN_CTX = 4096
#: Context length beyond which we downgrade KV cache to q8_0 to stay viable.
_LONG_CTX_GB = 16384
#: Below this VRAM we downgrade KV cache to q8_0 unconditionally.
_TIGHT_VRAM_GB = 8.0


def _pick_kv_quant(vram_total_gb: float, context_length: int) -> KVCacheQuantization:
    if vram_total_gb < _TIGHT_VRAM_GB or context_length >= _LONG_CTX_GB:
        return KVCacheQuantization.Q8_0
    return KVCacheQuantization.FP16


def build_flags(
    spec: ModelSpec,
    *,
    vram_total_gb: float,
    pcie_gbps: float,
    experts_offloaded_fraction: float = 0.0,
    ram_bandwidth_gbps: float = 30.0,
    context_length: int = 4096,
    is_pool_provider: bool = False,
    kv_cache_quant: KVCacheQuantization | None = None,
) -> RuntimeFlags:
    """Produce the runtime flags for a given model + hardware bottleneck."""
    flags = RuntimeFlags(
        ngl=99,
        n_cpu_moe=None,
        ubatch=512,
        parallel=1,
        no_mmap=False,
        mlock=False,
        flash_attn=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )

    kv = kv_cache_quant or _pick_kv_quant(vram_total_gb, context_length)
    if kv != KVCacheQuantization.FP16:
        flags.cache_type_k = kv.value
        flags.cache_type_v = kv.value

    if spec.is_moe and experts_offloaded_fraction > 0 and spec.n_experts:
        flags.n_cpu_moe = max(1, round(spec.n_experts * experts_offloaded_fraction))
        flags.ubatch = 2048 if ram_bandwidth_gbps < _SLOW_RAM_GBPS else 1024
        flags.no_mmap = True
        flags.mlock = True

    if vram_total_gb < _LOW_VRAM_GB and not is_pool_provider:
        flags.parallel = 1

    if context_length >= _FLASH_ATTN_MIN_CTX:
        flags.flash_attn = True

    return flags
