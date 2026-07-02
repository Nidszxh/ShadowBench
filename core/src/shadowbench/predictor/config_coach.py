"""Config Coach — translate a predicted bottleneck into exact runtime flags.

Implements the flag heuristics from ``DATAFLOW.md §1.5–1.6``:
  * ``--n-cpu-moe`` scaled to the offloaded-expert fraction on MoE models.
  * ``-ub`` / ``--ubatch`` raised on slow-PCIe / high-RAM systems so GPU kernels don't stall waiting on
    streamed expert weights.
  * ``--parallel`` forced to 1 on low-VRAM machines (multi-slot silently multiplies KV allocation), unless the
    node is acting as a pool provider.
"""

from __future__ import annotations

from shadowbench.predictor.models import ModelSpec, RuntimeFlags

#: Below this VRAM we force single-slot to protect weight memory.
_LOW_VRAM_GB = 12.0
#: Below this measured PCIe bandwidth we widen the micro-batch to hide transfer latency.
_SLOW_PCIE_GBPS = 16.0


def build_flags(
    spec: ModelSpec,
    *,
    vram_total_gb: float,
    pcie_gbps: float,
    experts_offloaded_fraction: float = 0.0,
    is_pool_provider: bool = False,
) -> RuntimeFlags:
    """Produce the runtime flags for a given model + hardware bottleneck."""
    flags = RuntimeFlags()

    if spec.is_moe and experts_offloaded_fraction > 0 and spec.n_experts:
        # Map the offloaded fraction to the number of expert layers to keep on CPU.
        flags.n_cpu_moe = max(1, round(spec.n_layers * experts_offloaded_fraction))
        # Slow PCIe → bigger micro-batch keeps the GPU saturated while experts stream in.
        flags.ubatch = 2048 if pcie_gbps < _SLOW_PCIE_GBPS else 1024
        # Locking weights in RAM avoids mmap page-fault stalls during offload.
        flags.no_mmap = True

    if vram_total_gb < _LOW_VRAM_GB and not is_pool_provider:
        flags.parallel = 1

    return flags
