"""Mixture-of-Experts throughput estimation (``DATAFLOW.md §1.3``).

MoE models only activate a subset of experts per token, so decode reads far fewer bytes than the total weight
size implies. The predictor:

  1. Forces the non-expert base (attention/embeddings/norms) into VRAM — if it doesn't fit, predict a severe
     PCIe-fallback penalty.
  2. Fills remaining VRAM with as many experts as fit; the rest map to system RAM via an ``--n-cpu-moe``-style
     offload, streamed over PCIe.

Constants marked ``CALIBRATION`` are tuned against ``datasets/golden.jsonl`` in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowbench.predictor.dense import DEFAULT_VRAM_BANDWIDTH_GBPS

#: CALIBRATION: fallback share of params in non-expert base layers when n_params_active_billions is unknown.
FALLBACK_BASE_FRACTION = 0.15


def compute_base_fraction(
    n_params_billions: float,
    n_params_active_billions: float | None,
    n_experts: int,
    n_experts_active: int,
) -> float:
    if n_params_active_billions is None:
        return FALLBACK_BASE_FRACTION
    ratio = n_experts_active / max(n_experts, 1)
    numerator = n_params_active_billions - ratio * n_params_billions
    denominator = 1.0 - ratio
    if denominator <= 0 or numerator <= 0:
        return FALLBACK_BASE_FRACTION
    return numerator / denominator / n_params_billions


@dataclass(slots=True)
class MoeEstimate:
    tps: float
    base_fits_in_vram: bool
    experts_offloaded_fraction: float
    bottleneck: str


def estimate_moe_tps(
    total_weight_gb: float,
    kv_cache_gb: float,
    n_experts: int,
    n_experts_active: int,
    vram_total_gb: float,
    ram_bandwidth_gbps: float,
    pcie_gbps: float,
    *,
    vram_bandwidth_gbps: float = DEFAULT_VRAM_BANDWIDTH_GBPS,
    base_fraction: float | None = None,
) -> MoeEstimate:
    """Estimate decode tokens/sec for an MoE model with partial expert offload.

    Offloaded experts run on the **CPU** using system RAM bandwidth, *not* streamed back
    over PCIe (``llama.cpp``'s ``--n-cpu-moe`` keeps offloaded weights in system RAM and
    computes them on the CPU directly).  The ``pcie_gbps`` parameter is retained for future
    multi-GPU topologies.

    Args:
        total_weight_gb: Footprint of *all* weights (base + every expert) at the chosen quant.
        kv_cache_gb: KV-cache footprint at the target context length.
        n_experts: Total routed experts.
        n_experts_active: Experts activated per token.
        vram_total_gb: Usable GPU VRAM.
        ram_bandwidth_gbps: Measured system RAM read bandwidth (CPU-offloaded compute path).
        pcie_gbps: Measured host↔device bandwidth (reserved for multi-GPU setups).
        base_fraction: Fraction of total weight in non-expert base layers.
            If ``None``, derived from ``n_params_active_billions`` via the
            catalog model spec; falls back to ``FALLBACK_BASE_FRACTION``.
    """
    if base_fraction is None:
        base_fraction = FALLBACK_BASE_FRACTION
    base_gb = total_weight_gb * base_fraction
    expert_pool_gb = total_weight_gb - base_gb
    per_expert_gb = expert_pool_gb / max(n_experts, 1)

    # Bytes actually read per token: base layers + only the active experts.
    active_expert_gb = per_expert_gb * n_experts_active
    active_read_gb = base_gb + active_expert_gb

    base_fits = (base_gb + kv_cache_gb) <= vram_total_gb
    if not base_fits:
        # Base can't stay resident → everything runs on CPU with RAM bandwidth.
        tps = 1.0 / (active_read_gb / max(ram_bandwidth_gbps, 1e-6)) if active_read_gb > 0 else 0.0
        return MoeEstimate(
            tps=tps,
            base_fits_in_vram=False,
            experts_offloaded_fraction=1.0,
            bottleneck="cpu-only (RAM bandwidth-bound)",
        )

    # How many experts fit in the VRAM left over after base + KV?
    vram_for_experts = max(vram_total_gb - base_gb - kv_cache_gb, 0.0)
    experts_in_vram = (
        min(n_experts, int(vram_for_experts / per_expert_gb)) if per_expert_gb > 0 else 0
    )
    offloaded_fraction = 1.0 - (experts_in_vram / n_experts if n_experts else 0.0)

    # Split the active-expert read: VRAM-resident experts use GPU bandwidth;
    # CPU-offloaded experts run on CPU cores using system RAM bandwidth.
    active_expert_vram_gb = active_expert_gb * (1.0 - offloaded_fraction)
    active_expert_ram_gb = active_expert_gb * offloaded_fraction
    seconds_per_token = (
        base_gb + active_expert_vram_gb
    ) / vram_bandwidth_gbps + active_expert_ram_gb / max(ram_bandwidth_gbps, 1e-6)
    tps = 1.0 / seconds_per_token if seconds_per_token > 0 else 0.0
    bottleneck = (
        "expert-offload (CPU RAM-bound)"
        if offloaded_fraction > 0
        else "vram-resident (memory-bound)"
    )
    return MoeEstimate(
        tps=tps,
        base_fits_in_vram=True,
        experts_offloaded_fraction=offloaded_fraction,
        bottleneck=bottleneck,
    )
