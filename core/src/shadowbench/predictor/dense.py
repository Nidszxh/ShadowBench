"""Dense-model throughput estimation (``DATAFLOW.md §1.4``, dense branch).

Decode is memory-bound: every weight is read once per generated token, so tokens/sec ≈ effective memory
bandwidth ÷ resident weight size. When the model spills out of VRAM, the spilled fraction is streamed over the
(much slower) PCIe bus each token, producing the characteristic performance cliff.

Constants marked ``CALIBRATION`` are first-order defaults tuned against ``datasets/golden.jsonl`` in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

#: CALIBRATION: nominal resident-VRAM read bandwidth (GB/s) until a per-GPU value is measured (P1.2).
DEFAULT_VRAM_BANDWIDTH_GBPS = 400.0


@dataclass(slots=True)
class ThroughputEstimate:
    tps: float
    fits_in_vram: bool
    bottleneck: str


def estimate_dense_tps(
    weight_gb: float,
    kv_cache_gb: float,
    vram_total_gb: float,
    pcie_gbps: float,
    *,
    vram_bandwidth_gbps: float = DEFAULT_VRAM_BANDWIDTH_GBPS,
) -> ThroughputEstimate:
    """Estimate decode tokens/sec for a dense model.

    Args:
        weight_gb: Quantized weight footprint (see ``predictor.memory.dense_weight_gb``).
        kv_cache_gb: KV-cache footprint at the target context length.
        vram_total_gb: Usable GPU VRAM.
        pcie_gbps: *Measured* host↔device bandwidth (the spill-path speed).
        vram_bandwidth_gbps: Resident-path bandwidth.
    """
    total = weight_gb + kv_cache_gb
    if total <= vram_total_gb:
        tps = vram_bandwidth_gbps / total if total > 0 else 0.0
        return ThroughputEstimate(
            tps=tps, fits_in_vram=True, bottleneck="vram-resident (memory-bound)"
        )

    # Spill: KV stays in VRAM, weights fill the rest, remainder streams over PCIe every token.
    resident_weights = max(vram_total_gb - kv_cache_gb, 0.0)
    spilled = max(total - vram_total_gb, 0.0)
    seconds_per_token = resident_weights / vram_bandwidth_gbps + spilled / max(pcie_gbps, 1e-6)
    tps = 1.0 / seconds_per_token if seconds_per_token > 0 else 0.0
    return ThroughputEstimate(tps=tps, fits_in_vram=False, bottleneck="pcie-spill (severe)")
