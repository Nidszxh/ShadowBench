"""Throughput model sanity + monotonicity (DATAFLOW §1.3–1.4)."""

from __future__ import annotations

from shadowbench.predictor.dense import estimate_dense_tps
from shadowbench.predictor.moe import compute_base_fraction, estimate_moe_tps


def test_dense_resident_is_faster_than_spill() -> None:
    resident = estimate_dense_tps(
        weight_gb=4.0, kv_cache_gb=0.5, vram_total_gb=8.0, ram_bandwidth_gbps=50.0, pcie_gbps=12.0
    )
    spilled = estimate_dense_tps(
        weight_gb=10.0, kv_cache_gb=0.5, vram_total_gb=8.0, ram_bandwidth_gbps=50.0, pcie_gbps=12.0
    )
    assert resident.fits_in_vram is True
    assert spilled.fits_in_vram is False
    assert resident.tps > spilled.tps


def test_dense_more_vram_never_slower() -> None:
    small = estimate_dense_tps(
        12.0, 0.5, vram_total_gb=8.0, ram_bandwidth_gbps=50.0, pcie_gbps=12.0
    )
    large = estimate_dense_tps(
        12.0, 0.5, vram_total_gb=16.0, ram_bandwidth_gbps=50.0, pcie_gbps=12.0
    )
    assert large.tps >= small.tps


def test_moe_partial_offload_is_plausible() -> None:
    est = estimate_moe_tps(
        total_weight_gb=20.0,
        kv_cache_gb=0.5,
        n_experts=128,
        n_experts_active=8,
        vram_total_gb=8.0,
        ram_bandwidth_gbps=50.0,
        pcie_gbps=12.0,
    )
    assert est.base_fits_in_vram is True
    assert 0.0 < est.experts_offloaded_fraction < 1.0
    assert est.tps > 0


def test_compute_base_fraction_with_active_params() -> None:
    """Qwen3.6-35B-A3B: 35B total, 3.0B active, 256/8 → ~5.6% base."""
    frac = compute_base_fraction(35.0, 3.0, 256, 8)
    assert 0.04 < frac < 0.07, f"expected ~0.056, got {frac}"


def test_compute_base_fraction_fallback_on_none() -> None:
    """When n_params_active_billions is None, use FALLBACK_BASE_FRACTION (=0.15)."""
    frac = compute_base_fraction(35.0, None, 128, 8)
    assert frac == 0.15


def test_moe_base_spill_is_severe() -> None:
    # Tiny VRAM can't even hold the base layers → everything runs on CPU.
    est = estimate_moe_tps(
        20.0, 0.5, 128, 8, vram_total_gb=2.0, ram_bandwidth_gbps=50.0, pcie_gbps=12.0
    )
    assert est.base_fits_in_vram is False
    assert "cpu" in est.bottleneck
