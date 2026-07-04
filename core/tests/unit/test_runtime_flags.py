"""Config Coach flag rendering + heuristics (DATAFLOW §1.5–1.6)."""

from __future__ import annotations

from shadowbench.common.types import KVCacheQuantization, ModelTopology, Quantization, Task
from shadowbench.predictor.config_coach import build_flags
from shadowbench.predictor.models import ModelSpec, RuntimeFlags


def test_flags_to_cli_dense() -> None:
    cli = RuntimeFlags().to_cli()
    assert "--ngl 99" in cli
    assert "--n-cpu-moe" not in cli
    assert "--mlock" not in cli


def test_flags_to_cli_moe_offload() -> None:
    cli = RuntimeFlags(n_cpu_moe=64, no_mmap=True, mlock=True, ubatch=2048).to_cli()
    assert "--n-cpu-moe 64" in cli
    assert "--no-mmap" in cli
    assert "--mlock" in cli
    assert "-ub 2048" in cli


def _moe_spec() -> ModelSpec:
    return ModelSpec(
        id="x/moe",
        name="MoE",
        topology=ModelTopology.MOE,
        tasks=[Task.CODING],
        n_params_billions=35,
        n_layers=48,
        n_kv_heads=8,
        head_dim=128,
        available_quants=[Quantization.Q4_K_M],
        n_experts=128,
        n_experts_active=8,
    )


def test_low_vram_moe_offload_sets_cpu_moe_and_widens_ubatch() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=8.0,
        pcie_gbps=12.0,
        experts_offloaded_fraction=0.5,
        ram_bandwidth_gbps=20.0,
    )
    assert flags.parallel == 1
    assert flags.n_cpu_moe is not None
    assert flags.no_mmap is True
    assert flags.mlock is True
    # Slow RAM → widened micro-batch to hide CPU-offload latency.
    assert flags.ubatch == 2048


def test_fast_ram_uses_smaller_ubatch() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=8.0,
        pcie_gbps=32.0,
        experts_offloaded_fraction=0.5,
        ram_bandwidth_gbps=60.0,
    )
    assert flags.ubatch == 1024


def test_flash_attn_set_at_long_context() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=16.0,
        pcie_gbps=32.0,
        experts_offloaded_fraction=0.0,
        context_length=8192,
    )
    assert flags.flash_attn is True


def test_flash_attn_not_set_at_short_context() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=16.0,
        pcie_gbps=32.0,
        experts_offloaded_fraction=0.0,
        context_length=2048,
    )
    assert flags.flash_attn is False


def test_tight_vram_downgrades_kv_cache_to_q8() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=6.0,
        pcie_gbps=12.0,
    )
    assert flags.cache_type_k == "q8_0"
    assert flags.cache_type_v == "q8_0"


def test_long_context_downgrades_kv_cache_to_q8() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=16.0,
        pcie_gbps=32.0,
        context_length=32768,
    )
    assert flags.cache_type_k == "q8_0"
    assert flags.cache_type_v == "q8_0"


def test_ample_vram_keeps_fp16_kv_cache() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=24.0,
        pcie_gbps=32.0,
        context_length=4096,
    )
    assert flags.cache_type_k == "f16"
    assert flags.cache_type_v == "f16"


def test_explicit_kv_quant_overrides_heuristic() -> None:
    flags = build_flags(
        _moe_spec(),
        vram_total_gb=6.0,
        pcie_gbps=12.0,
        kv_cache_quant=KVCacheQuantization.FP16,
    )
    assert flags.cache_type_k == "f16"
