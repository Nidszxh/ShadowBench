"""Config Coach flag rendering + heuristics (DATAFLOW §1.5–1.6)."""

from __future__ import annotations

from shadowbench.common.types import ModelTopology, Quantization, Task
from shadowbench.predictor.config_coach import build_flags
from shadowbench.predictor.models import ModelSpec, RuntimeFlags


def test_flags_to_cli_dense() -> None:
    cli = RuntimeFlags().to_cli()
    assert "--ngl 99" in cli
    assert "--n-cpu-moe" not in cli


def test_flags_to_cli_moe_offload() -> None:
    cli = RuntimeFlags(n_cpu_moe=64, no_mmap=True, ubatch=2048).to_cli()
    assert "--n-cpu-moe 64" in cli
    assert "--no-mmap" in cli
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
        _moe_spec(), vram_total_gb=8.0, pcie_gbps=12.0, experts_offloaded_fraction=0.5
    )
    assert flags.parallel == 1
    assert flags.n_cpu_moe is not None
    assert flags.no_mmap is True
    # Slow PCIe → widened micro-batch to hide expert-streaming latency.
    assert flags.ubatch == 2048


def test_fast_pcie_uses_smaller_ubatch() -> None:
    flags = build_flags(
        _moe_spec(), vram_total_gb=8.0, pcie_gbps=32.0, experts_offloaded_fraction=0.5
    )
    assert flags.ubatch == 1024
