"""Predictor data contracts: model specs (from the catalog) and prediction outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shadowbench.common.types import ModelTopology, Quantization, Task


class ModelSpec(BaseModel):
    """A catalog entry describing one model's architecture (from ``datasets/models_catalog.json``)."""

    id: str = Field(description="HF repo id, e.g. 'Qwen/Qwen3.5-35B-A3B-GGUF'")
    name: str
    topology: ModelTopology
    tasks: list[Task]
    n_params_billions: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    context_default: int = 4096
    available_quants: list[Quantization]
    # MoE-only (None for dense).
    n_experts: int | None = None
    n_experts_active: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.topology is ModelTopology.MOE


class RuntimeFlags(BaseModel):
    """Config Coach output — exact flags handed to llama.cpp / Ollama."""

    ngl: int = Field(99, description="--ngl: layers offloaded to GPU")
    n_cpu_moe: int | None = Field(None, description="--n-cpu-moe: MoE expert layers kept on CPU")
    ubatch: int = Field(512, description="-ub / --ubatch: prefill micro-batch size")
    parallel: int = Field(1, description="--parallel: concurrent inference slots")
    no_mmap: bool = Field(False, description="--no-mmap: lock weights in RAM")

    def to_cli(self) -> str:
        parts = [f"--ngl {self.ngl}", f"-ub {self.ubatch}", f"--parallel {self.parallel}"]
        if self.n_cpu_moe is not None:
            parts.append(f"--n-cpu-moe {self.n_cpu_moe}")
        if self.no_mmap:
            parts.append("--no-mmap")
        return " ".join(parts)


class Prediction(BaseModel):
    """Throughput estimate for one (model, quant, hardware, context) combination."""

    model_id: str
    quantization: Quantization
    context_length: int
    predicted_tps: float
    weight_gb: float
    kv_cache_gb: float
    fits_in_vram: bool
    #: Human-readable note on the dominant bottleneck (compute / VRAM spill / PCIe).
    bottleneck: str


class Recommendation(BaseModel):
    """Final ranked recommendation returned by the Requirement Discovery engine."""

    prediction: Prediction
    flags: RuntimeFlags
    rationale: str
