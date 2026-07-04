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
    n_params_active_billions: float | None = None
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

    ngl: int = Field(
        99,
        description="--ngl: GPU layers (99 = all for known models, max layers is <99 for all current architectures)",
    )
    n_cpu_moe: int | None = Field(None, description="--n-cpu-moe: MoE expert layers kept on CPU")
    ubatch: int = Field(512, description="-ub / --ubatch: prefill micro-batch size")
    parallel: int = Field(1, description="--parallel: concurrent inference slots")
    no_mmap: bool = Field(False, description="--no-mmap: load all weights eagerly into RAM")
    mlock: bool = Field(False, description="--mlock: lock model pages in RAM to prevent swapping")
    flash_attn: bool = Field(
        False, description="--flash-attn: use flash attention to reduce KV cache usage"
    )
    cache_type_k: str = Field(
        "f16", description="--cache-type-k: KV cache key quantization (f16, q8_0, q4_0)"
    )
    cache_type_v: str = Field(
        "f16", description="--cache-type-v: KV cache value quantization (f16, q8_0, q4_0)"
    )

    def to_cli(self) -> str:
        parts = [f"--ngl {self.ngl}", f"-ub {self.ubatch}", f"--parallel {self.parallel}"]
        if self.flash_attn:
            parts.append("--flash-attn")
        if self.cache_type_k != "f16":
            parts.append(f"--cache-type-k {self.cache_type_k}")
        if self.cache_type_v != "f16":
            parts.append(f"--cache-type-v {self.cache_type_v}")
        if self.n_cpu_moe is not None:
            parts.append(f"--n-cpu-moe {self.n_cpu_moe}")
        if self.no_mmap:
            parts.append("--no-mmap")
        if self.mlock:
            parts.append("--mlock")
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
