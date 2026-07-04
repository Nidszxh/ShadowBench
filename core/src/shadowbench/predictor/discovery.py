"""Requirement Discovery engine (``ARCHITECTURE.md §3.2``).

Maps the 3-variable matrix — Task × Hardware × User profile — to a ranked recommendation. This is the single
public entry point of the Predictor module.
"""

from __future__ import annotations

from shadowbench.common.errors import PredictorError
from shadowbench.common.types import KVCacheQuantization, Quantization, Task, UserProfile
from shadowbench.predictor import memory
from shadowbench.predictor.catalog import candidates_for_task
from shadowbench.predictor.config_coach import build_flags
from shadowbench.predictor.dense import estimate_dense_tps
from shadowbench.predictor.models import ModelSpec, Prediction, Recommendation
from shadowbench.predictor.moe import estimate_moe_tps
from shadowbench.profiler.models import HardwareProfile

#: A recommendation must clear this floor to be considered "usable".
_MIN_USABLE_TPS = 5.0
#: Highest-quality quant we'll try first (best capability per model).
_QUANT_PREFERENCE = (
    Quantization.Q6_K,
    Quantization.Q5_K_M,
    Quantization.Q4_K_M,
    Quantization.Q3_K_M,
    Quantization.Q2_K,
)
#: KV-cache quant preference — best quality first; downgrades save VRAM at negligible quality cost.
_KV_CACHE_PREFERENCE = (
    KVCacheQuantization.FP16,
    KVCacheQuantization.Q8_0,
    KVCacheQuantization.Q4_0,
)


def recommend(
    profile: HardwareProfile,
    task: Task,
    user_profile: UserProfile,
    *,
    context_length: int | None = None,
) -> Recommendation:
    """Return the best-fit model + quant + flags for the hardware and intent.

    Raises:
        PredictorError: if no candidate clears the usability floor.
    """
    candidates = candidates_for_task(task)
    if not candidates:
        raise PredictorError(f"No catalog models tagged for task '{task.value}'.")

    scored: list[Recommendation] = []
    for spec in candidates:
        rec = _best_recommendation_for(spec, profile, context_length)
        if rec is not None:
            scored.append(rec)

    usable = [r for r in scored if r.prediction.predicted_tps >= _MIN_USABLE_TPS]
    if not usable:
        raise PredictorError("No usable model/quant fits this hardware within the speed floor.")

    usable.sort(key=lambda r: _rank_key(r, user_profile), reverse=True)
    return usable[0]


def _best_recommendation_for(
    spec: ModelSpec,
    profile: HardwareProfile,
    context_length: int | None,
) -> Recommendation | None:
    """Pick the highest-quality quant for one model that stays usable on this hardware."""
    ctx = context_length or spec.context_default
    vram_gb = (profile.gpu.vram_total_mb / 1000) if profile.gpu else 0.0
    pcie_gbps = profile.bandwidth.cpu_matmul_gbps if profile.bandwidth else 8.0
    ram_gbps = profile.bandwidth.system_ram_gbps if profile.bandwidth else 30.0

    for quant in _QUANT_PREFERENCE:
        if quant not in spec.available_quants:
            continue
        for kv_quant in _KV_CACHE_PREFERENCE:
            prediction, offloaded = _predict(
                spec, quant, ctx, vram_gb, pcie_gbps, ram_gbps, kv_quant=kv_quant
            )
            if prediction.predicted_tps >= _MIN_USABLE_TPS:
                flags = build_flags(
                    spec,
                    vram_total_gb=vram_gb,
                    pcie_gbps=pcie_gbps,
                    experts_offloaded_fraction=offloaded,
                    ram_bandwidth_gbps=ram_gbps,
                    context_length=ctx,
                    kv_cache_quant=kv_quant,
                )
                return Recommendation(
                    prediction=prediction,
                    flags=flags,
                    rationale=_rationale(spec, prediction),
                )
    return None


def _predict(
    spec: ModelSpec,
    quant: Quantization,
    ctx: int,
    vram_gb: float,
    pcie_gbps: float,
    ram_gbps: float = 30.0,
    *,
    kv_quant: KVCacheQuantization = KVCacheQuantization.FP16,
) -> tuple[Prediction, float]:
    """Run the topology-appropriate throughput model. Returns (prediction, offloaded_fraction)."""
    weight_gb = memory.dense_weight_gb(spec.n_params_billions, quant)
    kv_gb = memory.kv_cache_gb(
        spec.n_layers, spec.n_kv_heads, spec.head_dim, ctx, kv_quant=kv_quant
    )

    if spec.is_moe and spec.n_experts and spec.n_experts_active:
        from shadowbench.predictor.moe import compute_base_fraction

        base_frac = compute_base_fraction(
            spec.n_params_billions,
            spec.n_params_active_billions,
            spec.n_experts,
            spec.n_experts_active,
        )
        est = estimate_moe_tps(
            weight_gb,
            kv_gb,
            spec.n_experts,
            spec.n_experts_active,
            vram_gb,
            ram_gbps,
            pcie_gbps,
            base_fraction=base_frac,
        )
        prediction = Prediction(
            model_id=spec.id,
            quantization=quant,
            context_length=ctx,
            predicted_tps=round(est.tps, 1),
            weight_gb=round(weight_gb, 2),
            kv_cache_gb=round(kv_gb, 2),
            fits_in_vram=est.base_fits_in_vram and est.experts_offloaded_fraction == 0,
            bottleneck=est.bottleneck,
        )
        return prediction, est.experts_offloaded_fraction

    est_d = estimate_dense_tps(weight_gb, kv_gb, vram_gb, ram_gbps, pcie_gbps)
    prediction = Prediction(
        model_id=spec.id,
        quantization=quant,
        context_length=ctx,
        predicted_tps=round(est_d.tps, 1),
        weight_gb=round(weight_gb, 2),
        kv_cache_gb=round(kv_gb, 2),
        fits_in_vram=est_d.fits_in_vram,
        bottleneck=est_d.bottleneck,
    )
    return prediction, 0.0


def _rank_key(rec: Recommendation, user_profile: UserProfile) -> tuple[float, float]:
    """Ranking: speed-first sorts by t/s; intelligence-first sorts by capability (weight) then t/s."""
    tps = rec.prediction.predicted_tps
    capability = rec.prediction.weight_gb
    if user_profile is UserProfile.SPEED:
        return (tps, capability)
    return (capability, tps)


def _rationale(spec: ModelSpec, prediction: Prediction) -> str:
    return (
        f"{spec.name} @ {prediction.quantization.value}: "
        f"~{prediction.predicted_tps} tok/s ({prediction.bottleneck})."
    )
