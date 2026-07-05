"""Accuracy report: predicted vs. actual across ``datasets/golden.jsonl``. CI accuracy gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadowbench.common.types import KVCacheQuantization, Quantization
from shadowbench.predictor.catalog import load_catalog
from shadowbench.predictor.dense import estimate_dense_tps
from shadowbench.predictor.memory import dense_weight_gb, kv_cache_gb
from shadowbench.predictor.models import ModelSpec
from shadowbench.predictor.moe import compute_base_fraction, estimate_moe_tps


@dataclass(slots=True)
class AccuracyReport:
    n_samples: int
    median_abs_pct_error: float
    p90_abs_pct_error: float
    within_band_fraction: float
    mean_abs_pct_error: float = 0.0
    min_abs_pct_error: float = 0.0
    max_abs_pct_error: float = 0.0


@dataclass(slots=True)
class GoldenRow:
    schema_version: int
    gpu_name: str | None
    vram_total_mb: int
    system_ram_gb: int
    model_id: str
    quantization: str
    context_length: int
    measured_tps: float
    source: str = ""
    notes: str = ""
    host_to_device_gbps: float = 0.0
    system_ram_gbps: float = 30.0
    kv_cache_quant: str = "f16"


def evaluate(
    golden_path: str,
    *,
    error_band: float = 0.25,
) -> AccuracyReport:
    """Score the Predictor against the golden dataset.

    Args:
        golden_path: Path to ``golden.jsonl``.
        error_band: The ±fraction accuracy target (e.g., 0.25 = ±25%).

    Returns:
        An :class:`AccuracyReport` with aggregate error metrics.
    """
    rows = _load_golden(golden_path)
    if not rows:
        return AccuracyReport(
            n_samples=0,
            median_abs_pct_error=0.0,
            p90_abs_pct_error=0.0,
            within_band_fraction=0.0,
        )

    catalog = {spec.id: spec for spec in load_catalog()}

    abs_pct_errors: list[float] = []
    for row in rows:
        spec = catalog.get(row.model_id)
        if spec is None:
            continue

        vram_gb = row.vram_total_mb / 1000.0
        pcie_gbps = row.host_to_device_gbps
        ram_gbps = row.system_ram_gbps
        quant = _find_quant_in_catalog(spec, row.quantization)
        if quant is None:
            continue

        w_gb = dense_weight_gb(spec.n_params_billions, quant)
        kv_quant = KVCacheQuantization(row.kv_cache_quant)
        kv_gb = kv_cache_gb(
            spec.n_layers, spec.n_kv_heads, spec.head_dim, row.context_length, kv_quant=kv_quant
        )

        if spec.is_moe and spec.n_experts and spec.n_experts_active:
            base_frac = compute_base_fraction(
                spec.n_params_billions,
                spec.n_params_active_billions,
                spec.n_experts,
                spec.n_experts_active,
            )
            est = estimate_moe_tps(
                w_gb,
                kv_gb,
                spec.n_experts,
                spec.n_experts_active,
                vram_gb,
                ram_gbps,
                pcie_gbps,
                base_fraction=base_frac,
            )
            predicted_tps = est.tps
        else:
            est_d = estimate_dense_tps(w_gb, kv_gb, vram_gb, ram_gbps, pcie_gbps)
            predicted_tps = est_d.tps

        if predicted_tps <= 0:
            continue

        abs_pct_error = abs(predicted_tps - row.measured_tps) / row.measured_tps * 100.0
        abs_pct_errors.append(abs_pct_error)

    if not abs_pct_errors:
        return AccuracyReport(
            n_samples=0,
            median_abs_pct_error=0.0,
            p90_abs_pct_error=0.0,
            within_band_fraction=0.0,
        )

    sorted_errors = sorted(abs_pct_errors)
    n = len(sorted_errors)
    median = (
        sorted_errors[n // 2]
        if n % 2
        else (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2.0
    )
    p90_idx = min(math.ceil(0.9 * n) - 1, n - 1)
    p90 = sorted_errors[p90_idx]
    within_band = sum(1 for e in abs_pct_errors if e <= error_band * 100.0) / n

    return AccuracyReport(
        n_samples=n,
        median_abs_pct_error=round(median, 2),
        p90_abs_pct_error=round(p90, 2),
        within_band_fraction=round(within_band, 4),
        mean_abs_pct_error=round(sum(abs_pct_errors) / n, 2),
        min_abs_pct_error=round(sorted_errors[0], 2),
        max_abs_pct_error=round(sorted_errors[-1], 2),
    )


def _load_golden(path: str) -> list[GoldenRow]:
    """Load and parse the golden.jsonl file."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[GoldenRow] = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        data: dict[str, Any] = json.loads(line)
        rows.append(
            GoldenRow(
                schema_version=data.get("schema_version", 1),
                gpu_name=data.get("gpu_name"),
                vram_total_mb=data.get("vram_total_mb", 0),
                system_ram_gb=data.get("system_ram_gb", 0),
                host_to_device_gbps=data.get(
                    "cpu_matmul_gbps", data.get("host_to_device_gbps", 0.0)
                ),
                model_id=data.get("model_id", ""),
                quantization=data.get("quantization", ""),
                context_length=data.get("context_length", 4096),
                measured_tps=data.get("measured_tps", 0.0),
                source=data.get("source", ""),
                notes=data.get("notes", ""),
                system_ram_gbps=data.get("system_ram_gbps", 30.0),
                kv_cache_quant=data.get("kv_cache_quant", "f16"),
            )
        )
    return rows


def _find_quant_in_catalog(spec: ModelSpec, quant_str: str) -> Quantization | None:
    """Match a quantization string to the Quantization enum from the catalog."""

    for q in spec.available_quants:
        if q.value == quant_str:
            return q
    # Fallback: try direct lookup
    try:
        return Quantization(quant_str)
    except ValueError:
        return None
