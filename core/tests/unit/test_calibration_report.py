"""Accuracy report tests (P2.5).

Tests the report evaluation against synthetic golden.jsonl data.
"""

from __future__ import annotations

import json
from pathlib import Path

from shadowbench.calibration.report import evaluate


def _write_golden(path: Path, rows: list[dict]) -> Path:
    """Write a golden.jsonl file from a list of dicts."""
    p = path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    return p


def test_evaluate_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    report = evaluate(str(p))
    assert report.n_samples == 0


def test_evaluate_with_matching_model(tmp_path: Path) -> None:
    """Test against a model in the catalog with exact match."""
    rows = [
        {
            "schema_version": 1,
            "gpu_name": "NVIDIA RTX 4090",
            "vram_total_mb": 24576,
            "system_ram_gb": 64,
            "cpu_matmul_gbps": 24.0,
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "measured_tps": 100.0,
            "source": "test",
            "notes": "",
        },
    ]
    p = _write_golden(tmp_path, rows)
    report = evaluate(str(p))
    assert report.n_samples == 1
    assert report.median_abs_pct_error > 0  # predictor is not perfect
    assert report.mean_abs_pct_error > 0
    assert report.min_abs_pct_error > 0
    assert report.max_abs_pct_error >= report.median_abs_pct_error


def test_evaluate_unknown_model_skipped(tmp_path: Path) -> None:
    rows = [
        {
            "schema_version": 1,
            "gpu_name": "Unknown GPU",
            "vram_total_mb": 8192,
            "system_ram_gb": 16,
            "cpu_matmul_gbps": 12.0,
            "model_id": "nonexistent/model",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "measured_tps": 50.0,
            "source": "test",
        },
    ]
    p = _write_golden(tmp_path, rows)
    report = evaluate(str(p))
    assert report.n_samples == 0


def test_evaluate_multiple_rows(tmp_path: Path) -> None:
    # Llama 3 8B Q4_K_M on RTX 4090: predicted ~63 t/s, set measured close to it
    rows = [
        {
            "schema_version": 1,
            "gpu_name": "RTX 4090",
            "vram_total_mb": 24576,
            "system_ram_gb": 64,
            "cpu_matmul_gbps": 24.0,
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "measured_tps": 60.0,
            "source": "test",
        },
        {
            "schema_version": 1,
            "gpu_name": "RTX 4090",
            "vram_total_mb": 24576,
            "system_ram_gb": 64,
            "cpu_matmul_gbps": 24.0,
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            "quantization": "Q5_K_M",
            "context_length": 4096,
            "measured_tps": 50.0,
            "source": "test",
        },
    ]
    p = _write_golden(tmp_path, rows)
    report = evaluate(str(p))
    assert report.n_samples == 2
    assert report.median_abs_pct_error > 0
    assert 0 < report.within_band_fraction <= 1.0
    assert report.mean_abs_pct_error >= report.median_abs_pct_error


def test_evaluate_within_band(tmp_path: Path) -> None:
    # Set a high bandwidth and match measured to predicted (~63 t/s) so error is small
    rows = [
        {
            "schema_version": 1,
            "gpu_name": "RTX 4090",
            "vram_total_mb": 24576,
            "system_ram_gb": 64,
            "cpu_matmul_gbps": 50.0,
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "measured_tps": 65.0,
            "source": "test",
        },
    ]
    p = _write_golden(tmp_path, rows)
    report = evaluate(str(p))
    assert report.n_samples == 1
    assert report.within_band_fraction == 1.0


def test_evaluate_bands_reported(tmp_path: Path) -> None:
    rows = [
        {
            "schema_version": 1,
            "gpu_name": "RTX 4090",
            "vram_total_mb": 24576,
            "system_ram_gb": 64,
            "cpu_matmul_gbps": 24.0,
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "measured_tps": 100.0,
            "source": "test",
        },
    ]
    p = _write_golden(tmp_path, rows)
    report = evaluate(str(p))
    assert report.p90_abs_pct_error >= report.median_abs_pct_error
    assert report.mean_abs_pct_error >= 0
