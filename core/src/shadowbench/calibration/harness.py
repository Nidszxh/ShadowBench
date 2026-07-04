"""Ground-truth benchmark harness.  [Phase 2 — P2.4]

Wraps ``llama-bench`` (or an Ollama run) to measure *real* tokens/sec, so every run can contribute a data
point to ``datasets/golden.jsonl``. This measured value is the calibration target for the Predictor formulas.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GroundTruth:
    model_id: str
    quantization: str
    context_length: int
    measured_tps: float
    source: str = ""


@dataclass(slots=True)
class BenchmarkResult:
    ground_truth: GroundTruth
    raw_output: str


def find_llama_bench(*, prefer_path: str | None = None) -> str:
    """Locate the ``llama-bench`` executable.

    Checks, in order:
      1. ``prefer_path`` (explicit override).
      2. ``LLAMA_BENCH_PATH`` environment variable.
      3. ``llama-bench`` on the system ``PATH``.
      4. Common install locations.

    Raises:
        FileNotFoundError: if ``llama-bench`` cannot be found.
    """
    candidates: list[str] = []
    if prefer_path:
        candidates.append(prefer_path)
    if env_path := os.environ.get("LLAMA_BENCH_PATH"):
        candidates.append(env_path)

    candidates.append("llama-bench")

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 or "usage" in (result.stdout + result.stderr).lower():
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    msg = (
        "llama-bench not found. Install llama.cpp (https://github.com/ggerganov/llama.cpp) "
        "or set LLAMA_BENCH_PATH."
    )
    raise FileNotFoundError(msg)


def measure_tps(
    model_path: str,
    *,
    context_length: int = 4096,
    llama_bench_path: str | None = None,
    extra_args: list[str] | None = None,
) -> GroundTruth:
    """Run a short real benchmark and return measured tokens/sec.

    Args:
        model_path: Path to the GGUF model file.
        context_length: Target context length for benchmarking.
        llama_bench_path: Explicit path to ``llama-bench`` (auto-detected if ``None``).
        extra_args: Additional flags passed to ``llama-bench``.

    Returns:
        A :class:`GroundTruth` with measured t/s parsed from the output.
    """
    bench = find_llama_bench(prefer_path=llama_bench_path)
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cmd = [
        bench,
        "-m",
        model_path,
        "-p",
        str(context_length),
        "-n",
        "128",
        "-o",
        "json",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed (exit {result.returncode}):\n  stderr: {result.stderr.strip()}"
        )

    raw = result.stdout.strip()
    measurements = _parse_llama_bench_json(raw)

    if not measurements:
        raise RuntimeError(f"No usable measurement found in llama-bench output:\n{raw[:2000]}")

    measurement = measurements[-1]

    tps_val = measurement.get("tps", 0.0)
    model_label = str(measurement.get("model", ""))
    model_id, quant = _parse_model_name(model_path_obj.stem, model_label)

    return GroundTruth(
        model_id=model_id,
        quantization=quant,
        context_length=context_length,
        measured_tps=float(tps_val),  # type: ignore[arg-type]
        source="",
    )


def _parse_llama_bench_json(output: str) -> list[dict[str, object]]:
    """Parse llama-bench JSON output into structured measurements."""
    data = json.loads(output)
    if isinstance(data, dict):
        data = data.get("results", [data])

    measurements: list[dict[str, object]] = []
    for entry in data if isinstance(data, list) else [data]:
        if not isinstance(entry, dict):
            continue
        tps_raw = entry.get("t/s") or entry.get("avg_ts")
        if tps_raw is None:
            continue
        tps = float(tps_raw)
        if tps <= 0:
            continue
        model_str = str(entry.get("model_type", "") or entry.get("model", ""))
        measurements.append({"tps": tps, "model": model_str})

    return measurements


def _parse_model_name(stem: str, bench_label: str) -> tuple[str, str]:
    """Extract model_id and quantization from the filename or benchmark label.

    Typical GGUF filenames: ``Meta-Llama-3-8B-Instruct-Q4_K_M.gguf``
    The benchmark label might be ``"llama 8B Q4_K_M"``.
    """
    quant = _find_quant(stem) or _find_quant(bench_label) or "Q4_K_M"
    model_id = stem
    if quant and quant in model_id:
        model_id = model_id[: model_id.index(quant)].rstrip("-._ ")
    return model_id, quant


_QUANT_PATTERNS = re.compile(r"(Q[0-9]_[K0-9](?:_[A-Z])?|FP16|FP32|BF16)", re.IGNORECASE)


def _find_quant(text: str) -> str | None:
    match = _QUANT_PATTERNS.search(text)
    return match.group(0).upper() if match else None
