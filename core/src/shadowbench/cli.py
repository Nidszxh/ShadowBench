"""ShadowBench CLI — thin orchestration over modules. Desktop app calls the same functions via IPC."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from shadowbench import __version__
from shadowbench.calibration.harness import GroundTruth
from shadowbench.common.errors import ShadowBenchError
from shadowbench.common.logging import configure_logging
from shadowbench.common.types import Task, UserProfile
from shadowbench.predictor.discovery import recommend as run_recommend
from shadowbench.predictor.validate_catalog import find_catalog_path, validate_entries
from shadowbench.profiler.detect import profile_hardware

app = typer.Typer(
    name="shadowbench",
    help="Predict LLM throughput on your hardware and pool inference across the LAN.",
    no_args_is_help=True,
    add_completion=False,
)
catalog_app = typer.Typer(
    name="catalog",
    help="Manage the model catalog (datasets/models_catalog.json).",
)
app.add_typer(catalog_app, name="catalog")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"shadowbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def profile(
    no_stress: bool = typer.Option(
        False,
        "--no-stress",
        help="Skip the 3-second GEMM burner; system RAM bandwidth is still measured (~0.1s).",
    ),
) -> None:
    """Detect hardware and print a HardwareProfile as JSON."""
    result = profile_hardware(run_stress_test=not no_stress)
    console.print_json(result.model_dump_json(indent=2))


@app.command()
def recommend(
    task: Task = typer.Option(Task.CODING, "--task", help="What you want the model for."),
    user_profile: UserProfile = typer.Option(
        UserProfile.INTELLIGENCE, "--profile", help="Optimize for speed or intelligence."
    ),
    context: int = typer.Option(4096, "--context", help="Target context length (tokens)."),
    no_stress: bool = typer.Option(
        False,
        "--no-stress",
        help="Skip the 3-second GEMM burner; system RAM bandwidth is still measured (~0.1s).",
    ),
) -> None:
    """Recommend the best model + quantization + runtime flags for this machine."""
    try:
        hw = profile_hardware(run_stress_test=not no_stress)
        rec = run_recommend(hw, task, user_profile, context_length=context)
    except ShadowBenchError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="ShadowBench Recommendation", show_header=False)
    table.add_row("Model", rec.prediction.model_id)
    table.add_row("Quantization", rec.prediction.quantization.value)
    table.add_row("Predicted", f"{rec.prediction.predicted_tps} tok/s")
    table.add_row("Context", str(rec.prediction.context_length))
    table.add_row("Weights", f"{rec.prediction.weight_gb} GB")
    table.add_row("KV cache", f"{rec.prediction.kv_cache_gb} GB")
    table.add_row("Bottleneck", rec.prediction.bottleneck)
    table.add_row("Flags", rec.flags.to_cli())
    console.print(table)


@app.command()
def bench(
    model_path: str = typer.Argument(..., help="Path to the GGUF model file to benchmark."),
    context: int = typer.Option(4096, "--context", help="Target context length (tokens)."),
    contribute: bool = typer.Option(
        False, "--contribute", help="Append the measured result to datasets/golden.jsonl."
    ),
    golden_path: str | None = typer.Option(
        None,
        "--golden-path",
        help="Path to golden.jsonl (default: auto-detect).",
    ),
) -> None:
    """Measure real tokens/sec for a local GGUF model using llama-bench."""
    from shadowbench.calibration.harness import measure_tps as run_benchmark

    try:
        gt = run_benchmark(model_path, context_length=context)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Benchmark error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Benchmark Result", show_header=False)
    table.add_row("Model", gt.model_id)
    table.add_row("Quantization", gt.quantization)
    table.add_row("Context", str(gt.context_length))
    table.add_row("Measured", f"{gt.measured_tps} tok/s")
    console.print(table)

    if contribute:
        _append_to_golden(gt, golden_path)
        console.print("[green]✓[/green] Appended to golden.jsonl")


def _append_to_golden(gt: GroundTruth, golden_path: str | None) -> None:
    """Append a benchmark row to golden.jsonl."""
    from shadowbench.profiler.detect import profile_hardware

    if golden_path:
        p = Path(golden_path)
    else:
        env_path = os.environ.get("SHADOWBENCH_GOLDEN_PATH")
        if env_path:
            p = Path(env_path)
        else:
            here = Path(__file__).resolve()
            p = here.parent.parent.parent.parent / "datasets" / "golden.jsonl"
            if not p.exists():
                msg = (
                    f"Golden dataset not found at {p}. "
                    "Use --golden-path <path> or set SHADOWBENCH_GOLDEN_PATH to point to your golden.jsonl."
                )
                console.print(f"[red]Error:[/red] {msg}")
                raise typer.Exit(code=1)

    p.parent.mkdir(parents=True, exist_ok=True)

    hw = profile_hardware(run_stress_test=True)

    row = {
        "schema_version": 1,
        "gpu_name": hw.gpu.name if hw.gpu else None,
        "vram_total_mb": hw.gpu.vram_total_mb if hw.gpu else 0,
        "system_ram_gb": round(hw.system.ram_total_mb / 1000) if hw.system else 0,
        "cpu_matmul_gbps": round(hw.bandwidth.cpu_matmul_gbps, 1) if hw.bandwidth else 0.0,
        "system_ram_gbps": round(hw.bandwidth.system_ram_gbps, 1) if hw.bandwidth else 30.0,
        "model_id": gt.model_id,
        "quantization": gt.quantization,
        "context_length": gt.context_length,
        "measured_tps": gt.measured_tps,
        "source": "cli",
        "notes": "",
        "kv_cache_quant": "f16",
    }

    existing = []
    if p.exists():
        existing = [json.loads(line) for line in p.read_text().strip().splitlines() if line.strip()]

    existing.append(row)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in existing) + "\n")


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", help="Bind port for the OpenAI-compatible proxy."),
) -> None:
    """Start the local Shadow Pool proxy (Phase 4)."""
    _ = port
    console.print("[yellow]`serve` lands in Phase 4 (P4.3).[/yellow]")
    raise typer.Exit(code=0)


@catalog_app.command()
def validate(
    catalog_path: str | None = typer.Option(
        None,
        "--catalog",
        help="Path to models_catalog.json (default: auto-detect).",
    ),
) -> None:
    """Validate models_catalog.json against the ModelSpec schema. Exits 0 if valid, 1 otherwise.

    \b
    Examples:
      uv run shadowbench catalog validate
      uv run shadowbench catalog validate --catalog /path/to/models_catalog.json
    """
    path = find_catalog_path(catalog_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    count = validate_entries(raw.get("models", []))
    if count:
        console.print(f"[red]FAIL  {count} validation error(s)[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/green]  ({len(raw.get('models', []))} entries)")


@catalog_app.command()
def add(
    hf_id: str = typer.Argument(
        ..., help="Hugging Face model ID, e.g. 'Qwen/Qwen2.5-7B' or 'Qwen/Qwen2.5-7B-GGUF'."
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="HF access token (needed for gated models). Falls back to $HF_TOKEN.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the catalog entry to FILE instead of stdout.",
    ),
    append: bool = typer.Option(
        False,
        "--append",
        help="Insert the entry into the existing catalog and write back (implies --output).",
    ),
    catalog_path: str | None = typer.Option(
        None,
        "--catalog",
        help="Path to models_catalog.json (default: auto-detect, used with --append).",
    ),
) -> None:
    """Fetch model metadata from Hugging Face and generate a catalog entry.

    Tries the REST API, falls back to raw ``config.json``. Validates before output; warns on missing fields.

    \b
    Examples:
      uv run shadowbench catalog add Qwen/Qwen2.5-7B
      uv run shadowbench catalog add Qwen/Qwen2.5-7B --token hf_xxx
      uv run shadowbench catalog add Qwen/Qwen2.5-7B --append
      uv run shadowbench catalog add Qwen/Qwen2.5-7B --output entry.json
    """
    headers: dict[str, str] = {}
    t = token or os.environ.get("HF_TOKEN")
    if t:
        headers["Authorization"] = f"Bearer {t}"

    api_failures: list[str] = []

    def _fetch(url: str, label: str) -> dict[str, object] | None:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            code = exc.code
            hint = " Try --token <HF_TOKEN> or set HF_TOKEN." if code == 401 else ""
            api_failures.append(f"{label} returned HTTP {code}{hint}")
            return None
        except OSError as exc:
            api_failures.append(f"{label} — network error: {exc}")
            return None

    name = hf_id.split("/")[-1] if "/" in hf_id else hf_id

    # 1. API metadata
    api_data: dict[str, object] | None = _fetch(
        f"https://huggingface.co/api/models/{hf_id}", "HF API"
    )

    # 2. Raw config.json (richer architecture data)
    config: dict[str, object] = {}
    raw_config: dict[str, object] | None = _fetch(
        f"https://huggingface.co/{hf_id}/raw/main/config.json", "config.json"
    )
    if raw_config:
        config = raw_config
    elif api_data:
        raw_config_data = api_data.get("config")
        if isinstance(raw_config_data, dict):
            config = cast("dict[str, object]", raw_config_data)

    tags: list[str] = []
    if api_data:
        raw_card = api_data.get("cardData")
        if isinstance(raw_card, dict):
            card = cast("dict[str, object]", raw_card)
            raw_tags = card.get("tags")
            if isinstance(raw_tags, list):
                tags = cast("list[str]", raw_tags)

    # Topology heuristics
    name_lower = name.lower()
    topology_from_name = not api_data
    is_moe = (
        "moe" in name_lower
        or "a3b" in name_lower
        or "mixtral" in name_lower
        or "deepseek" in name_lower
        or any("moe" in (t or "").lower() for t in tags)
    )

    # Tasks heuristic
    tasks = ["general"]
    if "code" in name_lower or "coder" in name_lower:
        tasks.append("coding")
    if "reason" in name_lower:
        tasks.append("reasoning")

    def _safe_int(*keys: str, default: int = 0) -> int:
        for k in keys:
            v = config.get(k)
            if isinstance(v, int | float | str):
                return int(v)
        return default

    hidden_size: int = int(_safe_int("hidden_size"))
    n_heads: int = _safe_int("num_attention_heads", "num_heads")
    n_kv_heads: int = _safe_int("num_key_value_heads") or n_heads
    head_dim: int = _safe_int("head_dim") or (hidden_size // n_heads if n_heads else 0)
    n_layers: int = _safe_int("num_hidden_layers", "num_layers", "num_decoder_layers")
    context: int = _safe_int(
        "max_position_embeddings", "max_sequence_length", "seq_length", default=4096
    )

    n_params: float = 0.0
    if api_data:
        raw_p = api_data.get("_params")
        if isinstance(raw_p, int | float):
            n_params = round(raw_p / 1e9, 2)
        if not n_params and api_data.get("cardData"):
            card_data = api_data.get("cardData")
            raw_p = None
            if isinstance(card_data, dict):
                raw_p = card_data.get("params")
            if isinstance(raw_p, int | float):
                n_params = round(raw_p / 1e9, 2)
        if not n_params and api_data.get("safetensors"):
            st = api_data.get("safetensors", {})
            if isinstance(st, dict):
                raw_p = st.get("parameters")
                if isinstance(raw_p, dict):
                    total = (
                        raw_p.get("F32", 0)
                        or raw_p.get("BF16", 0)
                        or raw_p.get("F16", 0)
                        or sum(raw_p.values())
                        if raw_p
                        else 0
                    )
                    n_params = round(total / 1e9, 2)

    n_experts: int | None = _safe_int("num_local_experts", "num_experts") or None
    n_experts_active: int | None = _safe_int("num_experts_per_tok", "num_experts_per_token") or None

    entry: dict[str, object] = {
        "id": hf_id,
        "name": name,
        "topology": "moe" if is_moe else "dense",
        "tasks": tasks,
        "n_params_billions": n_params if n_params else 0,
        "n_layers": n_layers if n_layers else 0,
        "n_kv_heads": n_kv_heads if n_kv_heads else 0,
        "head_dim": head_dim if head_dim else 0,
        "context_default": context,
        "available_quants": ["Q2_K", "Q4_K_M", "Q5_K_M", "Q6_K"],
    }
    if is_moe:
        entry["n_experts"] = n_experts or 8
        entry["n_experts_active"] = n_experts_active or 2
        if n_params and n_experts and n_experts_active:
            entry["n_params_active_billions"] = round(n_params * n_experts_active / n_experts, 2)

    # Validate the generated entry
    violations = validate_entries([entry])
    if violations:
        print("Error: generated entry failed validation — see messages above", file=sys.stderr)
        for e in api_failures:
            print(f"  Warning: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    if api_failures:
        for e in api_failures:
            print(f"Warning: {e}", file=sys.stderr)
    missing = []
    if not n_params:
        missing.append("n_params_billions")
    if not n_layers:
        missing.append("n_layers")
    if not n_kv_heads:
        missing.append("n_kv_heads")
    if not head_dim:
        missing.append("head_dim")
    if missing:
        print(
            f"Warning: could not auto-detect: {', '.join(missing)} — fill these in manually.",
            file=sys.stderr,
        )
    if topology_from_name:
        print(
            "Warning: topology inferred from model name — verify it is correct.",
            file=sys.stderr,
        )

    # Render output
    payload = json.dumps(entry, indent=2)
    if output or append:
        dest = Path(output) if output else find_catalog_path(catalog_path)
        if append:
            raw = json.loads(dest.read_text(encoding="utf-8"))
            models: list[dict[str, object]] = raw.get("models", [])
            models.append(entry)
            raw["models"] = models
            dest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            console.print(f"[green]✓[/green] Appended entry to {dest}")
        else:
            dest.write_text(payload + "\n", encoding="utf-8")
            console.print(f"[green]✓[/green] Written to {dest}")
    else:
        console.print_json(payload)
        console.print(
            "\n[yellow]Review the entry and paste it into datasets/models_catalog.json[/yellow]"
            "\n[yellow]Or re-run with --append to insert it automatically.[/yellow]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
