"""ShadowBench command-line interface.

Thin orchestration layer over the modules — the desktop app (via IPC) calls the same functions. Keep logic in
the modules; keep this file about parsing args and rendering output.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from shadowbench import __version__
from shadowbench.common.errors import ShadowBenchError
from shadowbench.common.logging import configure_logging
from shadowbench.common.types import Task, UserProfile
from shadowbench.predictor.discovery import recommend as run_recommend
from shadowbench.profiler.detect import profile_hardware

app = typer.Typer(
    name="shadowbench",
    help="Predict LLM throughput on your hardware and pool inference across the LAN.",
    no_args_is_help=True,
    add_completion=False,
)
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
    no_stress: bool = typer.Option(False, "--no-stress", help="Skip the bandwidth stress kernel."),
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
    no_stress: bool = typer.Option(False, "--no-stress", help="Skip the bandwidth stress kernel."),
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
    contribute: bool = typer.Option(
        False, "--contribute", help="Append the measured result to datasets/golden.jsonl."
    ),
) -> None:
    """Measure real tokens/sec for a local model (Phase 2)."""
    _ = contribute
    console.print("[yellow]`bench` lands in Phase 2 (P2.4).[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", help="Bind port for the OpenAI-compatible proxy."),
) -> None:
    """Start the local Shadow Pool proxy (Phase 4)."""
    _ = port
    console.print("[yellow]`serve` lands in Phase 4 (P4.3).[/yellow]")
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
