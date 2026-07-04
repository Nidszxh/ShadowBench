# ShadowBench Datasets

Two public data assets that make ShadowBench's predictions honest and improvable.

## `models_catalog.json`

Known-model metadata (topology, parameter count, layer/head config, available quants) consumed by the
Predictor. Add a model by opening a PR that appends an entry. Fields map 1:1 to
`shadowbench.predictor.models.ModelSpec`.

## `golden.jsonl`

The **accuracy ground truth**: one JSON object per line, each a real (hardware × model × quant × context) →
measured tokens/sec measurement. The Predictor is scored against this file in CI (`calibration/report.py`), and
CI fails if median error regresses past the target band.

### Row schema

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Currently `1` |
| `gpu_name` | string \| null | `null` for CPU-only |
| `vram_total_mb` | int | `0` for CPU-only |
| `system_ram_gb` | int | |
| `cpu_matmul_gbps` | float | CPU matmul benchmark (reserved for future multi-GPU topology) |
| `model_id` | string | Must match a `models_catalog.json` id |
| `quantization` | string | e.g. `Q4_K_M` |
| `context_length` | int | |
| `measured_tps` | float | Real tokens/sec from `llama-bench` |
| `source` | string | Attribution (e.g. `github:@you`) |
| `notes` | string | Optional context |

### Contributing a measurement

```bash
uv run shadowbench bench --contribute    # (Phase 2) runs llama-bench and appends a row
```

Then open a PR. **No PII** — never include hostname, IP, or username. See `SECURITY.md`.
