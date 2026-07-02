# shadowbench (core)

The Python sidecar that powers ShadowBench — hardware profiling, throughput prediction, and (later) the
peer-to-peer inference pool. It runs standalone as a CLI and is embedded by the Tauri desktop app as a sidecar
process.

See [`../PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) for the package layout and
[`../DATAFLOW.md`](../DATAFLOW.md) for the math each module implements.

## Install (development)

```bash
uv sync --all-extras          # core + gpu + pool + dev deps
uv run shadowbench --help
```

## CLI

```bash
shadowbench profile                       # detect hardware, print a HardwareProfile as JSON
shadowbench recommend --task coding \      # recommend a model + quant + runtime flags
    --profile intelligence
shadowbench bench --contribute             # measure real tokens/sec, append to the golden dataset
shadowbench serve                          # start the local OpenAI-compatible proxy (Phase 4)
```

## Layout

| Package | Module |
|---|---|
| `profiler/` | Hardware detection + PCIe/compute stress kernel + GGUF parsing |
| `predictor/` | Dense/MoE throughput math, Config Coach, Requirement Discovery |
| `pool/` | mDNS discovery, TLS transport, OpenAI-compatible proxy (Phase 4) |
| `orchestrator/` | Model download + local engine process management (Phase 3) |
| `storage/` | SQLite datastore for predicted-vs-actual calibration (Phase 3) |
| `calibration/` | Ground-truth harness, accuracy report, opt-in telemetry sync |
| `ipc/` | JSON-RPC-over-stdio bridge for the Tauri frontend (Phase 3) |
| `common/` | Shared config, logging, typed errors, cross-module models |
