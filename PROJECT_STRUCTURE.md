# ShadowBench — Project Structure

A **component-wise monorepo**. The guiding rule: the three architectural modules from
[`ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md) — **Profiler**, **Predictor**, **Shadow Pool** — map to three
self-contained Python packages with narrow, typed public interfaces. You can build, test, and reason about
each in isolation.

## Top-level layout

```
ShadowBench/
├── core/            # Python sidecar — the brain (Profiler, Predictor, Pool math, IPC)
├── frontend/        # Tauri desktop shell (Rust core + React/TS UI)  [Phase 3]
├── datasets/        # Public model catalog + accuracy golden dataset
├── schemas/         # Cross-language contracts (IPC, DB) shared by core + frontend
├── docs/            # MkDocs Material site (published to GitHub Pages)
├── scripts/         # Dev bootstrap & maintenance scripts
├── .github/         # CI/CD, issue/PR templates, CODEOWNERS, Dependabot
└── *.md             # README, ARCHITECTURE, DATAFLOW, ROADMAP, MILESTONES, this file
```

## Design principles

1. **Module = package = clear boundary.** `profiler/`, `predictor/`, `pool/` never import each other's
   internals — they exchange plain typed models defined in `common/types.py` (and each module's `models.py`).
2. **Interfaces before implementations.** Hardware/GPU access sits behind an abstract `GpuBackend` so NVIDIA,
   Apple, AMD, and CPU-only paths are swappable and independently testable.
3. **Pure math is dependency-free.** The predictor formulas take dataclasses in and return dataclasses out —
   no I/O, no hardware calls — so they're trivially unit-testable against the golden dataset.
4. **The CLI and the IPC server are thin.** They orchestrate; all logic lives in the modules. The desktop app
   and the CLI call the *same* functions.

## `core/` — Python sidecar (detailed)

```
core/
├── pyproject.toml                 # PEP 621 metadata, deps, ruff/mypy/pytest config
├── README.md
├── src/shadowbench/
│   ├── __init__.py                # version, PROTOCOL_VERSION
│   ├── __main__.py                # `python -m shadowbench` → cli
│   ├── cli.py                     # Typer CLI: `profile`, `recommend`, `serve`, `bench`
│   │
│   ├── common/                    # shared, dependency-light primitives
│   │   ├── config.py              # settings, paths (platformdirs), env overrides
│   │   ├── logging.py             # structured logging setup
│   │   ├── errors.py              # ShadowBenchError hierarchy (typed, user-facing)
│   │   └── types.py               # cross-module shared models (HardwareProfile, ...)
│   │
│   ├── profiler/                  # ── MODULE 1: Hardware Profiler ──
│   │   ├── models.py              # GpuInfo, SystemInfo, BandwidthResult, HardwareProfile
│   │   ├── detect.py              # orchestrator: assemble a full HardwareProfile
│   │   ├── system.py              # RAM/CPU via psutil
│   │   ├── bandwidth.py           # bounded PCIe/compute stress kernel (GB/s, TFLOPS)
│   │   ├── gguf.py                # GGUF header parser (topology, quant, layers)
│   │   └── gpu/                   # swappable per-vendor backends
    │   │       ├── base.py            # GpuBackend ABC + registry/auto-select
    │   │       ├── nvidia.py          # pynvml + nvidia-smi fallback
    │   │       ├── intel.py           # sysfs (vendor 0x8086) — integrated + discrete
    │   │       ├── apple.py           # system_profiler (unified memory)
    │   │       ├── amd.py             # ROCm SMI + sysfs fallback
    │   │       └── cpu.py             # graceful CPU-only fallback
│   │
│   ├── predictor/                 # ── MODULE 2: Predictor Engine ──
│   │   ├── models.py              # Prediction, Recommendation, RuntimeFlags
│   │   ├── memory.py              # dense footprint + KV-cache formulas (DATAFLOW §1.1–1.2)
│   │   ├── dense.py               # dense throughput + VRAM-overflow curve
│   │   ├── moe.py                 # MoE base/expert split + offload throughput (§1.3–1.4)
│   │   ├── config_coach.py        # exact runtime-flag builder (§1.5–1.6)
│   │   ├── discovery.py           # Requirement Discovery: Task×Hardware×Profile → ranked
│   │   └── catalog.py             # loads datasets/models_catalog.json
│   │
│   ├── pool/                      # ── MODULE 3: Shadow Pool (P2P) ──  [Phase 4]
│   │   ├── discovery.py           # mDNS/zeroconf advertise + peer table (TTL)
│   │   ├── transport.py           # TLS WebSocket streaming bridge
│   │   ├── proxy.py               # OpenAI-compatible /v1/chat/completions
│   │   ├── router.py              # local-vs-peer routing + failover
│   │   └── security.py            # self-signed certs, pairing (PIN/QR), sandbox
│   │
│   ├── orchestrator/              # download + run local engines  [Phase 3]
│   │   ├── downloader.py          # resumable chunked GGUF fetch + checksum
│   │   └── runner.py              # spawn/manage llama.cpp / Ollama
│   │
│   ├── storage/                   # local datastore + calibration data  [Phase 3]
│   │   ├── db.py                  # SQLite access layer
│   │   └── schema.sql             # hardware_profiles, benchmark_runs (DATAFLOW §5)
│   │
│   ├── calibration/               # accuracy self-correction loop  [Phase 2/5]
│   │   ├── harness.py             # ground truth via llama-bench
│   │   ├── report.py              # predicted-vs-actual report (CI artifact)
│   │   └── sync.py                # opt-in, PII-stripped telemetry upload
│   │
│   └── ipc/                       # Tauri bridge  [Phase 3]
│       └── server.py              # JSON-RPC over stdio: profile_system/analyze_requirement/run_model
│
└── tests/
    └── unit/                      # pure-math + parser tests (fast, no hardware)
```

## `frontend/` — Tauri shell (Phase 3, scaffolded later)

```
frontend/
├── package.json
├── src/                    # React/TS UI: views/, components/, api/ (IPC client), styles/
└── src-tauri/             # Rust: main.rs, commands.rs, sidecar.rs (manages Python), updater.rs
    ├── Cargo.toml
    └── tauri.conf.json
```

## `datasets/` — public data assets

```
datasets/
├── models_catalog.json    # known model families: topology (dense/moe), params, quants
├── golden.jsonl           # (hardware × model × quant × context) → measured tokens/sec
└── README.md              # schema + contribution guide
```

## `schemas/` — cross-language contracts

```
schemas/
└── ipc/ipc.schema.json    # the frontend↔core IPC command/response contract (versioned)
```

## How a request flows through the components

```
CLI / Tauri UI
   │  profile_system()
   ▼
profiler.detect ──► HardwareProfile ─────────────┐
                                                  ▼
                       predictor.discovery ──► predictor.dense / predictor.moe
                                                  │        (uses predictor.memory)
                                                  ▼
                       predictor.config_coach ──► Recommendation (model, quant, flags, t/s)
                                                  │
   one-click ◄────────────────────────────────────┘
   │
   ▼
orchestrator.downloader ──► orchestrator.runner ──► local llama.cpp / Ollama
   │                                                        │
   │  (optional) pool.proxy routes to a faster peer  ◄──────┘
   ▼
storage.db logs predicted-vs-actual ──► calibration.report / calibration.sync
```

## Naming & convention rules

- **One responsibility per file.** If a file needs a section comment to separate concerns, split it.
- **`models.py` per module** holds that module's public dataclasses; truly shared types go to `common/types.py`.
- **No hardware or network I/O in `predictor/`.** Keeps the math pure and CI-testable everywhere.
- **Every public function is typed and docstringed**, referencing the `DATAFLOW.md` section it implements.
- **Tests mirror the source tree** (`tests/unit/test_moe.py` ↔ `predictor/moe.py`).
