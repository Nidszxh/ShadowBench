# ShadowBench — Implementation Plan

Granular execution steps, prerequisites, testing strategy, and deployment approach, organized by roadmap phase.

## Prerequisites (Before Phase 1)

- [x] Confirm target OS support matrix for v1 (recommend: Windows + macOS + Linux — all three required for OSS credibility)
- [x] Establish a small internal library of test model metadata (Dense: Llama-3-8B, Mistral-7B, Phi-3; MoE: Qwen3.5-35B-A3B, a DeepSeek MoE variant)
- [x] Set up a repo structure separating: `/core` (Python), `/frontend` (Tauri), `/schemas`, `/datasets`, `/docs`
- [x] Decide sidecar language early — **Python chosen** for fastest iteration on math; risk of sidecar bundling cost acknowledged, gated at P3.0

## Phase 1 — Hardware Profiler: Implementation Steps

1. **GPU detection layer**
   - NVIDIA: `pynvml` + `nvidia-smi` fallback for VRAM total/free, model name, temperature, driver version. ✅ DONE
   - Intel: sysfs vendor-0x8086 detection for integrated and discrete GPUs (shared memory fraction). ✅ DONE
   - Apple Silicon: native `system_profiler` calls for unified memory reporting. ✅ DONE
   - AMD: `rocm-smi --json` + sysfs vendor-0x1002 fallback for VRAM total/free, model name. ✅ DONE
   - Fallback: graceful "unknown GPU" path with CPU-only estimation. ✅ DONE
2. **System RAM/CPU layer** — `psutil` for core count, total RAM, current utilization. ✅ DONE
3. **PCIe bandwidth stress kernel** — bounded ~3s GEMM loop; cap runtime and memory footprint. ✅ DONE
4. **GGUF metadata reader** — parse topology/quant/layers from file headers; validate against file size; fall back to catalog on malformed data. ✅ DONE
5. **Anonymized profile schema** — PII-stripping with memory bucketing. ✅ DONE

**Testing strategy:**
- Unit test the math functions against hand-computed reference values. ✅ DONE
- Validate GPU detection across at least one NVIDIA, one Apple Silicon, one AMD, and one Intel machine.
- Regression-test the stress kernel's timing bounds.
- AMD/Intel backend tests: assert graceful fallback on machines without the respective GPU.

## Phase 2 — Predictor Engine: Implementation Steps

1. **Dense throughput model** — VRAM-overflow degradation curve based on measured PCIe bandwidth. ✅ DONE
2. **MoE throughput model** — base-layer / active-expert VRAM allocation with offload routing. ✅ DONE
3. **Requirement Discovery Engine** — Task × Hardware × Profile matrix lookup. ✅ DONE
4. **Config Coach** — exact runtime flag string generation. ✅ DONE
5. **Prefill/decode-aware batch tuning** — dynamic `-ub`/`--ubatch` scaling and `--parallel` capping. ✅ DONE
6. **Model catalog auto-update** — implement a CI job (or standalone script) that:
   - Watches popular Hugging Face GGUF repos or accepts a manual trigger
   - Validates topology, quantization, param count sanity checks
   - Opens a PR with the new catalog entry
   - Blocks the PR if any field fails validation
7. **golden.jsonl** — seed dataset with initial benchmark rows. ✅ DONE

**Testing strategy:**
- Build a small "golden dataset" of known model/hardware combinations with real-world measured t/s. ✅ DONE
- Table-driven tests for the Dense vs. MoE branch selection logic. ✅ DONE
- Accuracy regression gate: CI fails if median prediction error worsens. ✅ DONE (workflow scaffolded)

## Phase 3 — Desktop Shell & Ecosystem Plumbing: Implementation Steps

### 3.0 Risk Spike: Sidecar Bundling (BLOCKING GATE)

1. Attempt to build a Python sidecar binary with PyInstaller on Windows, macOS, and Linux.
2. Attempt code signing on macOS (notarization) and Windows (Authenticode).
3. Measure resulting binary size and startup latency.
4. **If any platform fails:** convene a decision on extracting hot paths to Rust or adopting Nuitka as an alternative.

**Testing strategy:**
- Build artifact runs `shadowbench profile` and produces valid JSON output on all three platforms.
- Binary does not trigger OS security warnings on stock Windows Defender / macOS Gatekeeper.

### 3.1 PyPI Package

1. Configure `pyproject.toml` with `[project.urls]`, classifiers, and `README.md` for PyPI rendering.
2. Set up [trusted publishing](https://docs.pypi.org/trusted-publishers/) via GitHub OIDC (no PyPI tokens in CI).
3. Add a `publish.yml` workflow triggered on tag push (e.g., `v0.3.0`).
4. Verify `pip install shadowbench` works on a clean VM for all three platforms.

**Testing strategy:**
- CI dry-run publish on every PR (uses Test PyPI or `--dry-run`).
- Integration test: fresh venv, `pip install shadowbench`, run `uv run shadowbench profile`.

### 3.2 Tauri Desktop Shell

1. Initialize Tauri project in `frontend/`.
2. Design a scannable single-page dashboard (React/TypeScript).
3. Wire IPC bridge: frontend actions call Rust commands that delegate to the Python sidecar.
4. Bundle the Python sidecar (from P3.0) as a Tauri sidecar binary.
5. Implement auto-updater (Tauri's built-in mechanism).

### 3.3 Web Dashboard

1. Implement a lightweight FastAPI `--web` server in `shadowbench/ipc/server.py` alongside the existing Typer CLI.
2. Design a minimal SPA (or server-rendered templates) that surfaces: hardware profile, recommendation, download progress, run status.
3. Expose the same API endpoints that the Tauri IPC would call, so both UIs share logic.
4. Add a `shadowbench serve --web --port 8080` CLI command that opens the browser.

**Testing strategy:**
- End-to-end: `shadowbench serve --web` → browser opens → profile → recommend → download → run produces output.
- Headless test: curl the web API endpoints and assert valid JSON responses.

### 3.4 Inference Engine Abstraction

1. Define a `Runner` protocol/ABC with methods: `measure_tps(model_path, context_length, flags) -> float`, `supported_quantizations() -> list[Quantization]`.
2. Refactor the existing `GroundTruth` harness to implement the `Runner` protocol (llama.cpp backend).
3. Implement an `OllamaRunner` that wraps the Ollama API (`/api/generate`).
4. Add a `--backend` flag to `shadowbench bench` to select between runners.
5. Document how to add a new backend in `CONTRIBUTING.md`.

**Testing strategy:**
- Both runners produce consistent t/s results for the same model on the same hardware (within ±10%).
- Runner selection via CLI flag works and surfaces clear errors for unsupported backends.

### 3.5 Download Orchestrator

1. Implement pre-flight disk-space check (model size × 1.2 safety margin).
2. Implement range-request resumable download with progress callbacks.
3. Verify checksum (SHA-256) post-download; reject on mismatch.
4. Surface progress and errors to both Tauri UI and web dashboard.

### 3.6 Docker Image

1. Write multi-stage `Dockerfile`:
   - Stage 1: Python build with `uv` + `pip install shadowbench`.
   - Stage 2: minimal `python:3.12-slim` base with the installed package.
2. Publish to Docker Hub (`shadowbench/shadowbench`) and GitHub Container Registry (`ghcr.io/shadowbench/shadowbench`).
3. Multi-arch builds (linux/amd64, linux/arm64) via Docker Buildx.
4. Add `docker-compose.yml` for multi-peer setups (useful for Phase 4 testing).

**Testing strategy:**
- `docker run shadowbench/shadowbench profile` produces valid JSON.
- Image size < 200 MB (Python + deps; verify with `docker images`).
- Arm64 build verified on Apple Silicon or AWS Graviton.

### 3.7 Opt-in Telemetry Sync

1. Extend the SQLite schema with a `telemetry_queue` table for offline batching.
2. Implement the sync payload builder (already stubbed in `calibration/sync.py`) — strip all PII.
3. Write an automated privacy test: assert zero hostname/IP/user IDs leave the machine in the payload.
4. Implement exponential-backoff upload on connectivity.
5. Wire telemetry logging into both the CLI `bench` command and the desktop/web run flow.
6. **Default-off**: first-run prompt with clear disclosure of what is sent.

**Testing strategy:**
- Privacy test: craft a payload with known PII tokens, run the scrubber, assert they are absent.
- Integration test: run a benchmark, confirm a queued telemetry row appears, mock the upload endpoint.
- Outlier filtering test: submit a physically-implausible t/s value and confirm it is rejected server-side.

## Phase 4 — P2P Networking: Implementation Steps

1. **mDNS service** — background service broadcasting `{node_id, hardware_summary, available_models}`; peer table with TTL-based expiry.
   - **🆕 Manual peer-add fallback** — UI input for `host:port` when mDNS yields nothing; clear diagnostic message when multicast appears blocked.
2. **TLS bootstrap** — generate self-signed certs on first launch; trust-on-first-use with fingerprint pinning; PIN/QR pairing confirmation.
3. **Local proxy server** — OpenAI-compatible `/v1/chat/completions`; routing layer queries peer table and predictor engine to decide local-vs-remote execution.
   - Provider advertises current queue depth in mDNS heartbeat.
   - Concurrent-routing arbitration: multiple requesters routed to same peer.
4. **WebSocket streaming bridge** — consistent chunked-JSON framing regardless of source.
5. **QUIC/WebRTC fallback** — implement as a secondary transport, activated when WebSocket/TCP fails (deferred to .x release unless testing proves TCP blocking is common).

**Testing strategy:**
- Multi-machine integration test (minimum 2 physical or VM nodes) validating discovery, routing, token streaming.
- mDNS fallback test: block multicast on a test node, confirm manual peer-add works.
- Network fault injection: block WebSocket port mid-session and confirm clean error (not crash).
- Security test: confirm provider sandbox has no filesystem/env access from crafted prompt.

## Phase 5 — Resiliency & v1.0: Implementation Steps

1. **`llama-swap`-style context manager** — suspend/resume of a peer inference session when host needs local compute; verify no data loss.
2. **Failover engine** — on peer disconnect, requeue in-flight request to next best peer; mirror conversation history so context isn't lost.
3. **🆕 Circuit breaker for peer routing** — rolling latency/failure rate tracker per peer; temporary deprioritization with exponential-backoff re-probing.
4. **v1.0 hardening** — perf budgets, error taxonomy, full docs, protocol-version negotiation, reproducible builds, SBOM.

**Testing strategy:**
- Chaos testing: kill a provider node mid-stream repeatedly; assert consistent failover with no crashes.
- Load test: simulate multiple concurrent peer sessions on a single host node.
- Reproducible build test: rebuild from source at the same git tag and get a byte-identical binary (where possible).

## Phase 6 (Future) — Ecosystem Integration

Deferred until Phases 3-5 are shipping with real users.

1. **Public accuracy dashboard** — deploy a live chart on the MkDocs site showing predicted-vs-actual, median error per model family, regression history. Fed by the golden dataset.
2. **Homebrew / apt / scoop packages** — OS-native package manager formulae for the CLI.
3. **Hugging Face model catalog sync** — CI job that auto-discovers new GGUF variants matching known model families and opens catalog PRs.
4. **GitHub Actions reusable workflow** — `shadowbench/benchmark-action` for CI benchmarking.
5. **VS Code extension** — right-click Hugging Face link → predict on hardware (deferred).

## Deployment Pipeline

| Stage | Approach |
|---|---|
| Build (Python CLI) | `uv build` → `pypa/gh-action-pypi-publish` with trusted publishing |
| Build (Tauri desktop) | Tauri's cross-platform GitHub Actions matrix (Windows/macOS/Linux) |
| Build (Docker) | `docker/build-push-action` with multi-arch (amd64 + arm64) |
| Code signing | macOS notarization + Windows Authenticode in CI |
| Distribution | GitHub Releases + PyPI + Docker Hub/GHCR + (future) Homebrew/apt/scoop |
| Telemetry opt-in | Explicit, default-off opt-in prompt on first launch; clear disclosure |
| Versioning | Semantic versioning with explicit protocol-version compatibility notes |

## Suggested Milestone Demos

| End of Phase | Demo |
|---|---|
| 1 | `shadowbench profile` prints hardware profile on NVIDIA, Apple, AMD, Intel, and CPU-only machines |
| 2 | `shadowbench recommend` outputs model + flags + predicted t/s with published accuracy report |
| 3 | Desktop app profiles → recommends → downloads → runs; `pip install shadowbench`; `docker run shadowbench/shadowbench`; web dashboard at `:8080`; telemetry feeding golden dataset |
| 4 | Two laptops on same Wi-Fi share inference through the local proxy; manual peer-add works when mDNS is blocked |
| 5 | Simulated peer disconnect recovers gracefully; circuit breaker de-prioritizes flaky peers; v1.0 ships signed |
