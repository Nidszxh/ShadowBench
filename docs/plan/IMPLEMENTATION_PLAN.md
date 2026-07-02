# ShadowBench — Implementation Plan

Granular execution steps, prerequisites, testing strategy, and deployment approach, organized by roadmap phase.

## Prerequisites (Before Phase 1)

- [ ] Confirm target OS support matrix for v1 (recommend: Windows + macOS first, Linux stretch goal — NVIDIA/Apple Silicon paths differ significantly)
- [ ] Establish a small internal library of test model metadata (Dense: Llama-3-8B, Mistral-7B, Phi-3; MoE: Qwen3.5-35B-A3B, a DeepSeek MoE variant)
- [ ] Set up a repo structure separating: `/core` (Rust/Python sidecar logic), `/frontend` (Tauri UI), `/schemas` (SQLite migrations)
- [ ] Decide sidecar language early (Python for fastest iteration on the math/ML side vs. native Rust for perf and simpler packaging) — this affects the IPC contract in Phase 3

## Phase 1 — Hardware Profiler: Implementation Steps

1. **GPU detection layer**
   - NVIDIA: `pynvml` bindings for VRAM total/free, model name, temperature, clock state.
   - Apple Silicon: native Metal/`system_profiler` calls for unified memory reporting.
   - AMD: ROCm SMI or platform-equivalent bindings.
   - Fallback: graceful "unknown GPU" path with CPU-only estimation.
2. **System RAM/CPU layer** — `psutil` (or Rust `sysinfo` crate) for core count, total RAM, current utilization.
3. **PCIe bandwidth stress kernel** — implement a bounded-duration (~3s) matrix multiplication that forces host↔device memory transfer, timed to compute observed GB/s. Cap runtime and memory footprint to keep it "non-invasive."
4. **Model metadata table** — local JSON/SQLite table of known model families, tagged `dense` or `moe`, with parameter counts and available quantization levels. Read directly from GGUF header metadata where possible instead of hardcoding, to avoid staleness.

**Testing strategy:**
- Unit test the math functions (memory formulas) against hand-computed reference values.
- Validate GPU detection across at least one NVIDIA, one Apple Silicon, and one AMD (or CPU-only) machine.
- Regression-test the stress kernel's timing bounds so it never runs longer than its stated budget under load.

## Phase 2 — Predictor Engine: Implementation Steps

1. **Dense throughput model** — implement the VRAM-overflow degradation curve based on measured PCIe bandwidth from Phase 1.
2. **MoE throughput model** — implement the base-layer / active-expert VRAM allocation logic and the throughput equation (see `DATAFLOW.md §1.3–1.4`).
3. **Requirement Discovery Engine** — build the Task × Hardware × Profile matrix lookup; start with a small hardcoded candidate list per task category, designed to be swapped for a queryable index later.
4. **Config Coach** — generate the exact runtime flag string (`--n-cpu-moe`, `-ub`, `--ngl`, `--parallel`) as a function of predicted bottleneck.
5. **Prefill/decode-aware batch tuning** — implement the dynamic `-ub`/`--ubatch` scaling and `--parallel` capping logic described in `DATAFLOW.md §1.6`.

**Testing strategy:**
- Build a small "golden dataset" of known model/hardware combinations with real-world measured t/s (from personal testing or public benchmarks) and assert predictions fall within an acceptable error band (e.g., ±20%).
- Table-driven tests for the Dense vs. MoE branch selection logic.

## Phase 3 — Desktop Shell: Implementation Steps

1. **Tauri scaffolding** — initialize project, wire up TypeScript/React (or Vue) frontend shell.
2. **IPC contract** — define a stable command interface between frontend and Rust core (`profile_system()`, `analyze_requirement(intent, profile)`, `run_model(config)`), versioned so future protocol changes don't break the UI.
3. **Chunked download manager** — implement resumable, chunked `.gguf` downloads with progress reporting back to the UI; verify checksums post-download.
4. **Process orchestration** — spawn/manage the local `llama.cpp`/Ollama process with the Config Coach's generated flags; capture stdout/stderr for error surfacing in the UI.

**Testing strategy:**
- End-to-end test: profile → recommend → download (small test model) → launch → verify local endpoint responds.
- Interrupt/resume testing for the download manager (kill connection mid-download, confirm resume works).
- Process-crash handling: verify UI surfaces a clear error if the spawned inference process exits unexpectedly.

## Phase 4 — P2P Networking: Implementation Steps

1. **mDNS service** — background service broadcasting `{node_id, hardware_summary, available_models}`; listens for peer broadcasts and maintains a local peer table with TTL-based expiry (to drop stale/disconnected peers).
2. **TLS bootstrap** — generate self-signed certs on first launch; establish a simple trust-on-first-use model between discovered peers (see Architect's Review for hardening notes).
3. **Local proxy server** — implement the OpenAI-compatible `/v1/chat/completions` endpoint; add a routing layer that queries the peer table and predictor engine to decide local-vs-remote execution.
4. **WebSocket streaming bridge** — proxy streams tokens from the executing node (local or remote) back to the caller using consistent chunked-JSON framing regardless of source.
5. **QUIC/WebRTC fallback** — implement as a secondary transport, activated when WebSocket/TCP connection setup fails or degrades.

**Testing strategy:**
- Multi-machine integration test (minimum 2 physical or VM nodes) validating discovery, routing, and token streaming end-to-end.
- Network fault injection: block the WebSocket port mid-session and confirm fallback transport engages.
- Security test: confirm a provider node's sandboxed inference context has no filesystem/env access from a crafted prompt payload.

## Phase 5 — Resiliency & Data Loop: Implementation Steps

1. **`llama-swap`-style context manager** — implement suspend/resume of a peer inference session when the host needs local compute; verify no data loss in the paused conversation state.
2. **Failover engine** — on peer disconnect, requeue the in-flight request to the next best peer and mirror conversation history so context isn't lost mid-stream.
3. **SQLite schema + migrations** — implement `hardware_profiles` and `benchmark_runs` tables (see `DATAFLOW.md §5`); wire up logging on every completed run.
4. **Sync client** — background job that batches and uploads anonymized benchmark records when internet connectivity is available; must be opt-in and clearly disclosed to the user.

**Testing strategy:**
- Chaos testing: kill a provider node mid-stream repeatedly and confirm consistent failover behavior with no crashes.
- Data integrity test: confirm sync payloads contain no identifying information (hostname, IP, user identifiers) before upload.
- Load test: simulate multiple concurrent peer sessions on a single host node to validate `--parallel` and swap logic under contention.

## Deployment Pipeline

| Stage | Approach |
|---|---|
| Build | Tauri's cross-platform build pipeline (GitHub Actions matrix: Windows/macOS/Linux runners) |
| Code signing | Required for macOS (notarization) and Windows (Authenticode) to avoid OS-level security warnings on a peer-networking app |
| Distribution | GitHub Releases for early access; consider a lightweight auto-updater (Tauri's built-in updater) given the app touches networking and security-sensitive code paths |
| Telemetry opt-in | Explicit, default-off opt-in prompt on first launch for the crowdsourced benchmark sync (Phase 5.3) |
| Versioning | Semantic versioning with explicit compatibility notes for the P2P protocol version (peers on mismatched protocol versions should fail discovery gracefully, not crash) |

## Suggested Milestone Demos

| End of Phase | Demo |
|---|---|
| 1 | CLI prints hardware profile + measured PCIe bandwidth |
| 2 | CLI recommends a model + flags for a given task/hardware combo |
| 3 | Desktop app profiles → recommends → downloads → runs a model locally |
| 4 | Two laptops on the same Wi-Fi share inference load through the local proxy |
| 5 | Simulated peer disconnect recovers gracefully; benchmark data syncs to a shared dataset |
