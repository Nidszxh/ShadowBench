# ShadowBench — Master Roadmap

This roadmap is structured so that **every phase ends with a functioning, testable increment** of the application — progressing from raw hardware math to a fully distributed peer pool.

---

## Phase 0: Project Foundation & Governance (COMPLETE)

**Objective:** Scaffold the monorepo, CI, and community files so every phase ships on a production-grade foundation.

| Task | Description |
|---|---|
| 0.1 Repo Scaffold | Monorepo layout (`core/`, `frontend/`, `schemas/`, `datasets/`, `docs/`), `uv` + `pyproject.toml`, `ruff` + `mypy`, pre-commit |
| 0.2 Governance & Community | `LICENSE` (Apache-2.0), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `GOVERNANCE.md`, issue/PR templates, `CODEOWNERS` |
| 0.3 CI/CD | GitHub Actions matrix (Windows/macOS/Linux): lint, typecheck, test, coverage gate (`--cov-fail-under=60`). Dependabot, branch protection |
| 0.4 Docs Site | MkDocs Material auto-deployed to GitHub Pages |
| 0.5 Release Skeleton | `CHANGELOG.md`, `PROTOCOL_VERSION` constant, semantic versioning convention |

**Exit criteria:** A repo anyone can clone, run `make check` on, and get green on all three platforms.

---

## Phase 1: The Core Mathematical Hardware Profiler (COMPLETE)

**Objective:** Build the terminal-based engine that benchmarks local hardware and calculates memory constraints.

| Task | Description |
|---|---|
| 1.1 System Sensing Pipeline | Backend engine that reads exact GPU model, total/free VRAM, system RAM capacity, and current background memory utilization |
| 1.2 Bus Bandwidth Stressor | Lightweight 3-second GEMM kernel to measure PCIe transfer speed (GB/s) and compute throughput (TFLOPS) |
| 1.3 GGUF Metadata Reader | Parse model topology (Dense vs. MoE), parameter count, quantization from GGUF file headers — never hardcode |
| 1.4 GPU Backend Implementation | NVIDIA (`pynvml` + `nvidia-smi` fallback), Intel (sysfs), Apple Silicon (`system_profiler`), AMD (ROCm SMI + sysfs), CPU-only fallback |
| 1.5 Anonymized Profile Schema | PII-stripping with memory bucketing, designed from the start for safe public sync |

> **✅ Complete:** AMD ROCm SMI detection, Intel sysfs detection, and NVIDIA ``nvidia-smi`` fallback — all three are implemented and working.

**Exit criteria:** A CLI tool (`uv run shadowbench profile`) that outputs a machine's hardware profile and raw compute/bandwidth numbers.

---

## Phase 2: The Intent & Performance Predictor Engine (COMPLETE)

**Objective:** Build the algorithmic layer that translates a user's task and hardware specs into a specific model recommendation.

| Task | Description |
|---|---|
| 2.1 Dense + MoE Throughput Models | VRAM-resident vs. CPU-offload estimation for both topologies, plus precise KV-cache formula |
| 2.2 Config Coach | Exact runtime flags (`--ngl`, `--n-cpu-moe`, `--ubatch`, `--parallel`, `--flash-attn`, KV cache quantization) |
| 2.3 Requirement Discovery | Task × Hardware × User Profile → ranked model candidates |
| 2.4 Ground-Truth Harness | Wraps `llama-bench` to record real t/s; every run can contribute to the golden dataset |
| 2.5 Public Accuracy Dataset | Versioned `datasets/golden.jsonl` with seed rows; CI-published accuracy report |
| 2.6 Model Catalog | `datasets/models_catalog.json` with 6+ model entries; loader in `predictor/catalog.py` |

> **⚠️ Pending:** The **model catalog auto-update** mechanism (pulling new models from Hugging Face or accepting community PRs with CI validation) is not yet built. Without it the catalog will stagnate.

**Exit criteria:** A CLI tool (`uv run shadowbench recommend --task coding --profile intelligence`) that recommends a model, quantization, predicted t/s, and runtime flags, backed by a published accuracy report.

---

## Phase 3: Desktop Shell, Pipelines & Ecosystem Plumbing (Week 3)

**Objective:** Deliver a usable desktop app *and* the ecosystem infrastructure that makes ShadowBench easy to install, configure, and extend.

| Task | Description |
|---|---|
| **3.0 Spike: Python sidecar bundling** | Verify PyInstaller (or Nuitka) can produce a signed, runnable sidecar binary on Windows, macOS, and Linux **before** any Tauri work. If this fails the stack decision must be revisited. |
| **3.1 PyPI package** | Ship `shadowbench` to PyPI — `pip install shadowbench` must work for the CLI. Highest-leverage adoption item. |
| **3.2 Tauri Architecture Setup** | Set up the Tauri desktop environment; design a scannable, single-page dashboard (TypeScript/HTML5) |
| **3.3 Web Dashboard** | Ship a `shadowbench serve --web` command that opens a browser-based UI (profiling → recommend → download → run). Works on headless servers, no desktop app required. Shares the same Python backend as Tauri. |
| **3.4 IPC Bridge** | Wire frontend actions to the Rust sidecar so all modules run instantly on click |
| **3.5 Inference Engine Abstraction** | Abstract the `llama-bench`/`llama.cpp` dependency behind a `Runner` interface. Add Ollama support as a second backend. Makes the tool engine-agnostic and unlocks the user's preferred runtime. |
| **3.6 One-Click Engine Orchestrator** | Downloader manager that fetches the target `.gguf` file via chunked streaming, checksums post-download, checks free disk space before starting, and spawns the local execution process |
| **3.7 Docker Image** | `docker run shadowbench/shadowbench` for instant try-it-out. Critical for Phase 4 multi-peer testing. Publish to Docker Hub alongside GitHub Releases. |
| **3.8 Opt-in Telemetry Sync** | Log predicted-vs-actual t/s on every run; sync anonymized records to the shared dataset when online. **Moved from Phase 5** — the accuracy dataset needs to grow *during* Phase 3, not after. Default-off, privacy-verified. |

**Exit criteria:** A working desktop app (profile → recommend → download → run) AND `pip install shadowbench` AND `docker run shadowbench/shadowbench` AND a web dashboard, all with telemetry feeding the golden dataset.

---

## Phase 4: Local mDNS Networking & Unified API Proxy (Week 4)

**Objective:** Turn standalone instances into a collaborative local network.

| Task | Description |
|---|---|
| 4.1 Zero-Config Discovery | mDNS background service; peers auto-announce hardware profile and loaded models; manual peer-add fallback for AP client-isolation environments |
| 4.2 LAN Security Tunnels | Self-signed TLS on launch; encrypted `wss://` WebSocket channels; PIN/QR pairing confirmation (blocks mDNS spoofing); TLS fingerprint pinning on first accept |
| 4.3 Smart Load-Balancing Proxy | `localhost:8080/v1/chat/completions`; evaluates pool state and routes to best-fit peer, streaming tokens back |
| 4.4 WebSocket Streaming Bridge | Consistent chunked-JSON framing regardless of source peer |

**Exit criteria:** Two or more machines on the same network can discover each other and route inference requests between them through the local proxy.

---

## Phase 5: Resiliency, Self-Correction & v1.0 (Week 5)

**Objective:** Harden the app against real-world edge cases and prepare the v1.0 release.

| Task | Description |
|---|---|
| 5.1 Dynamic Optimizer Loop | System monitor that pauses a background peer stream, steps down memory footprint, and swaps context gracefully when a provider node needs local compute |
| 5.2 Network Drop Failover Engine | Client-side proxy detects dropped peer connections, hot-swaps to the next best node in the LAN queue, and resumes token output without raising an exception |
| 5.3 Circuit Breaker for Flaky Peers | Rolling latency/failure rate tracking per peer; temporary deprioritization with exponential-backoff re-probing |
| 5.4 v1.0 Hardening | Performance budgets, error taxonomy, full documentation, protocol-version negotiation, reproducible builds, SBOM |

> ℹ️ **Note:** Telemetry sync moved to Phase 3 (P3.8). Phase 5 focuses on resiliency and the v1.0 release gate.

**Exit criteria:** The system tolerates peer disconnects and thermal/resource contention gracefully. Signed, documented, community-ready v1.0 release.

---

## Phase 6 (Future): Ecosystem Integration

**Objective:** Embed ShadowBench into developer workflows beyond the desktop app.

| Task | Description |
|---|---|
| 6.1 Accuracy Dashboard on Docs Site | Live predicted-vs-actual chart from `golden.jsonl`; median error per model family; CI-failing regression gate |
| 6.2 Homebrew / apt / scoop Packages | OS-native package managers for `shadowbench` CLI |
| 6.3 Hugging Face Catalog Sync | CI job that watches popular GGUF repos and auto-suggests catalog additions |
| 6.4 GitHub Actions for CI Benchmarking | Reusable action to benchmark a model in CI and compare against golden dataset |
| 6.5 VS Code Extension (Stretch) | Right-click a Hugging Face model → "Predict on my hardware" |

> These are deferred until Phases 3–5 are shipping and the project has real users providing feedback.

---

## Roadmap Summary

| Phase | Focus | Deliverable | Status |
|---|---|---|---|
| 0 | Project Foundation | Repo scaffold, CI green, governance | COMPLETE |
| 1 | Hardware profiling | CLI hardware profiler | COMPLETE |
| 2 | Prediction math | CLI model recommender + accuracy dataset | COMPLETE (catalog auto-update pending) |
| 3 | Desktop shell & ecosystem plumbing | Desktop app + web UI + PyPI + Docker + telemetry | NOT STARTED |
| 4 | P2P networking | Multi-machine local pool | NOT STARTED |
| 5 | Resiliency & v1.0 | Production-hardened release | NOT STARTED |
| 6 | Ecosystem integration | Package managers, CI actions, dashboards | DEFERRED |

> See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for granular execution steps, prerequisites, and testing strategy per phase.
