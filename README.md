# ShadowBench

**A crowd-sourced, peer-to-peer benchmarking and local inference pooling tool for open-source AI models.**

---

## 1. Overview

ShadowBench solves a common pain point for developers working with open-source LLMs: figuring out whether a model will actually run well on a given machine *before* downloading a multi-gigabyte file, and unlocking extra performance by pooling compute across nearby machines on the same network.

It has two core capabilities:

1. **The Benchmarker** — profiles local hardware (CPU, GPU, RAM, VRAM, PCIe bandwidth) and predicts token-per-second throughput for a target model and quantization level, without requiring the model to be downloaded first.
2. **The Shadow Pool** — a zero-configuration, peer-to-peer network that lets nearby machines (e.g., teammates on the same Wi-Fi/LAN during a hackathon) share inference compute through a single, OpenAI-compatible local API endpoint.

## 2. Problem Statement

Running open-source models (Llama 3, Mistral, Phi, Qwen, DeepSeek, etc.) on consumer hardware is trial-and-error. Developers routinely:

- Download 10–20GB model files only to discover the model is unusably slow on their machine.
- Have no visibility into how quantization, VRAM limits, or PCIe bottlenecks will affect throughput.
- Sit next to teammates with idle, more powerful GPUs with no easy way to borrow that compute.

## 3. Goals

- Predict real-world tokens/sec for a given model + quantization + hardware combination **before download**, using live micro-benchmarks rather than static specs.
- Correctly model both **Dense** and **Mixture-of-Experts (MoE)** architectures, which have fundamentally different memory and throughput characteristics.
- Recommend the optimal model, quantization, and runtime flags for a user's stated task (e.g., "coding," "general chat") and hardware profile.
- Enable secure, zero-configuration LAN-based compute pooling between peers, exposed through a standard OpenAI-compatible API.
- Continuously improve prediction accuracy via anonymized, crowd-sourced telemetry (predicted vs. actual t/s).

## 4. Non-Goals

- ShadowBench does not implement its own inference engine — it orchestrates existing engines (Ollama, llama.cpp).
- ShadowBench does not coordinate over the public internet; the Shadow Pool is LAN/Wi-Fi scoped only.

## 5. Core Modules

| Module | Responsibility |
|---|---|
| **Profiler** | Detects CPU, GPU, RAM, and VRAM; runs a brief compute/bandwidth stress test |
| **Predictor** | Applies Dense vs. MoE-aware math to estimate throughput and recommend runtime flags |
| **Shadow Pool** | mDNS peer discovery, encrypted WebSocket transport, and a local load-balancing proxy |

See [`ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md) for system design and [`DATAFLOW.md`](docs/plan/DATAFLOW.md) for how requests move through the system.

## 6. Recommended Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Desktop shell | **Tauri** (Rust core + HTML/TS frontend) | Sub-20MB binaries, native system access, lighter than Electron |
| Core logic | Rust/Go, or Python sidecar | Hardware profiling, math engine, mDNS discovery |
| Inference backend | **Ollama** / **llama.cpp** | Mature, fast local inference — no need to build a runner from scratch |
| P2P networking | WebSockets (TLS), with QUIC/WebRTC fallback | Low-latency streaming with resilience to flaky LAN routers |
| Local storage | SQLite (embedded) | Stores predicted-vs-actual benchmark data for model self-correction |

## 7. Quick Start (Target Developer Workflow)

> These steps describe the intended end-state developer experience once Phase 3 (desktop shell) is complete. See `IMPLEMENTATION_PLAN.md` for current build steps.

1. **Install** the ShadowBench desktop app (Tauri build for Windows/macOS/Linux).
2. **Launch** the app — it automatically profiles your CPU, GPU, RAM, and runs a short (~5 second) stress test.
3. **Select intent** — choose a task (e.g., "Coding," "General Chat") and a profile (Speed-first vs. Intelligence-first).
4. **Review the recommendation** — ShadowBench shows the best-fit model, quantization, predicted tok/s, and the exact runtime flags it will use.
5. **One-click run** — ShadowBench downloads the `.gguf` file, spawns the local inference engine with optimized flags, and exposes it at `http://localhost:8080/v1/chat/completions`.
6. **(Optional) Join a pool** — on the same Wi-Fi/LAN, teammates running ShadowBench auto-discover each other via mDNS. Enable the local proxy to transparently offload inference to a faster peer when your own hardware is the bottleneck.

## 8. Documentation Index

- [`ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md) — system components and how they're organized
- [`DATAFLOW.md`](docs/plan/DATAFLOW.md) — request/response lifecycle, P2P protocol, database schema
- [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md) — the component-wise monorepo layout
- [`MILESTONES.md`](docs/plan/MILESTONES.md) — production open-source delivery plan (v0.0.1 → v1.0.0)
- [`ROADMAP.md`](docs/plan/ROADMAP.md) — original phased feature plan
- [`IMPLEMENTATION_PLAN.md`](docs/plan/IMPLEMENTATION_PLAN.md) — granular build steps, testing, deployment
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`SECURITY.md`](./SECURITY.md) · [`GOVERNANCE.md`](./GOVERNANCE.md)

## 9. Building From Source (current state)

Phase 0 scaffolding is in place: the Python core (`core/`) profiles hardware and recommends models today.

```bash
scripts/bootstrap.sh              # installs deps + hooks, runs checks
# or manually:
cd core && uv sync --all-extras
uv run shadowbench profile        # print this machine's hardware profile
uv run shadowbench recommend --task coding --profile intelligence
uv run shadowbench catalog validate                    # validate model catalog
uv run shadowbench catalog add Qwen/Qwen2.5-7B         # auto-fetch a new model entry
```
