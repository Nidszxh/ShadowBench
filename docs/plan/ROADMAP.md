# ShadowBench — Master Roadmap

This roadmap is structured so that **every phase ends with a functioning, testable increment** of the application — progressing from raw hardware math to a fully distributed peer pool.

---

## Phase 1: The Core Mathematical Hardware Profiler (Week 1)

**Objective:** Build the terminal-based engine that benchmarks local hardware and calculates memory constraints.

| Task | Description |
|---|---|
| 1.1 System Sensing Pipeline | Backend engine (Python/C++/Rust) that reads exact GPU model, total/free VRAM, system RAM capacity, and current background memory utilization |
| 1.2 Bus Bandwidth Stressor | Lightweight 3-second floating-point matrix-multiplication kernel to stress the PCIe lane and measure actual data transfer speed (GB/s), rather than relying on theoretical specs |
| 1.3 Static Model Metadata Parser | Local configuration table mapping popular open-source model topologies (Dense vs. MoE) and their quantization scales (`Q4_K_M`, `Q8_0`, etc.) |

**Exit criteria:** A CLI tool that outputs a machine's hardware profile and raw compute/bandwidth numbers.

---

## Phase 2: The Intent & Performance Predictor Engine (Week 2)

**Objective:** Build the algorithmic layer that translates a user's task and hardware specs into a specific model recommendation.

| Task | Description |
|---|---|
| 2.1 Requirement Discovery Mapping | Multi-track evaluation engine: given a user Intent (e.g., Coding) and Profile (e.g., Accuracy-First), filter the metadata table for specialized architectures (e.g., Qwen-Coder, DeepSeek-Distill) |
| 2.2 MoE-Aware Speed Estimator | Implement the throughput estimation formula; determine whether base layers fit in VRAM and apply `--n-cpu-moe`-equivalent performance penalty curves using Phase 1's measured PCIe bandwidth |
| 2.3 Config Coach Output | Output an optimization payload: exact model file string, predicted tokens/sec, and required runtime arguments (`-ub 2048`, `--threads`, `--ngl`) |

**Exit criteria:** Given a task + hardware profile, the CLI recommends a specific model, quant level, predicted t/s, and runtime flags.

---

## Phase 3: The Native Desktop Shell & Wrapper (Week 3)

**Objective:** Wrap the backend engines in a clean, production-ready desktop UI.

| Task | Description |
|---|---|
| 3.1 Tauri Architecture Setup | Set up the Tauri desktop environment; design a scannable, single-page dashboard (TypeScript/HTML5) |
| 3.2 IPC Bridge | Wire frontend actions ("Profile System," "Analyze Requirement") to the Rust sidecar so Phase 1 & 2 engines run instantly on click |
| 3.3 One-Click Engine Orchestrator | Downloader manager that fetches the target `.gguf` file via chunked streaming, saves it to a structured local directory, and spawns the local execution process |

**Exit criteria:** A working desktop app that profiles hardware, recommends a model, downloads it, and launches local inference — single machine, no networking yet.

---

## Phase 4: Local mDNS Networking & Unified API Proxy (Week 4)

**Objective:** Turn standalone instances into a collaborative local network.

| Task | Description |
|---|---|
| 4.1 Zero-Config Discovery | Integrate an mDNS background service into the Tauri backend; machines on the same Wi-Fi/LAN auto-announce hardware profile and loaded models |
| 4.2 LAN Security Tunnels | Generate ad-hoc, self-signed TLS certificates on launch; establish encrypted `wss://` WebSocket channels with strict payload-only sandboxing |
| 4.3 Smart Load-Balancing Proxy | Local server at `localhost:8080/v1/chat/completions`; evaluates the network pool and routes to the best-fit peer, streaming tokens back to the local UI |

**Exit criteria:** Two or more machines on the same network can discover each other and route inference requests between them through the local proxy.

---

## Phase 5: Resiliency, Self-Correction, & Global Matrix Sync (Week 5)

**Objective:** Harden the app against real-world edge cases and build the crowd-sourced verification loop.

| Task | Description |
|---|---|
| 5.1 Dynamic Optimizer Loop (`llama-swap` integration) | System monitor that pauses a background peer stream, steps down memory footprint, and swaps context gracefully when a provider node needs local compute |
| 5.2 Network Drop Failover Engine | Client-side proxy detects dropped peer connections, hot-swaps to the next best node in the LAN queue, and resumes token output without raising an exception |
| 5.3 Crowdsourced Feedback Sync | Embedded SQLite tracking database logs predicted vs. actual TPS on every run; syncs anonymized data to a central open-source dataset when online, to continually improve prediction accuracy |

**Exit criteria:** The system tolerates peer disconnects and thermal/resource contention gracefully, and contributes to a growing, self-improving prediction dataset.

---

## Roadmap Summary

| Phase | Focus | Deliverable |
|---|---|---|
| 1 | Hardware profiling | CLI hardware profiler |
| 2 | Prediction math | CLI model recommender |
| 3 | Desktop shell | Single-machine desktop app |
| 4 | P2P networking | Multi-machine local pool |
| 5 | Resiliency & data loop | Production-hardened, self-improving system |

> See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for granular execution steps, prerequisites, and testing strategy per phase.
