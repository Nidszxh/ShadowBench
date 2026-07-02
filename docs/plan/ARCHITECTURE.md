# ShadowBench — System Architecture

## 1. Architectural Style

ShadowBench is a **decoupled, single-binary desktop application** (Tauri) with three internally decoupled backend modules — Profiler, Predictor, and Shadow Pool — coordinated by a Rust core and exposed to a web-based frontend via IPC. Peer-to-peer behavior is handled entirely at the LAN layer, with no external coordination server.

## 2. High-Level Component Diagram

```
                              +─────────────────────────────+
                              |      FRONTEND DASHBOARD      |
                              |  (HTML5 / TypeScript / React) |
                              +───────────────┬───────────────+
                                              │ IPC Bridge
                                              ▼
                              +─────────────────────────────+
                              |   TAURI BACKEND CORE (Rust)   |
                              +─────────────────────────────+
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
  +─────────────────────+       +──────────────────────+        +─────────────────────+
  |  MODULE 1: PROFILER  |       |  MODULE 2: PREDICTOR |        |  MODULE 3: SHADOW    |
  |                       |       |                      |        |  POOL (P2P)          |
  +─────────────────────+       +──────────────────────+        +─────────────────────+
  | pynvml / psutil       |       | Dense bottleneck math|        | mDNS / Bonjour       |
  | Memory bandwidth test |       | MoE active-subset math|       | OpenAI-compatible    |
  | Compute (TFLOPS) test |       | Runtime flag builder |        |   local proxy         |
  +───────────┬───────────+       +──────────┬───────────+        | WebSocket streaming   |
              │                              │                    +──────────┬───────────+
              ▼                              ▼                               ▼
      +───────────────────+                                       +───────────────────+
      |   LOCAL HARDWARE   |                                       |  LOCAL LAN / WI-FI  |
      |  (CPU / GPU / RAM) |                                       |      NETWORK        |
      +───────────────────+                                       +─────────┬───────────+
                                                                              ▼
                                                                    +───────────────────+
                                                                    |    PEER NODES      |
                                                                    | (Friend's Laptop)  |
                                                                    +───────────────────+
```

## 3. Module Breakdown

### 3.1 Module 1 — Hardware Profiler

Runs a non-invasive scan in under 5 seconds, avoiding the need to download a model just to test it.

- **VRAM & System RAM detection**: `pynvml` (NVIDIA), native system APIs for Apple Silicon/AMD, `psutil` for general system RAM/CPU.
- **Bus bandwidth stress test**: a short (3–5 second) matrix-multiplication kernel measures real PCIe transfer speed (GB/s) and compute throughput (TFLOPS) — this is critical for predicting the performance cliff when a model spills from VRAM into system RAM.
- **Thermal awareness**: monitors GPU temperature/clock state so sustained pooling sessions can detect and react to thermal throttling.

### 3.2 Module 2 — Predictor Engine

The predictor cannot treat all models the same — it must branch based on model topology.

- **Dense model path**: total weight size is checked against available VRAM. If it doesn't fit, throughput degrades sharply due to PCIe-bound fallback to system RAM.
- **MoE model path**: only a fraction of parameters ("active experts") are used per token. The predictor computes how much of the base (non-expert) architecture and active expert set fits in VRAM, then estimates throughput using an offloading model equivalent to llama.cpp's `--n-cpu-moe` behavior.
- **Config Coach**: rather than a binary "yes/no," the predictor outputs the exact recommended runtime flags (e.g., `--n-cpu-moe`, `-ub`, `--ngl`, `--parallel`) tuned to the detected hardware.
- **Requirement Discovery Engine**: maps a 3-variable input matrix — **Task** (coding, chat, reasoning), **Hardware threshold** (VRAM + RAM), and **User profile** (speed-first vs. intelligence-first) — to a ranked model recommendation.

```
[ User Spec: "Coding Assignment" + "High Accuracy" ]
                        │
                        ▼
+─────────────────────────────────────────────────────────+
|              REQUIREMENT DISCOVERY ENGINE                |
+─────────────────────────────────────────────────────────+
| 1. Parse Intent   → Filter to coder/reasoning models     |
| 2. Assess Memory  → Combined VRAM + system RAM capacity  |
| 3. Calculate Max  → Highest viable quantization level    |
+──────────────────────────┬────────────────────────────────+
                            ▼
        [ Recommendation: Qwen3.5-35B-A3B (Q4_K_M) ]
        "Will hit 14 tok/s using --n-cpu-moe flag"
```

### 3.3 Module 3 — Shadow Pool (LAN P2P)

Turns standalone instances into a collaborative compute layer for hackathons, labs, or shared workspaces.

- **Zero-config discovery**: mDNS/Bonjour — the same mechanism used by wireless printers or AirDrop. Peers on the same Wi-Fi/LAN automatically announce their hardware profile and locally available models, with no external server.
- **Unified API proxy**: each node exposes a local, OpenAI/Ollama-compatible endpoint (`http://localhost:xxxx/v1/chat/completions`). Any existing tool or script that speaks the OpenAI API format works against ShadowBench unmodified.
- **Intelligent routing**: incoming requests are evaluated against the known pool state; if a peer can run the target model faster (e.g., more free VRAM), the request is transparently forwarded to that peer and results are streamed back token-by-token.
- **Transport resilience**: primary transport is TLS-encrypted WebSockets (`wss://`); if the LAN blocks or throttles TCP (common on shared/university networks), the system falls back to a QUIC/WebRTC datagram channel so a single dropped frame doesn't stall the entire token stream.

## 4. Security & Isolation Model

Because the pool runs on shared/uncontrolled networks (hackathons, university labs, coworking spaces), isolation is a first-class architectural concern:

- **Transport encryption**: ad-hoc, self-signed TLS certificates are generated locally on app startup; all peer traffic is encrypted end-to-end over `wss://`.
- **Request sandboxing**: a provider node only accepts a raw prompt string into its local inference context. Incoming requests have **no** access to the provider's filesystem, environment variables, or other local network interfaces.
- **Trust boundary**: the mDNS broadcast and pairing process forms the trust boundary — a node only shares compute with peers actively discovered on the same local network segment.

## 5. Resiliency Layer

| Failure Mode | Mitigation |
|---|---|
| Peer disconnects mid-inference | Proxy detects the socket drop, hot-swaps to the next best peer in the pool, and resumes streaming without raising an exception to the caller |
| GPU thermal throttling on a provider node | Profiler detects the clock/temperature change, lowers that node's advertised compute weight in the mDNS broadcast, and shifts new requests to cooler peers |
| Requested model available at a different quantization | Predictor recalibrates on the fly and surfaces an explicit throughput trade-off prompt to the client before proceeding |
| Router blocks/drops WebSocket (TCP) traffic | Falls back to an unreliable QUIC/WebRTC datagram channel so single dropped frames don't stall the stream |
| Host needs its own compute mid-session | `llama-swap`-style manager pauses the peer session, frees VRAM for the local task, then restores the peer context afterward |

## 6. Local Data Layer

See [`DATAFLOW.md`](./DATAFLOW.md#4-local-datastore-schema) for the full SQLite schema used to record hardware profiles and predicted-vs-actual benchmark runs, which power the crowd-sourced calibration loop.

## 7. Recommended Stack Rationale

| Layer | Choice | Why |
|---|---|---|
| Desktop shell | Tauri | <20MB binary size; Rust backend with native system access; HTML/TS frontend |
| Core logic sidecar | Python, or Go/Rust | System profiling, matrix stress tests, mDNS discovery |
| Inference backend | Ollama / llama.cpp | Proven local inference performance; ShadowBench builds a routing/optimization layer on top rather than reimplementing an LLM runner |
| P2P networking | FastAPI + WebSockets (Python), or Go/Rust | Local proxy and mDNS discovery service |
