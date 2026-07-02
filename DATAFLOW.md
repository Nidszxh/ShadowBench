# ShadowBench — Dataflow & Math Engine

This document traces exactly how data moves through ShadowBench, and the mathematical models used to predict performance.

## 1. Deep-Dive Math Engine

ShadowBench cannot predict performance with a single formula — Dense and Mixture-of-Experts (MoE) models have fundamentally different memory and compute behavior, so the engine branches per model topology.

### 1.1 Baseline Dense Memory Formula

A quick first-order estimate of a dense model's memory footprint:

```
Memory Required (GB) ≈ (Parameters (Billions) × Quantization Bits / 8) × 1.2
```

The `1.2` multiplier accounts for context-window overhead (KV cache).

### 1.2 KV Cache Overhead Formula (Precise)

Context memory scales with user input length, not just model weights:

```
M_KV = 2 × n_layers × n_heads × d_head × b_precision × c_context
```

| Symbol | Meaning |
|---|---|
| `n_layers` | Number of transformer layers |
| `n_heads` | Number of KV heads (accounts for GQA/MQA) |
| `d_head` | Head dimension |
| `b_precision` | Bytes per token (typically 2 for FP16) |
| `c_context` | Target context length (e.g., 4096, 8192) |

### 1.3 MoE Layer-Splitting Model

For MoE models (e.g., Qwen3.5-35B-A3B), weights split into two categories:

- `Size_NonExpert` — attention, embedding, and normalization layers (always resident)
- `Size_ExpertTotal` — total weight of all routing experts (only a subset active per token)

**Allocation priority:**

1. Force 100% of `Size_NonExpert` into VRAM. If it doesn't fit, predict a severe PCIe-fallback penalty.
2. Compute remaining VRAM (`VRAM_rem`). Allocate a fractional share `K` of active experts into VRAM; the remaining `(1 − K)` experts map to system RAM via an `--n-cpu-moe`-style offload flag.

**Throughput prediction:**

```
T = 1 / (
      (Active_Params_VRAM / GPU_TFLOPS) +
      (Active_Params_RAM / System_Memory_Bandwidth) +
      (Shared_Layers_Overhead / PCIe_Bandwidth)
    )
```

### 1.4 Dense vs. MoE Decision Branch

```
                        [ Model Profile Task ]
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   [ Is it a Dense Model? ]                 [ Is it an MoE Model? ]
              │                                        │
   VRAM strict limit applies.              VRAM must hold the base
   Spillover into system RAM               architecture + a partial
   predicts a severe t/s drop              subset of experts. Predict
   from PCIe bottlenecks.                  viable speed via expert
                                            offload routing.
```

### 1.5 Config Coach Output Example

Given an 8GB VRAM / 32GB RAM machine targeting a 35B MoE model:

> "Set `--n-cpu-moe 128` and enable `--no-mmap` to lock weights in system RAM for a targeted 14 tokens/sec."

### 1.6 Prefill vs. Decode Batch Tuning

LLM inference has two distinct compute phases that respond differently to partial MoE offloading:

- **Prefill** (prompt processing): compute-heavy, benefits from high memory bandwidth.
- **Decode** (token generation): bound by memory access speed.

When offloading experts to system RAM, an undersized micro-batch causes the GPU to stall waiting for the CPU to stream expert weights over PCIe. The predictor compensates by dynamically tuning:

- **`-ub` / `--ubatch`** — scaled up (e.g., 512 → 2048/4096) on slow-PCIe/high-RAM systems to keep GPU kernels saturated.
- **`--parallel`** — forced to `1` on low-VRAM machines (default multi-slot concurrency silently multiplies KV cache allocation and can starve weight memory), unless the machine is explicitly acting as a pool provider.

## 2. End-to-End Dataflow: P2P Inference Request

Sequence when a local script's request gets delegated to a faster peer:

```
[ Your Code Script ]   [ ShadowBench Proxy ]   [ mDNS Registry ]   [ Friend's Laptop ]
        │                       │                      │                   │
        │  1. POST Prompt       │                      │                   │
        ├──────────────────────>│                      │                   │
        │                       │ 2. Check Available Pools                 │
        │                       ├─────────────────────>│                   │
        │                       │<──────────────────────┤                   │
        │                       │   Returns: Friend_Node (RTX 4060)         │
        │                       │                                          │
        │                       │ 3. Forward Encrypted Data over WebSocket │
        │                       ├─────────────────────────────────────────>│
        │                       │                                          │ [ Runs Model ]
        │                       │                                          │ [ Generates Token ]
        │                       │ 4. Stream Raw Tokens (Chunked JSON)      │
        │                       │<─────────────────────────────────────────┤
        │  5. Stream Output     │                                          │
        │<──────────────────────┤                                          │
```

**Step-by-step:**

1. **Request** — a local script sends a standard OpenAI-format request to `http://localhost:8080/v1/chat/completions`:
   ```json
   {
     "model": "Qwen3.5-35B-A3B",
     "messages": [{"role": "user", "content": "Write a fast matrix sort in C++"}]
   }
   ```
2. **Routing decision** — the proxy checks its mDNS-populated router table and compares estimated local vs. peer completion time; if a peer is significantly faster, it reroutes.
3. **Transport** — the prompt is wrapped in an encrypted TLS WebSocket frame and sent to the peer's local IP (e.g., `192.168.1.45:9000`).
4. **Execution & return** — the peer's background service runs the prompt through its local `llama.cpp`/Ollama instance and streams tokens back over the open socket as they're generated.

## 3. Dataflow: Requirement Discovery → One-Click Download

```
[ User UI ]        [ Tauri Backend (Rust) ]   [ Hugging Face API ]   [ Local Disk / Engine ]
    │                        │                        │                       │
    │ 1. Select task +       │                        │                       │
    │    speed/accuracy      │                        │                       │
    │    profile             │                        │                       │
    ├───────────────────────>│                        │                       │
    │                        │ 2. Read hardware +      │                       │
    │                        │    run prediction math  │                       │
    │                        │ 3. Query top match      │                       │
    │                        ├───────────────────────>│                       │
    │                        │<────────────────────────┤                       │
    │                        │   Returns: GGUF URL(s)  │                       │
    │ 4. Display match       │                                                │
    │<───────────────────────┤                                                │
    │ 5. Click "One-Click    │                                                │
    │    Setup"              │                                                │
    ├────────────────────────────────────────────────────────────────────────>│
    │                        │                                                │ [ Downloads file ]
    │                        │                                                │ [ Spawns llama.cpp/Ollama ]
    │                        │                                                │ [ Injects optimal flags ]
```

### 3.1 Selection Logic (Conceptual)

```python
# Conceptual backend logic for model selection
def recommend_best_model(user_intent, user_profile, hardware):
    # Step 1: Filter base architecture families by task
    if user_intent == "coding":
        candidates = ["Qwen3.5-35B-A3B", "DeepSeek-R1-Distill-14B", "Llama-3-8B"]
    else:
        candidates = ["Mistral-7B", "Phi-3-Medium"]

    for model in candidates:
        # Step 2: Compute memory constraints per quantization type
        if is_moe(model):
            base_layer_size = calculate_base_layers(model, quant="Q4_K_M")   # ~7GB
            expert_slice_size = calculate_active_experts(model)              # ~3.5GB/step

            if base_layer_size < hardware.vram_available:
                predicted_speed = simulate_moe_throughput(hardware, base_layer_size)

                if user_profile == "intelligence-first" and predicted_speed > 10:
                    return {
                        "model": model,
                        "quant": "Q4_K_M",
                        "optimal_flags": "--ngl 99 --n-cpu-moe 128 -ub 2048",
                        "estimated_tps": predicted_speed,
                    }
```

## 4. Peer-to-Peer Protocol Lifecycle

```
[ Node A (Client) ]            [ Local Network (mDNS) ]           [ Node B (Provider) ]
        │                                 │                                 │
        │ 1. Broadcast Discovery Query    │                                 │
        ├────────────────────────────────>│                                 │
        │                                 │ 2. Forward Query                │
        │                                 ├────────────────────────────────>│
        │                                 │ 3. Unicast TLS Identity + HW    │
        │                                 │<────────────────────────────────┤
        │ 4. Establish Secure WebSocket (WSS)                               │
        ├───────────────────────────────────────────────────────────────────>│
        │ 5. POST /v1/chat/completions (proxy)                              │
        ├───────────────────────────────────────────────────────────────────>│
        │ 6. Stream Tokens (Chunked JSON Frames)                            │
        │<───────────────────────────────────────────────────────────────────┤
```

**Security controls applied at each hop:**

- All node-to-node traffic uses ad-hoc, self-signed TLS certificates generated at startup (`wss://` only).
- Providing nodes sandbox incoming requests to prompt-string-only access — no filesystem, environment, or loopback network access.

## 5. Local Datastore Schema

Stored in an embedded SQLite database within the Tauri binary; used to compare predicted vs. actual performance and recalibrate the predictor over time.

```
+---------------------------------------------------------------------------------+
|                                  SQLite SCHEMA                                  |
+---------------------------------------------------------------------------------+
|  [hardware_profiles]                                                            |
|  - id (UUID, PK)                                                                |
|  - gpu_model (TEXT)            e.g., "NVIDIA GeForce RTX 4060 Laptop GPU"       |
|  - vram_total_mb (INTEGER)     e.g., 8192                                       |
|  - system_ram_gb (INTEGER)     e.g., 32                                         |
|  - pcie_bandwidth_gbps (REAL)  e.g., 12.4 (measured via live stress test)       |
+---------------------------------------------------------------------------------+
                                      │
                                      ▼ (1 to Many)
+---------------------------------------------------------------------------------+
|  [benchmark_runs]                                                               |
|  - id (UUID, PK)                                                                |
|  - hardware_profile_id (FK)                                                     |
|  - model_name (TEXT)           e.g., "Qwen/Qwen3.5-35B-A3B-GGUF"                |
|  - quantization (TEXT)         e.g., "Q4_K_M"                                   |
|  - context_length (INTEGER)    e.g., 4096                                       |
|  - predicted_tps (REAL)        e.g., 14.2                                       |
|  - actual_tps (REAL)           e.g., 13.8                                       |
+---------------------------------------------------------------------------------+
```

On every successful local run, ShadowBench logs predicted vs. actual t/s. When connectivity is available, anonymized records sync to a central open-source dataset to continually improve prediction accuracy across the developer community.

## 6. Edge Case → Data Response Matrix

| Scenario | System Response |
|---|---|
| Node drops mid-inference | Proxy detects the socket drop, hot-swaps to next best peer, mirrors conversation history state, resumes streaming |
| Host GPU thermal throttling | Profiler detects temperature/clock change, lowers advertised compute weight in mDNS broadcast, shifts new requests to cooler peers |
| Requested model available at different quantization on host | Predictor recalibrates on the fly and returns a fallback alert (e.g., "peer only has Q8_K instead of Q4_K_M — throughput drops from 18 to 6 t/s. Proceed?") |
