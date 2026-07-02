# ShadowBench

**Predict whether an open-source LLM will run well on your machine — before you download it — and pool
inference compute with nearby machines on the same network.**

- **Benchmarker** — profiles your CPU/GPU/RAM/VRAM and predicts tokens/sec for a target model + quantization,
  without downloading the model first.
- **Shadow Pool** — zero-config LAN peer discovery so teammates can share GPU compute through one
  OpenAI-compatible endpoint.

## Where to start

| I want to… | Read |
|---|---|
| Understand the system | [Architecture](architecture.md) |
| See the prediction math | [Dataflow & Math](dataflow.md) |
| Navigate the code | [Project Structure](structure.md) |
| Follow the delivery plan | [Milestones](milestones.md) |
| Contribute | [`CONTRIBUTING.md`](https://github.com/OWNER/ShadowBench/blob/main/CONTRIBUTING.md) |

!!! note
    The top-level `*.md` files in the repo are the source of truth. This site mirrors them for browsing; the
    nav pages are thin symlink-style includes maintained during Phase 0 (P0.4).
