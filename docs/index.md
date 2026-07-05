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
| Understand the system | [Architecture](plan/ARCHITECTURE.md) |
| See the prediction math | [Dataflow & Math](plan/DATAFLOW.md) |
| Navigate the code | [Project Structure](../PROJECT_STRUCTURE.md) |
| Follow the delivery plan | [Roadmap](plan/ROADMAP.md) |
| Contribute | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
