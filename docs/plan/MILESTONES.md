# ShadowBench — Production Milestones

> **Goal:** ship ShadowBench as a **production-quality, open-source tool** — signed installers,
> a public accuracy dataset, a documented threat model, and a contributor-friendly codebase.
>
> **Stack:** Python sidecar (Profiler / Predictor / Pool math) + Tauri desktop shell (Rust core, React/TS UI).
> **Realistic timeline:** ~20–24 weeks to `v1.0.0`. Each phase exits with a **signed, versioned, public release**.

For the system design see [`ARCHITECTURE.md`](./ARCHITECTURE.md), the math in [`DATAFLOW.md`](./DATAFLOW.md),
and the concrete repo layout in [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md).

---

## Why this differs from the 5-week roadmap

The original [`ROADMAP.md`](./ROADMAP.md) is a good *feature* order for a demo. This document rebuilds it for
**production + open-source**, which adds concerns the demo plan skips:

| Dimension | Demo roadmap | **This plan (production OSS)** |
|---|---|---|
| Timeline | ~5 weeks | ~20–24 weeks to `v1.0.0` |
| Accuracy loop | Phase 5 afterthought | Core feature **and** a public open dataset (from Phase 2) |
| Every phase exits with | A working increment | A **signed, installable, documented public release** |
| OSS infra | License only | Governance, docs site, release cadence, security policy, contributor UX |
| Hardware support | Whatever you own | Documented, tested support matrix (NVIDIA / Apple / AMD / CPU) |
| Security | "sandbox the prompt" | Public threat model, pairing confirmation, external review before P2P ships |

---

## Phase 0 — Project Foundation & Governance (Weeks 1–2)

The OSS bedrock. Skipping this is what makes open-source projects stall after `v0.1`.

- **P0.1 Repo & tooling** — monorepo (`core/`, `frontend/`, `schemas/`, `datasets/`, `docs/`), `uv` + `pyproject.toml`, `ruff` + `black` + `mypy`, pre-commit.
- **P0.2 Governance & community files** — `LICENSE` (Apache-2.0, for the patent grant on a networking/security tool), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `GOVERNANCE.md`, issue/PR templates, `CODEOWNERS`.
- **P0.3 CI/CD from commit #1** — GitHub Actions matrix (Win/macOS/Linux): lint, typecheck, test, coverage gate. Branch protection, Dependabot, secret scanning.
- **P0.4 Docs site** — MkDocs Material auto-deployed to GitHub Pages; existing markdown becomes the seed.
- **P0.5 Release skeleton** — Changesets/semantic-release, `CHANGELOG.md`, and a **P2P protocol version** constant kept separate from the app version.

> **Public deliverable — `v0.0.1`:** a repo anyone can clone, lint, and test green. Nothing runs yet, but the scaffolding is production-grade.

## Phase 1 — Hardware Profiler (Weeks 3–5)

Maps to roadmap Phase 1, hardened for real hardware diversity.

- **P1.1 Detection layer** — `pynvml` (NVIDIA), `system_profiler`/Metal (Apple unified memory), ROCm SMI (AMD), `psutil` (RAM/CPU). Every path has a graceful fallback; **no crash on exotic hardware**.
- **P1.2 Bandwidth stress kernel** — bounded ~3s matrix-mult → real GB/s + TFLOPS, hard-capped runtime, regression-tested against its time budget.
- **P1.3 GGUF metadata reader** — parse headers for topology/quant; never hardcode.
- **P1.4 Public support matrix** — a documented, tested "known-good hardware" table; community adds rows via PR.
- **P1.5 Anonymized profile schema** — designed from the start to strip PII (this schema is what later syncs publicly).

> **Public deliverable — `v0.1.0`:** `shadowbench profile` CLI on PyPI. First installable artifact.

## Phase 2 — Predictor & the Open Accuracy Dataset (Weeks 6–9) ⭐

Maps to roadmap Phase 2. Accuracy is central — and a **community asset**, not a private gate.

- **P2.1 Dense + MoE throughput models** + the KV-cache formula ([`DATAFLOW.md §1.2–1.4`](./DATAFLOW.md)).
- **P2.2 Config Coach** — exact flags (`--n-cpu-moe`, `-ub`, `--ngl`, `--parallel`) + prefill/decode batch tuning (§1.6).
- **P2.3 Requirement Discovery** — Task × Hardware × Profile → ranked candidates.
- **P2.4 Ground-truth harness** — wraps `llama-bench` to record real t/s; every run can contribute.
- **P2.5 🌟 Public accuracy dataset** — versioned `datasets/golden.jsonl` + a CI-published accuracy report (predicted-vs-actual, median error). **CI fails on accuracy regression.**

> **Public deliverable — `v0.2.0`:** CLI recommends model + flags + predicted t/s, with a published accuracy report proving it's honest.

## Phase 3 — Desktop Shell & One-Click Orchestrator (Weeks 10–13)

Maps to roadmap Phase 3. Packaging + code-signing become first-class.

- **P3.1 Tauri shell** — scannable dashboard (React/TS), accessible (keyboard nav, contrast).
- **P3.2 Versioned IPC contract** — Python sidecar bundled via PyInstaller as a Tauri sidecar. **Spike this in Phase 1** so it isn't a Week-10 surprise — it's the main tax of the Python choice.
- **P3.3 Resumable chunked downloader** — checksum-verified, interrupt/resume.
- **P3.4 Process orchestration** — spawn/manage `llama.cpp`/Ollama with Config Coach flags; surface crashes.
- **P3.5 SQLite + auto-logging** — every real run feeds the accuracy loop from here on, not Phase 5.
- **P3.6 Code signing pipeline** — macOS notarization + Windows Authenticode, wired into CI now.

> **Public deliverable — `v0.3.0`:** signed installers (Win/macOS/Linux) on GitHub Releases + auto-updater.

## Phase 4 — Shadow Pool P2P (Weeks 14–18)

Maps to roadmap Phase 4. The security surface goes public here.

- **P4.1 mDNS discovery** — peer table with TTL expiry.
- **P4.2 TLS + pairing** — self-signed certs + **PIN/QR pairing confirmation** (raw trust-on-first-use on open Wi-Fi is a real attack vector).
- **P4.3 Load-balancing proxy** — OpenAI-compatible `/v1/chat/completions`; routes local-vs-remote.
- **P4.4 WebSocket streaming bridge** — consistent chunked-JSON framing regardless of source.
- **P4.5 Public security review** — documented threat model in `SECURITY.md` + external review/bug bounty before shipping.
- ⏸️ **QUIC/WebRTC fallback → deferred** to a later minor release unless real LAN testing proves TCP gets blocked.

> **Public deliverable — `v0.4.0` (beta):** two machines pool inference over LAN, with a published threat model.

## Phase 5 — Resiliency, Self-Correction & v1.0 (Weeks 19–24)

Maps to roadmap Phase 5 + everything "1.0-worthy."

- **P5.1 Failover engine** — peer drop → hot-swap + state mirroring, chaos-tested.
- **P5.2 `llama-swap`-style suspend/resume** when the host reclaims compute.
- **P5.3 Opt-in telemetry sync (privacy-verified)** — anonymized public-dataset upload; automated test asserting **zero** hostname/IP/user IDs leave the machine. Default-off, clearly disclosed.
- **P5.4 v1.0 hardening** — perf budgets, error taxonomy, full docs, protocol-version negotiation, reproducible builds, SBOM.

> **Public deliverable — `v1.0.0`:** signed, documented, self-improving, community-ready.

---

## Continuous tracks (every phase)

| Track | What "production open-source" demands |
|---|---|
| **Public accuracy dashboard** | Predicted-vs-actual visible on the docs site — the credibility metric. |
| **Release discipline** | Every phase ends in a signed, versioned, changelogged public release. |
| **Security posture** | `SECURITY.md`, disclosure process, dependency scanning, threat model reviewed at P4 *and* P5. |
| **Contributor UX** | Good-first-issues, "how the math works" docs, CI a stranger can pass on their first PR. |
| **Reproducibility** | Pinned deps, lockfiles, SBOM, reproducible cross-platform builds. |

## Top risks

1. **Python-sidecar packaging + code signing** — the biggest tax of the stack choice. De-risk with a throwaway *signed* hello-world sidecar build by Phase 1, or it bites in Phase 3.
2. **P2P security is public-facing.** An OSS tool inviting strangers to share GPU compute over LAN is a target. `SECURITY.md` + threat model + pairing confirmation are prerequisites for trusting `v1.0`, not polish.

## Review-driven additions

An external architecture review ([`../../ARCHITECTURE_REVIEW.md`](../../ARCHITECTURE_REVIEW.md)) surfaced gaps now folded into the phases above:

| Item | Phase |
|---|---|
| GGUF malformed-metadata fallback | P1.3 |
| Cold-start accuracy expectation (pre-calibration ±band) | P2.5 |
| Pre-flight disk-space check + resumable partials | P3.3 |
| Signed/checksummed model downloads before spawn | P3.3 |
| mDNS discovery fallback (manual peer-add for AP client-isolation) | P4.1 |
| TLS fingerprint pinning on pairing (anti-spoofing) | P4.2 |
| Concurrent-routing arbitration (queue-depth advertisement) | P4.3 |
| Requester-side proxy supervision / auto-restart | P4.3 / P5.1 |
| Circuit breaker for flaky peers | P5.1 |
| Telemetry outlier filtering (anti-poisoning) | P5.3 |

Two suggestions need a decision (see the review): **`sysinfo`/Rust vs. the Python sidecar** (revisit at P3.2) and **adopting `libp2p`/`iroh` before hand-rolling P2P crypto** (spike at P4.2).

## Version → milestone map

| Version | Phase | Headline |
|---|---|---|
| `v0.0.1` | 0 | Repo scaffold, CI green |
| `v0.1.0` | 1 | Hardware profiler CLI (PyPI) |
| `v0.2.0` | 2 | Model recommender + public accuracy report |
| `v0.3.0` | 3 | Signed desktop app, single-machine |
| `v0.4.0` | 4 | LAN inference pooling (beta) |
| `v1.0.0` | 5 | Hardened, self-improving, community-ready |
