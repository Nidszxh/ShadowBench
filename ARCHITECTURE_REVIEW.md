# ShadowBench — Architect's Review & Response

A structured triage of an external architecture review. Each finding is tagged:

- ✅ **Addressed** — already handled in the current scaffold (with a pointer).
- ⚠️ **Partial** — partially handled; hardening tracked in a milestone.
- 🆕 **New** — a real gap this review surfaced; now folded into the roadmap.
- 💡 **Optimization** — a suggestion evaluated with a recommendation (may defer to a prior decision).

---

## Security

### ⚠️ Peer authentication vs. encryption only — mDNS spoofing
**Finding:** TLS is encryption, not identity. mDNS is trivially spoofable on a shared subnet; a malicious "fast peer" could harvest prompts.
**Status:** The pairing step (PIN/QR) in [`SECURITY.md`](./SECURITY.md) already blocks silent trust-on-first-use. **Hardened here:** pairing now **pins the peer's TLS fingerprint** on first accept (TOFU); a later fingerprint change for a known `node_id` is rejected as impersonation. Tracked in **P4.2**.

### ✅ Rate limiting / resource quota (DoS)
Already in the `SECURITY.md` threat model: per-peer rate limits + request-size caps on the proxy. Enforced in `pool/proxy.py` (P4.3). Extended below with a **circuit breaker** (see Optimizations).

### ⚠️ → 🆕 Signed model downloads
**Finding:** `.gguf` files are auto-fed to a spawned process without integrity checks.
**Status:** `orchestrator/downloader.py` already takes an `expected_sha256`. **Upgraded to a requirement:** verify against a known-good SHA-256 (and publisher signature where available) **before** the engine process is spawned. No hash → explicit user confirmation. Tracked in **P3.3**.

## Data validation / correctness

### 🆕 GGUF metadata trust
**Finding:** malformed/missing/spoofed GGUF metadata isn't handled.
**Response:** `profiler/gguf.py` will validate required keys and fall back to the `models_catalog.json` entry (or an explicit "unknown topology → conservative dense estimate") rather than crashing. Never trust header-declared param counts that contradict file size by >X%. Tracked in **P1.3**.

### 🆕 Predictor calibration drift / bad-faith submissions
**Finding:** the crowd-sourced sync loop has no outlier filtering — a broken GPU reporting nonsense TFLOPS pollutes the shared dataset.
**Response:** `calibration/sync.py` gains a validation gate: physically-plausible bounds (TFLOPS/bandwidth/tps ranges), per-`(gpu,model,quant)` median + MAD outlier rejection, and a submission cap per anonymized profile. Server-side re-validation before a row enters the public set. Tracked in **P5.3**.

### 🆕 Cold-start problem
**Finding:** before any crowd-sourced data exists, predictions are pure theory — early-accuracy expectations must be set accordingly.
**Response:** documented explicitly in `MILESTONES.md` P2 — the ±band starts wide (±25%) precisely because Phase 2 ships pre-calibration; the golden dataset tightens it over time. This is a feature of the ratcheting accuracy gate, not a defect.

## Scaling bottlenecks

### 🆕 mDNS doesn't survive AP client isolation
**Finding:** many enterprise/university APs block device-to-device multicast, silently breaking discovery with no fallback.
**Response:** add a **manual peer-add fallback** (paste `host:port` or scan a QR that encodes it) when mDNS yields nothing, plus a clear "multicast blocked on this network" diagnostic. Tracked in **P4.1**. (Distinct from the *transport* TCP→QUIC fallback already noted for P4.4.)

### 🆕 Single-proxy bottleneck / requester-side crash
**Finding:** only provider-side failover is covered; a crash of the requester's own proxy mid-session isn't.
**Response:** the local proxy gets a supervisor (auto-restart) and the Tauri host surfaces a clean "local proxy restarting" state; in-flight requests return a retriable error rather than hanging. Tracked in **P4.3 / P5.1**.

## Edge cases

### 🆕 Concurrent recommendation conflicts
**Finding:** two clients routed to the same "best" peer — no arbitration described.
**Response:** the router treats a peer's advertised free-VRAM/queue-depth as a claimed resource; on contention it queues or falls to the next-best peer. Provider advertises current queue depth in its mDNS/heartbeat payload. Tracked in **P4.3**.

### 🆕 Partial downloads / disk-space check
**Finding:** no pre-flight disk check before a 15–20 GB transfer.
**Response:** `orchestrator/downloader.py` checks free space (size × safety margin) before starting and resumes partials via HTTP range. Tracked in **P3.3**.

### ✅ Version skew between peers
Already handled: `PROTOCOL_VERSION` is tracked independently of app version; mismatched peers fail discovery gracefully (see `GOVERNANCE.md`, `MILESTONES.md` P4).

## Optimization suggestions

### 💡 libp2p noise vs. hand-rolled TLS — **Adopt (evaluate in P4.2)**
Bundling peer identity + encryption + NAT patterns beats hand-rolling cert generation. Evaluate `libp2p` (or `iroh`, below) at P4.2 before writing bespoke crypto. **Recommendation: adopt, pending a spike.**

### 💡 `iroh` / `quinn` for QUIC vs. bespoke WebSocket+QUIC — **Adopt for the fallback (P4.4)**
Using a mature Rust QUIC stack instead of a hand-built dual-stack removes custom transport code that's expensive to security-audit. Note: this pushes transport into the Rust/Tauri layer rather than the Python sidecar. **Recommendation: adopt for the QUIC fallback; keep primary WebSocket simple.**

### ⚖️ `sysinfo` (Rust) instead of the Python `psutil`/`pynvml` sidecar — **Decision needed**
This **conflicts with the Phase-0 decision** to use a Python sidecar for fastest iteration on the prediction math. Trade-off: Rust `sysinfo` avoids IPC + Python-bundling overhead and helps the <20 MB binary goal, but loses Python's ML/hardware-library velocity. **Recommendation:** keep the Python sidecar through Phase 2 (where math iteration dominates); **re-evaluate at P3.2** if packaging/binary-size becomes the dominant pain. Flagged for your call.

### ✅ SQLite WAL mode
Already enabled — `storage/schema.sql` sets `PRAGMA journal_mode = WAL;` for concurrent benchmark-logging writes alongside UI reads.

### 🆕 Circuit breaker for peer routing
**Finding:** react to flaky peers *before* routing, not only after a disconnect.
**Response:** the router tracks rolling latency/failure rates per peer and temporarily deprioritizes nodes that trip a threshold, with exponential-backoff re-probing. Tracked in **P5.1** alongside the failover engine.

---

## Summary

| Category | ✅ Addressed | ⚠️ Partial→hardened | 🆕 New (now tracked) | 💡 Optimization |
|---|---|---|---|---|
| Security | 1 | 2 | — | — |
| Correctness | — | — | 3 | — |
| Scaling | 1 | — | 2 | — |
| Edge cases | 1 | — | 2 | — |
| Optimizations | 1 | — | 1 | 3 (2 adopt, 1 decision) |

**Two items need your decision:** (1) `sysinfo`/Rust vs. Python sidecar, and (2) adopting `libp2p`/`iroh` before hand-rolling P2P crypto. Both are deferred to their implementation phase and flagged here so they aren't forgotten.
