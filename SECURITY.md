# Security Policy

ShadowBench runs a peer-to-peer inference pool on shared, often-untrusted networks (hackathons, labs,
coworking Wi-Fi). Security is a first-class concern, not an afterthought. This document covers how to report
issues and the threat model we hold ourselves to.

## Reporting a vulnerability

**Do not open a public issue for security problems.**

- Use GitHub's **private vulnerability reporting** (Security → Report a vulnerability) on this repository, or
- Email the maintainers at `security@shadowbench.dev` (placeholder — update before public launch).

We aim to acknowledge within **72 hours** and provide a remediation timeline within **7 days**. Please include
reproduction steps, affected version/commit, and impact. We support coordinated disclosure and will credit
reporters unless you prefer to remain anonymous.

## Supported versions

Until `v1.0.0`, only the latest `main` and the most recent tagged release receive security fixes.

## Threat model (Shadow Pool)

The pool's trust boundary is the local network segment discovered via mDNS. The controls below are
**requirements**, verified by automated tests before the P2P feature (Phase 4) ships.

| Threat | Control |
|---|---|
| Eavesdropping on peer traffic | All node-to-node traffic over TLS (`wss://`) using certs generated locally at startup. |
| Malicious peer joining the pool | Explicit **pairing confirmation** (PIN/QR) before a peer can send work — not silent trust-on-first-use. |
| Prompt payload escaping into the host | Provider nodes accept **prompt strings only** — no filesystem, environment, or loopback network access from a request. Enforced and fuzz-tested. |
| Resource exhaustion / DoS from a peer | Per-peer rate limits and request-size caps on the proxy. |
| Telemetry leaking PII | Opt-in only, default-off. An automated test asserts payloads contain **no** hostname, IP, or user identifier before any upload. |

## Out of scope (pre-1.0)

- Attacks originating from the public internet — the pool is LAN-scoped by design.
- Compromise of the host OS or the underlying inference engine (`llama.cpp` / Ollama).

## Dependencies

We run Dependabot and secret scanning, and publish an SBOM with each release. Report suspected supply-chain
issues through the same private channel above.
