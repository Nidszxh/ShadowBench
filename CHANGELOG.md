# Changelog

All notable changes to ShadowBench are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-07-05

### Added

- **Catalog validation CI.** New `catalog validate` and `catalog add <hf-id>` CLI commands for
  managing `datasets/models_catalog.json`. The `validate` command checks every entry against the
  `ModelSpec` schema (topology consistency, positive params, valid quants/tasks). The `add` command
  auto-fetches model metadata from the Hugging Face API and prints a catalog entry template.
  A dedicated `.github/workflows/validate-catalog.yml` workflow validates the catalog on every PR
  that touches it.
- **Intel GPU backend.** New `profiler/gpu/intel.py` backend detects Intel integrated and discrete
  GPUs via Linux sysfs (vendor 0x8086) — no extra packages required.
- **Documentation restructure.** MkDocs `docs_dir` moved from `docs/docs/` to `docs/` (removed
  nested `docs/` directory). Nav updated, broken links fixed across `README.md`, `PROJECT_STRUCTURE.md`,
  `MILESTONES.md`, and the MkDocs site. GitHub-flavoured Markdown in `CONTRIBUTING.md` — all `black`
  references replaced with `ruff`.

### Fixed

- **GPU detection across all vendors.** NVIDIA backend now falls back to `nvidia-smi` when `pynvml`
  is not installed. AMD backend implemented via `rocm-smi --json` with sysfs fallback
  (vendor 0x1002). All three GPU backends work with zero pip extras — users only need the optional
  `nvidia` extra for NVML-specific telemetry.
- **`--no-stress` predictions match full stress test.** System RAM bandwidth is now always measured
  (~0.1 s, standalone `measure_ram_bandwidth()`) regardless of the `--no-stress` flag, so the
  predictor always has real bandwidth data for CPU-offload estimation.
- **Conservative fallback defaults.** `_FALLBACK_RAM_GBPS` lowered from 30.0 to 15.0 GB/s
  (DDR4 single-channel floor) for the rare case no measurement is available.

### Changed

- `profile_hardware()` always returns a populated `BandwidthResult` (only GEMM fields are zeroed
  when `--no-stress` is set).
- `--no-stress` help text now explains it only skips the 3-second GEMM burner.

## [0.2.0] - 2026-07-05

### Added

- **Phase 0 foundation.** Component-wise monorepo scaffold: `core/` (Python sidecar) with `profiler/`,
  `predictor/`, `pool/`, `orchestrator/`, `storage/`, `calibration/`, and `ipc/` packages.
- Governance & community files: `LICENSE` (Apache-2.0), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `GOVERNANCE.md`.
- CI matrix (lint / typecheck / test) across Windows, macOS, and Linux.
- `MILESTONES.md`, `ROADMAP.md`, and `PROJECT_STRUCTURE.md`.
- Seed `datasets/models_catalog.json` and `datasets/golden.jsonl` schema.
- **Phase 1: Hardware Profiler.** GPU detection (NVIDIA, Apple Silicon, AMD stub, CPU fallback),
  system RAM/CPU sensing, PCIe bandwidth stress kernel, GGUF metadata reader, anonymized profile schema.
- **Phase 2: Predictor & Accuracy Dataset.** Dense/MoE throughput models, Config Coach runtime flags,
  Requirement Discovery engine, ground-truth harness (`llama-bench`), accuracy report evaluator.
- **CLI commands:** `shadowbench profile`, `shadowbench recommend`, `shadowbench bench` with `--contribute`.
- **First PyPI release.** `pip install shadowbench` is now the primary distribution channel.

[0.2.1]: https://github.com/Nidszxh/ShadowBench/releases/tag/v0.2.1
[0.2.0]: https://github.com/Nidszxh/ShadowBench/releases/tag/v0.2.0
