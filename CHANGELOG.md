# Changelog

All notable changes to ShadowBench are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 0 foundation.** Component-wise monorepo scaffold: `core/` (Python sidecar) with `profiler/`,
  `predictor/`, `pool/`, `orchestrator/`, `storage/`, `calibration/`, and `ipc/` packages.
- Governance & community files: `LICENSE` (Apache-2.0), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `GOVERNANCE.md`.
- CI matrix (lint / typecheck / test) across Windows, macOS, and Linux.
- `MILESTONES.md` and `PROJECT_STRUCTURE.md`.
- Seed `datasets/models_catalog.json` and `datasets/golden.jsonl` schema.

[Unreleased]: https://github.com/OWNER/ShadowBench/commits/main
