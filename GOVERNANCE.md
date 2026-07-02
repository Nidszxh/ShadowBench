# ShadowBench Governance

ShadowBench is an open-source project. This document describes how decisions are made while the project is
young; it will evolve as the community grows.

## Roles

- **Users** — anyone running ShadowBench. Feedback and hardware benchmark contributions are first-class.
- **Contributors** — anyone whose PR has been merged.
- **Maintainers** — contributors with commit rights, listed in [`.github/CODEOWNERS`](./.github/CODEOWNERS).
  They review PRs, triage issues, and cut releases.

## Decision making

- **Everyday changes** (bug fixes, docs, tests, new hardware rows): lazy consensus — a maintainer approval and
  green CI is enough to merge.
- **Substantial changes** (new modules, IPC/protocol changes, dependency additions, security-relevant code):
  open an issue or discussion first. Requires two maintainer approvals.
- **Protocol / breaking changes**: must document the P2P protocol-version bump and migration path. See the
  versioning policy below.

## Versioning

- Semantic Versioning (`MAJOR.MINOR.PATCH`).
- The **P2P protocol version** is tracked separately (`PROTOCOL_VERSION` in `core/src/shadowbench/__init__.py`).
  Peers on mismatched protocol versions must fail discovery gracefully, never crash.

## Releases

Maintainers cut releases from `main` following [`MILESTONES.md`](./MILESTONES.md). Every release is signed,
changelogged, and published to GitHub Releases (and PyPI for the `core` package).

## Becoming a maintainer

Sustained, high-quality contributions over time. Existing maintainers nominate and confirm by consensus.
