# Contributing to ShadowBench

Thanks for helping build ShadowBench! This guide gets you from clone to green PR.

## Ground rules

- Be kind — see [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).
- Security issues go through [`SECURITY.md`](./SECURITY.md), **not** public issues.
- One logical change per PR. Keep the diff reviewable.
- Every code change ships with tests and updated docs.

## Repo map

The codebase is a component-wise monorepo — read [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md) first.
The three core modules (`profiler/`, `predictor/`, `pool/`) are independent packages; pick the one your
change touches and stay within its boundary.

## Development setup (`core/`)

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python envs.

```bash
# from repo root
cd core
uv sync --all-extras          # create venv + install core, gpu, pool, and dev deps
uv run shadowbench --help     # sanity check the CLI
```

Then install the git hooks (run once, from the repo root):

```bash
uv run --project core pre-commit install
```

## The dev loop

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy
make test        # pytest with coverage
make format      # auto-fix with ruff --fix + ruff format
make check       # everything CI runs, locally
```

All four gates (`lint`, `typecheck`, `test`, coverage floor) must pass before a PR merges — CI enforces the
same commands on the Windows/macOS/Linux matrix.

## Coding standards

- **Python 3.11+**, fully type-annotated. `mypy --strict`-clean in `core/`.
- **Pure math stays pure.** Nothing in `predictor/` may do hardware or network I/O — that keeps it testable
  on any machine and against the golden dataset.
- **Reference the spec.** When you implement a formula, cite the `DATAFLOW.md` section in the docstring.
- **Errors are typed.** Raise a subclass of `ShadowBenchError` (see `common/errors.py`), never a bare
  `Exception`, for anything a user might see.
- Tests mirror the source tree: `predictor/moe.py` ↔ `tests/unit/test_moe.py`.

## Contributing hardware data

The accuracy of the Predictor depends on real-world measurements. The single most valuable contribution is a
benchmark row:

```bash
uv run shadowbench bench --contribute   # runs llama-bench, appends to datasets/golden.jsonl
```

Open a PR with the new `datasets/golden.jsonl` rows — see [`datasets/README.md`](./datasets/README.md) for the
schema. New hardware also earns a row in the support matrix.

## Commit & PR conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Fill in the PR template; link the issue it closes.
- Keep the changelog updated via the changeset flow noted in the PR template.

## Good first issues

Look for the `good first issue` and `help wanted` labels. The `predictor/` unit tests and `datasets/` rows are
the friendliest entry points — pure functions, no hardware required.
