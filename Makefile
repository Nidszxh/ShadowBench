# ShadowBench developer convenience targets.
# All Python work happens inside core/ via uv.

CORE := core

.PHONY: help setup lint format typecheck test check clean docs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the dev environment and install hooks
	cd $(CORE) && uv sync --all-extras
	uv run --project $(CORE) pre-commit install

lint: ## Lint (ruff) + format check (ruff-format)
	cd $(CORE) && uv run ruff check .
	cd $(CORE) && uv run ruff format --check .

format: ## Auto-fix lint + formatting
	cd $(CORE) && uv run ruff check --fix .
	cd $(CORE) && uv run ruff format .

typecheck: ## Static type checking (mypy)
	cd $(CORE) && uv run mypy src

test: ## Run the test suite with coverage
	cd $(CORE) && uv run pytest

check: lint typecheck test ## Everything CI runs

docs: ## Serve the docs site locally
	uv run --project $(CORE) mkdocs serve -f docs/mkdocs.yml

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(CORE)/.pytest_cache $(CORE)/.mypy_cache $(CORE)/.ruff_cache $(CORE)/htmlcov
