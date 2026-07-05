"""Catalog validation — run in CI on every PR that touches ``datasets/models_catalog.json``.

Usage:

    uv run python -m shadowbench.predictor.validate_catalog
    uv run python -m shadowbench.predictor.validate_catalog --catalog path/to/models_catalog.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from shadowbench.common.types import Quantization, Task
from shadowbench.predictor.models import ModelSpec

_EXPECTED_TASKS = {t.value for t in Task}
_VALID_QUANTS = {q.value for q in Quantization}


def find_catalog_path(override: str | None = None) -> Path:
    """Locate ``models_catalog.json``: ``override`` arg → env var → parent-dir walk. Exits 1 if not found."""
    if override:
        p = Path(override)
        if p.exists():
            return p
        print(f"ERROR: catalog not found at --catalog {override}", file=sys.stderr)
        sys.exit(1)
    env = os.environ.get("SHADOWBENCH_CATALOG_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "datasets" / "models_catalog.json"
        if candidate.exists():
            return candidate
    print(
        "ERROR: cannot locate models_catalog.json — use --catalog or SHADOWBENCH_CATALOG_PATH",
        file=sys.stderr,
    )
    sys.exit(1)


def validate_entries(entries: list[dict[str, object]]) -> int:
    """Validate catalog entries against ``ModelSpec``. Returns violation count (0 = valid); prints ``FAIL`` lines.

    Checks: Pydantic schema, topology consistency, positive fields, valid quants/tasks.
    """
    errors: list[str] = []
    for i, entry in enumerate(entries):
        idx = f"entry[{i}] ({entry.get('id', '<no-id>')})"

        # Pydantic schema validation
        try:
            spec = ModelSpec.model_validate(entry)
        except ValidationError as exc:
            for err in exc.errors():
                loc = " → ".join(str(segment) for segment in err["loc"])
                errors.append(f"{idx}.{loc}: {err['msg']}")
            continue

        # Topology-consistency checks
        if spec.is_moe:
            if spec.n_experts is None:
                errors.append(f"{idx}: MoE model must specify 'n_experts'")
            elif spec.n_experts < 1:
                errors.append(f"{idx}: 'n_experts' must be >= 1, got {spec.n_experts}")
            if spec.n_experts_active is None:
                errors.append(f"{idx}: MoE model must specify 'n_experts_active'")
            elif spec.n_experts_active < 1:
                errors.append(
                    f"{idx}: 'n_experts_active' must be >= 1, got {spec.n_experts_active}"
                )
            if spec.n_experts_active and spec.n_experts and spec.n_experts_active > spec.n_experts:
                errors.append(
                    f"{idx}: 'n_experts_active' ({spec.n_experts_active}) "
                    f"> 'n_experts' ({spec.n_experts})"
                )
        else:
            if spec.n_experts is not None:
                errors.append(f"{idx}: dense model must not specify 'n_experts'")
            if spec.n_experts_active is not None:
                errors.append(f"{idx}: dense model must not specify 'n_experts_active'")

        # Sanity checks
        if spec.n_params_billions <= 0:
            errors.append(
                f"{idx}: 'n_params_billions' must be positive, got {spec.n_params_billions}"
            )
        if spec.n_layers <= 0:
            errors.append(f"{idx}: 'n_layers' must be positive, got {spec.n_layers}")
        if spec.n_kv_heads <= 0:
            errors.append(f"{idx}: 'n_kv_heads' must be positive, got {spec.n_kv_heads}")
        if spec.head_dim <= 0:
            errors.append(f"{idx}: 'head_dim' must be positive, got {spec.head_dim}")
        if spec.context_default <= 0:
            errors.append(f"{idx}: 'context_default' must be positive, got {spec.context_default}")

        if not spec.available_quants:
            errors.append(f"{idx}: 'available_quants' must not be empty")
        for q in spec.available_quants:
            if q.value not in _VALID_QUANTS:
                errors.append(f"{idx}: unknown quant '{q.value}'")

        for task in spec.tasks:
            if task.value not in _EXPECTED_TASKS:
                errors.append(f"{idx}: unknown task '{task.value}'")

    for e in errors:
        print(f"FAIL  {e}", file=sys.stderr)
    return len(errors)


def main() -> None:
    """CLI entry point for ``python -m`` — parse args, validate, exit 0 or 1."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a models_catalog.json file against the ModelSpec schema."
    )
    parser.add_argument("--catalog", help="Path to models_catalog.json (default: auto-detect)")
    args = parser.parse_args()

    path = find_catalog_path(args.catalog)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("models", [])

    if not entries:
        print("FAIL  catalog contains no model entries", file=sys.stderr)
        sys.exit(1)

    count = validate_entries(entries)
    if count:
        print(f"\n{count} validation error(s)", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK  ({len(entries)} entries)")


if __name__ == "__main__":
    main()
