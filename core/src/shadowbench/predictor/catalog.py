"""Model catalog loader.

Reads ``datasets/models_catalog.json`` — the known-model metadata table (topology, params, quants). Kept as a
data file, not hardcoded, so contributors can extend it via PR without touching code.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from shadowbench.common.errors import ModelNotFoundError
from shadowbench.common.types import Task
from shadowbench.predictor.models import ModelSpec


def _find_catalog() -> Path:
    """Locate ``models_catalog.json`` via env override, else by walking up to the repo's ``datasets/``."""
    if override := os.environ.get("SHADOWBENCH_CATALOG_PATH"):
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "datasets" / "models_catalog.json"
        if candidate.exists():
            return candidate
    raise ModelNotFoundError(
        "Could not locate datasets/models_catalog.json; set SHADOWBENCH_CATALOG_PATH."
    )


@lru_cache(maxsize=1)
def load_catalog() -> tuple[ModelSpec, ...]:
    """Load and validate every model spec. Cached for the process lifetime."""
    raw = json.loads(_find_catalog().read_text(encoding="utf-8"))
    return tuple(ModelSpec.model_validate(entry) for entry in raw["models"])


def candidates_for_task(task: Task) -> list[ModelSpec]:
    """Return catalog models tagged for a given task."""
    return [spec for spec in load_catalog() if task in spec.tasks]
