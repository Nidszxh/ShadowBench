"""Model catalog loader. Reads ``models_catalog.json`` with filesystem → embedded fallback."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from shadowbench._catalog_data import CATALOG_JSON
from shadowbench.common.types import Task
from shadowbench.predictor.models import ModelSpec


def _find_catalog() -> dict[str, Any]:
    """Locate catalog data: env override → repo checkout → embedded fallback."""
    if override := os.environ.get("SHADOWBENCH_CATALOG_PATH"):
        return json.loads(Path(override).read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "datasets" / "models_catalog.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return CATALOG_JSON


@lru_cache(maxsize=1)
def load_catalog() -> tuple[ModelSpec, ...]:
    """Load and validate every model spec. Cached for the process lifetime."""
    return tuple(ModelSpec.model_validate(entry) for entry in _find_catalog()["models"])


def candidates_for_task(task: Task) -> list[ModelSpec]:
    """Return catalog models tagged for a given task."""
    return [spec for spec in load_catalog() if task in spec.tasks]
