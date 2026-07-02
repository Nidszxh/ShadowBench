"""Accuracy report: predicted vs. actual.  [Phase 2 — P2.5]

Computes error metrics across ``datasets/golden.jsonl`` and emits a report published as a CI artifact. CI
fails when median error regresses past the configured band — the accuracy gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AccuracyReport:
    n_samples: int
    median_abs_pct_error: float
    p90_abs_pct_error: float
    within_band_fraction: float


def evaluate(  # pragma: no cover - Phase 2
    golden_path: str,
    *,
    error_band: float = 0.25,
) -> AccuracyReport:
    """Score the Predictor against the golden dataset. ``error_band`` is the ±fraction target."""
    raise NotImplementedError("Accuracy reporting lands in Phase 2 (P2.5).")
