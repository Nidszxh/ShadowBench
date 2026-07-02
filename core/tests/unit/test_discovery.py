"""End-to-end Requirement Discovery against the real seed catalog."""

from __future__ import annotations

import pytest

from shadowbench.common.errors import PredictorError
from shadowbench.common.types import Task, UserProfile
from shadowbench.predictor.catalog import candidates_for_task, load_catalog
from shadowbench.predictor.discovery import recommend
from tests.conftest import ProfileBuilder


def test_catalog_loads() -> None:
    specs = load_catalog()
    assert len(specs) >= 3
    assert any(s.is_moe for s in specs)


def test_coding_candidates_exist() -> None:
    assert candidates_for_task(Task.CODING)


def test_recommend_on_capable_gpu(make_profile: ProfileBuilder) -> None:
    profile = make_profile(vram_mb=24576, pcie_gbps=16.0)
    rec = recommend(profile, Task.CODING, UserProfile.INTELLIGENCE)
    assert rec.prediction.predicted_tps >= 5.0
    assert rec.flags.to_cli()


def test_speed_profile_prefers_throughput(make_profile: ProfileBuilder) -> None:
    profile = make_profile(vram_mb=8192, pcie_gbps=12.0)
    rec = recommend(profile, Task.CODING, UserProfile.SPEED)
    assert rec.prediction.predicted_tps >= 5.0


def test_no_gpu_tiny_ram_raises(make_profile: ProfileBuilder) -> None:
    profile = make_profile(with_gpu=False, ram_mb=2048, pcie_gbps=4.0)
    with pytest.raises(PredictorError):
        recommend(profile, Task.CODING, UserProfile.INTELLIGENCE)
