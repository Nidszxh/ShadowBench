"""Privacy guarantees for the calibration sync path (SECURITY.md / P5.3)."""

from __future__ import annotations

import pytest

from shadowbench.calibration.sync import FORBIDDEN_FIELDS, build_sync_payload
from tests.conftest import ProfileBuilder


def test_anonymized_profile_has_no_forbidden_fields(make_profile: ProfileBuilder) -> None:
    anon = make_profile().anonymized()
    assert not (set(anon) & FORBIDDEN_FIELDS)
    # Memory is bucketed, not exact.
    assert anon["ram_bucket_gb"] % 4 == 0


def test_build_payload_rejects_pii(make_profile: ProfileBuilder) -> None:
    with pytest.raises(ValueError, match="PII"):
        build_sync_payload(make_profile(), runs=[{"model_id": "x", "hostname": "my-laptop"}])


def test_build_payload_ok_without_pii(make_profile: ProfileBuilder) -> None:
    payload = build_sync_payload(make_profile(), runs=[{"model_id": "x", "measured_tps": 12.0}])
    assert payload["runs"]
    assert "hardware" in payload
