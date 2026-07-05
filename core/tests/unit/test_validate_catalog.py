"""Tests for catalog validation (predictor/validate_catalog.py)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from shadowbench.predictor.validate_catalog import find_catalog_path, main, validate_entries

_VALID_DENSE = {
    "id": "test/Model-7B",
    "name": "Model-7B",
    "topology": "dense",
    "tasks": ["general", "coding"],
    "n_params_billions": 7.0,
    "n_layers": 32,
    "n_kv_heads": 32,
    "head_dim": 128,
    "context_default": 4096,
    "available_quants": ["Q4_K_M", "Q8_0"],
}

_VALID_MOE = {
    "id": "test/MoE-Model-8x7B",
    "name": "MoE-Model-8x7B",
    "topology": "moe",
    "tasks": ["general"],
    "n_params_billions": 46.0,
    "n_params_active_billions": 12.0,
    "n_layers": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "context_default": 32768,
    "available_quants": ["Q2_K", "Q3_K_M", "Q4_K_M"],
    "n_experts": 8,
    "n_experts_active": 2,
}


def test_validate_entries_on_catalog() -> None:
    entries: list[dict[str, object]] = [_VALID_DENSE, _VALID_MOE]
    assert validate_entries(entries) == 0


def test_validate_entries_missing_required_field() -> None:
    entry = dict(_VALID_DENSE)
    del entry["n_layers"]
    assert validate_entries([entry]) == 1


def test_validate_entries_wrong_type() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_layers"] = "not-an-int"
    assert validate_entries([entry]) == 1


def test_validate_entries_moe_missing_n_experts() -> None:
    entry = dict(_VALID_MOE)
    del entry["n_experts"]
    assert validate_entries([entry]) == 1


def test_validate_entries_dense_has_n_experts() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_experts"] = 8
    assert validate_entries([entry]) == 1


def test_validate_entries_negative_params() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_params_billions"] = -1.0
    assert validate_entries([entry]) == 1


def test_validate_entries_empty_quants() -> None:
    entry = dict(_VALID_DENSE)
    entry["available_quants"] = []
    assert validate_entries([entry]) == 1


def test_validate_entries_unknown_quant() -> None:
    entry = dict(_VALID_DENSE)
    entry["available_quants"] = ["Q10_K"]
    assert validate_entries([entry]) == 1


def test_find_catalog_path_override() -> None:
    data = {"models": [_VALID_DENSE, _VALID_MOE]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = Path(f.name)
    try:
        result = find_catalog_path(str(tmp))
        assert result.resolve() == tmp.resolve()
    finally:
        tmp.unlink()


def test_find_catalog_path_override_not_found() -> None:
    with pytest.raises(SystemExit):
        find_catalog_path("/nonexistent/path.json")


def test_find_catalog_path_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"models": [_VALID_DENSE]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = Path(f.name)
    try:
        monkeypatch.setenv("SHADOWBENCH_CATALOG_PATH", str(tmp))
        result = find_catalog_path()
        assert result.resolve() == tmp.resolve()
    finally:
        tmp.unlink()


def test_find_catalog_path_env_var_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHADOWBENCH_CATALOG_PATH", "/nonexistent/path.json")
    with pytest.raises(SystemExit):
        find_catalog_path("/another/nonexistent/path")


def test_validate_entries_moe_n_experts_zero() -> None:
    entry = dict(_VALID_MOE)
    entry["n_experts"] = 0
    assert validate_entries([entry]) == 1


def test_validate_entries_moe_missing_n_experts_active() -> None:
    entry = dict(_VALID_MOE)
    del entry["n_experts_active"]
    assert validate_entries([entry]) == 1


def test_validate_entries_moe_n_experts_active_zero() -> None:
    entry = dict(_VALID_MOE)
    entry["n_experts_active"] = 0
    assert validate_entries([entry]) == 1


def test_validate_entries_moe_active_exceeds_experts() -> None:
    entry = dict(_VALID_MOE)
    entry["n_experts"] = 4
    entry["n_experts_active"] = 8
    assert validate_entries([entry]) == 1


def test_validate_entries_dense_has_n_experts_active() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_experts_active"] = 2
    assert validate_entries([entry]) == 1


def test_validate_entries_negative_n_layers() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_layers"] = -1
    assert validate_entries([entry]) == 1


def test_validate_entries_zero_n_kv_heads() -> None:
    entry = dict(_VALID_DENSE)
    entry["n_kv_heads"] = 0
    assert validate_entries([entry]) == 1


def test_validate_entries_negative_head_dim() -> None:
    entry = dict(_VALID_DENSE)
    entry["head_dim"] = -128
    assert validate_entries([entry]) == 1


def test_validate_entries_zero_context() -> None:
    entry = dict(_VALID_DENSE)
    entry["context_default"] = 0
    assert validate_entries([entry]) == 1


def test_validate_entries_unknown_task() -> None:
    entry = dict(_VALID_DENSE)
    entry["tasks"] = ["bogus_task"]
    assert validate_entries([entry]) == 1


def test_main_valid(tmp_path: Path) -> None:
    catalog = tmp_path / "models_catalog.json"
    catalog.write_text(json.dumps({"models": [_VALID_DENSE]}) + "\n", encoding="utf-8")
    sys.argv = ["prog", "--catalog", str(catalog)]
    main()


def test_main_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog = tmp_path / "models_catalog.json"
    catalog.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--catalog", str(catalog)])
    with pytest.raises(SystemExit):
        main()


def test_main_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = dict(_VALID_MOE)
    del entry["n_experts"]
    catalog = tmp_path / "models_catalog.json"
    catalog.write_text(json.dumps({"models": [entry]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--catalog", str(catalog)])
    with pytest.raises(SystemExit):
        main()
