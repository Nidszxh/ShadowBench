"""Tests for the CLI (shadowbench.cli)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_catalog_validate_ok(cli_runner: CliRunner, tmp_path: Path) -> None:
    import shadowbench.cli

    catalog = tmp_path / "models_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "test/Model-7B",
                        "name": "Model-7B",
                        "topology": "dense",
                        "tasks": ["general"],
                        "n_params_billions": 7.0,
                        "n_layers": 32,
                        "n_kv_heads": 32,
                        "head_dim": 128,
                        "context_default": 4096,
                        "available_quants": ["Q4_K_M"],
                    }
                ]
            }
        )
        + "\n"
    )
    result = cli_runner.invoke(
        shadowbench.cli.app, ["catalog", "validate", "--catalog", str(catalog)]
    )
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_catalog_validate_fail(cli_runner: CliRunner, tmp_path: Path) -> None:
    import shadowbench.cli

    catalog = tmp_path / "models_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "test/Bad-Model",
                        "topology": "dense",
                        "tasks": ["general"],
                        "n_params_billions": -1,
                        "n_layers": 32,
                        "n_kv_heads": 32,
                        "head_dim": 128,
                        "context_default": 4096,
                        "available_quants": ["Q4_K_M"],
                    }
                ]
            }
        )
        + "\n"
    )
    result = cli_runner.invoke(
        shadowbench.cli.app, ["catalog", "validate", "--catalog", str(catalog)]
    )
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
