"""Unit tests for src/hotel_reservation/data/config_loader.py."""

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from loguru import logger

from src.hotel_reservation.data import config_loader


# --------------------------------------------------------------------------- #
#               Bridge Loguru → Standard Logging for pytest caplog            #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Forward Loguru logs to Python's logging system for pytest capture."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to Python's logging system."""
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                          Global pytest fixture                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Databricks-related env vars before and after each test."""
    for var in ["PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ["PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
#                         Tests for load_env()                                #
# --------------------------------------------------------------------------- #
def test_load_env_with_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that load_env reads PROFILE from .env when provided."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROFILE=my_profile\n")

    def fake_load_dotenv(dotenv_path: str, override: bool) -> None:
        os.environ["PROFILE"] = "my_profile"

    def fake_dotenv_values(path: str) -> dict[str, str]:
        return {"PROFILE": "my_profile"}

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with caplog.at_level(logging.DEBUG):
        host, token, profile = config_loader.load_env(str(env_file))

    assert profile == "my_profile"
    assert host is None and token is None
    assert "Loaded env" in caplog.text


def test_load_env_with_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that load_env reads host/token when no PROFILE is defined."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=https://dummy\nDATABRICKS_TOKEN=secret")

    def fake_load_dotenv(dotenv_path: str, override: bool) -> None:
        os.environ["DATABRICKS_HOST"] = "https://dummy"
        os.environ["DATABRICKS_TOKEN"] = "secret"

    def fake_dotenv_values(path: str) -> dict[str, str | None]:
        return {
            "DATABRICKS_HOST": "https://dummy",
            "DATABRICKS_TOKEN": "secret",
            "PROFILE": None,
        }

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with caplog.at_level(logging.DEBUG):
        host, token, profile = config_loader.load_env(str(env_file))

    assert host == "https://dummy"
    assert token == "secret"
    assert profile is None
    assert "Loaded env" in caplog.text


def test_load_env_missing_file(tmp_path: Path) -> None:
    """Test that load_env raises FileNotFoundError when .env file is missing."""
    env_file = tmp_path / "nope.env"
    with pytest.raises(FileNotFoundError):
        config_loader.load_env(str(env_file))


def test_load_env_missing_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that load_env raises OSError when no valid vars are found."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=\nDATABRICKS_TOKEN=\n")

    def fake_load_dotenv(dotenv_path: str, override: bool) -> None:
        """Simulate loading empty environment variables."""
        pass

    def fake_dotenv_values(path: str) -> dict[str, str]:
        """Return empty env dict (no profile/host/token)."""
        return {}

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with pytest.raises(OSError) as excinfo:
        config_loader.load_env(str(env_file))

    assert "Missing PROFILE" in str(excinfo.value)


# --------------------------------------------------------------------------- #
#                     Tests for load_project_config()                         #
# --------------------------------------------------------------------------- #
def test_load_project_config_valid(tmp_path: Path) -> None:
    """Test that YAML config loads correctly when env exists."""
    yaml_content: dict[str, Any] = {
        "dev": {"catalog_name": "cat_dev", "schema_name": "sch_dev"},
        "global_setting": {"owner": "team-mlops"},
    }
    config_file = tmp_path / "project.yml"
    config_file.write_text(yaml.safe_dump(yaml_content))

    env_config, global_config = config_loader.load_project_config(str(config_file), "dev")

    assert env_config == {"catalog_name": "cat_dev", "schema_name": "sch_dev"}
    assert "global_setting" in global_config
    assert "dev" not in global_config


def test_load_project_config_missing_env(tmp_path: Path) -> None:
    """Test that ValueError is raised when specified environment is missing."""
    yaml_content: dict[str, Any] = {"dev": {"x": 1}}
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValueError):
        config_loader.load_project_config(str(config_file), "prd")
