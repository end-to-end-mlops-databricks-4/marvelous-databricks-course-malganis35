"""Unit tests for src/mlops_course/utils/env_loader.py."""

import logging
import os
from pathlib import Path

import pytest
from loguru import logger

from src.mlops_course.utils import env_loader


# --------------------------------------------------------------------------- #
#                          Loguru → Standard Logging Bridge                    #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Propagate Loguru log records to Python's logging system."""

    def emit(self, record: logging.LogRecord) -> None:
        """Send a LogRecord to the standard logging handlers."""
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                                  Fixtures                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def cleanup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean up environment variables between tests."""
    for key in ["PROFILE", "DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST"]:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in ["PROFILE", "DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST"]:
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
#                                    Tests                                    #
# --------------------------------------------------------------------------- #
def test_load_environment_with_existing_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure environment variables load correctly from a valid .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROFILE=my_profile\nDATABRICKS_HOST=https://dummy")

    called: dict[str, object] = {}

    def fake_load_dotenv(dotenv_path: Path, override: bool) -> None:
        """Simulate loading .env file."""
        called["path"] = dotenv_path
        called["override"] = override
        os.environ["PROFILE"] = "my_profile"
        os.environ["DATABRICKS_HOST"] = "https://dummy"

    monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    # Verify load_dotenv call and side effects
    assert called["path"] == env_file
    assert called["override"] is True
    assert "Loaded environment" in caplog.text
    assert "Databricks profile set to" in caplog.text
    assert os.getenv("DATABRICKS_CONFIG_PROFILE") == "my_profile"
    assert os.getenv("DATABRICKS_HOST") == "https://dummy"


def test_load_environment_with_missing_env_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Ensure missing .env file logs a warning but still reports profile info."""
    env_file = tmp_path / "missing.env"

    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    assert "not found" in caplog.text or "⚠️" in caplog.text
    assert "Databricks profile set to" in caplog.text
    assert "Databricks host" in caplog.text


def test_load_environment_without_argument_uses_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure default behavior loads ENV_FILE or .env in project root."""
    default_env = tmp_path / ".env"
    default_env.write_text("PROFILE=default_profile")

    monkeypatch.setenv("ENV_FILE", str(default_env))

    with caplog.at_level(logging.INFO):
        env_loader.load_environment()

    assert "Loaded environment" in caplog.text
    assert os.getenv("DATABRICKS_CONFIG_PROFILE") == "default_profile"


def test_load_environment_without_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure .env file without PROFILE does not set DATABRICKS_CONFIG_PROFILE."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=https://example")

    def fake_load_dotenv(dotenv_path: Path, override: bool) -> None:
        """Mock dotenv loader for minimal .env file."""
        os.environ["DATABRICKS_HOST"] = "https://example"

    monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    assert os.getenv("DATABRICKS_CONFIG_PROFILE") is None
    assert "Databricks host: https://example" in caplog.text
