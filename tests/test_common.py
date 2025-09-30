"""Unit tests for hotel_reservation.marvelous.common."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------
# Mock Databricks and PySpark modules before import
# ---------------------------------------------------------------------
mock_pyspark = types.ModuleType("pyspark")
mock_pyspark_dbutils = types.ModuleType("pyspark.dbutils")
mock_pyspark_sql = types.ModuleType("pyspark.sql")

mock_spark_session = MagicMock(name="SparkSession")
mock_pyspark_sql.SparkSession = mock_spark_session
mock_dbutils_class = MagicMock(name="DBUtils")
mock_pyspark_dbutils.DBUtils = mock_dbutils_class

sys.modules["pyspark"] = mock_pyspark
sys.modules["pyspark.dbutils"] = mock_pyspark_dbutils
sys.modules["pyspark.sql"] = mock_pyspark_sql

# Mock databricks.sdk.WorkspaceClient
mock_databricks_sdk = types.ModuleType("databricks.sdk")
mock_workspace_client = MagicMock(name="WorkspaceClient")
mock_databricks_sdk.WorkspaceClient = mock_workspace_client
sys.modules["databricks.sdk"] = mock_databricks_sdk

# ---------------------------------------------------------------------
# Import module under test (E402 fix: move all mocks above)
# ---------------------------------------------------------------------
from hotel_reservation.marvelous import common  # noqa: E402


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_is_databricks_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return True when DATABRICKS_RUNTIME_VERSION is set."""
    monkeypatch.setitem(os.environ, "DATABRICKS_RUNTIME_VERSION", "14.3")
    assert common.is_databricks() is True


def test_is_databricks_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return False when DATABRICKS_RUNTIME_VERSION is missing."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert common.is_databricks() is False


@patch("hotel_reservation.marvelous.common.SparkSession")
@patch("hotel_reservation.marvelous.common.DBUtils")
def test_get_dbr_token_returns_token(
    mock_dbutils: MagicMock,
    mock_spark: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Should retrieve a mocked Databricks token."""
    monkeypatch.setitem(os.environ, "DATABRICKS_RUNTIME_VERSION", "14.3")

    # Mock Spark + DBUtils chain
    mock_token = "mocked-token"
    mock_api_token = MagicMock()
    mock_api_token.get.return_value = mock_token
    mock_context = MagicMock()
    mock_context.apiToken.return_value = mock_api_token
    mock_dbutils.return_value.notebook.entry_point.getDbutils().notebook().getContext.return_value = mock_context

    result = common.get_dbr_token()
    assert result == mock_token


def test_get_dbr_token_raises_when_not_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise ValueError when not in Databricks environment."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    with pytest.raises(ValueError, match="only supported on Databricks"):
        common.get_dbr_token()


@patch("hotel_reservation.marvelous.common.WorkspaceClient")
def test_get_dbr_host_returns_host(mock_ws_client: MagicMock) -> None:
    """Should return the workspace host from WorkspaceClient."""
    mock_instance = MagicMock()
    mock_instance.config.host = "https://mock-workspace"
    mock_ws_client.return_value = mock_instance

    result = common.get_dbr_host()
    assert result == "https://mock-workspace"
    mock_ws_client.assert_called_once()


def test_create_parser_data_ingestion_command() -> None:
    """Should correctly parse 'data_ingestion' command with common args."""
    args = [
        "data_ingestion",
        "--root_path",
        "/mnt/root",
        "--env",
        "/mnt/env",
        "--is_test",
        "1",
    ]
    parsed = common.create_parser(args)
    assert parsed.command == "data_ingestion"
    assert parsed.root_path == "/mnt/root"
    assert parsed.is_test == 1


def test_create_parser_model_train_register_command() -> None:
    """Should parse model_train_register command with all required args."""
    args = [
        "model_train_register",
        "--root_path",
        "/mnt/root",
        "--env",
        "/mnt/env",
        "--is_test",
        "0",
        "--git_sha",
        "abc123",
        "--job_run_id",
        "run_1",
        "--branch",
        "main",
    ]
    parsed = common.create_parser(args)
    assert parsed.command == "model_train_register"
    assert parsed.git_sha == "abc123"
    assert parsed.branch == "main"


def test_create_parser_post_commit_check_command() -> None:
    """Should parse post_commit_check command correctly."""
    args = [
        "post_commit_check",
        "--git_sha",
        "abc123",
        "--job_run_id",
        "run_22",
        "--job_id",
        "job_99",
        "--repo",
        "hotel_reservation",
        "--org",
        "openai",
    ]
    parsed = common.create_parser(args)
    assert parsed.command == "post_commit_check"
    assert parsed.repo == "hotel_reservation"
    assert parsed.org == "openai"


def test_create_parser_monitor_command() -> None:
    """Should parse monitor command with required common args."""
    args = [
        "monitor",
        "--root_path",
        "/mnt/root",
        "--env",
        "/mnt/env",
        "--is_test",
        "1",
    ]
    parsed = common.create_parser(args)
    assert parsed.command == "monitor"
    assert parsed.env == "/mnt/env"


def test_create_parser_invalid_command_raises() -> None:
    """Should raise SystemExit for missing subcommand."""
    with pytest.raises(SystemExit):
        common.create_parser([])
