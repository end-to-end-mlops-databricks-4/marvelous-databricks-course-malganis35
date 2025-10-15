"""Unit tests for src/hotel_reservation/utils/databricks_utils.py."""

import json
import logging
import os
import subprocess
import sys
import types
from types import ModuleType
from typing import Never

import pytest
from loguru import logger


# --------------------------------------------------------------------------- #
#                  Simulate missing databricks.connect module                 #
# --------------------------------------------------------------------------- #
class DummyDatabricksSession:
    """Dummy replacement for DatabricksSession used in tests."""

    builder: object | None = None  # Will be replaced dynamically


# Fake modules injected into sys.modules
fake_databricks_module: ModuleType = types.ModuleType("databricks")
fake_connect_submodule: ModuleType = types.ModuleType("databricks.connect")
fake_connect_submodule.DatabricksSession = DummyDatabricksSession
fake_databricks_module.connect = fake_connect_submodule

sys.modules["databricks"] = fake_databricks_module
sys.modules["databricks.connect"] = fake_connect_submodule


# Import now that the fake module exists
from src.hotel_reservation.utils import databricks_utils  # noqa: E402


# --------------------------------------------------------------------------- #
#                     Loguru → Standard logging bridge                        #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Forward Loguru messages to Python's logging system."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a LogRecord into standard logging."""
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                                   Fixtures                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Databricks-related environment variables are reset between tests."""
    for var in ["DATABRICKS_COMPUTE", "DATABRICKS_CLUSTER_ID"]:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ["DATABRICKS_COMPUTE", "DATABRICKS_CLUSTER_ID"]:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
#                                   Dummies                                   #
# --------------------------------------------------------------------------- #
class DummySpark:
    """Dummy Spark session placeholder."""

    def __init__(self, mode: str = "local") -> None:
        """Initialize with a mode indicator (local/serverless/cluster)."""
        self.mode = mode


class DummyRemoteBuilder:
    """Dummy remote Spark builder (for Databricks Connect simulation)."""

    def __init__(self, mode: str) -> None:
        """Initialize with the selected mode."""
        self.mode = mode

    def getOrCreate(self) -> DummySpark:
        """Return a dummy Spark instance."""
        return DummySpark(mode=self.mode)


class DummyDatabricksBuilder:
    """Mocked DatabricksSession.builder behavior."""

    def remote(self, cluster_id: str | None = None, serverless: bool = False) -> DummyRemoteBuilder:
        """Simulate remote session creation for cluster or serverless modes."""
        if cluster_id:
            return DummyRemoteBuilder(mode=f"cluster:{cluster_id}")
        if serverless:
            return DummyRemoteBuilder(mode="serverless")
        return DummyRemoteBuilder(mode="default")


class DummySparkBuilder:
    """Simulate SparkSession.builder with optional failure."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize with optional failure flag."""
        self.should_fail = should_fail
        self.called = False

    def getOrCreate(self) -> DummySpark:
        """Return dummy Spark or raise exception based on should_fail."""
        self.called = True
        if self.should_fail:
            raise Exception("Local Spark failure")
        return DummySpark(mode="local")


# --------------------------------------------------------------------------- #
#                                    Tests                                    #
# --------------------------------------------------------------------------- #
def test_create_spark_session_local_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test successful creation of a local Spark session."""
    dummy_builder = DummySparkBuilder(should_fail=False)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "local"
    assert "Local Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_serverless(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test fallback to Databricks serverless compute on local Spark failure."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "serverless"

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "Falling back to Databricks Connect" in caplog.text
    assert "Serverless" in caplog.text
    assert "Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_cluster(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test fallback to Databricks cluster mode on local Spark failure."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "cluster"
    os.environ["DATABRICKS_CLUSTER_ID"] = "clu-123"

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "cluster:clu-123"
    assert "Connecting to Databricks cluster" in caplog.text
    assert "Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_no_compute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test defaulting to serverless when no compute mode is defined."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "defaulting to serverless" in caplog.text or "Serverless" in caplog.text
    assert "Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_invalid_compute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test invalid compute mode triggers fallback to serverless."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "foobar"

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "No compute specified" in caplog.text or "defaulting to serverless" in caplog.text


def test_get_databricks_token_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test successful token generation via Databricks CLI."""
    dummy_output: dict[str, str] = {"token_value": "abc123", "expiry": "2030-01-01T00:00:00Z"}

    class DummyCompletedProcess:
        def __init__(self) -> None:
            self.stdout = json.dumps(dummy_output)
            self.returncode = 0

    def dummy_run(*args: object, **kwargs: object) -> DummyCompletedProcess:
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", dummy_run)

    with caplog.at_level("INFO"):
        result = databricks_utils.get_databricks_token("https://dummy-host")

    assert result == dummy_output
    assert "Temporary token acquired" in caplog.text
    assert "🔑 Automatically generating a Databricks temporary token" in caplog.text


def test_get_databricks_token_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that subprocess.run failure raises a CalledProcessError."""

    def dummy_run(*args: object, **kwargs: object) -> "Never":
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    monkeypatch.setattr(subprocess, "run", dummy_run)

    with pytest.raises(subprocess.CalledProcessError):
        databricks_utils.get_databricks_token("https://dummy-host")


def test_is_databricks_true_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment detection logic."""
    # False when variable is absent
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert not databricks_utils.is_databricks()

    # True when variable is present
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "13.3")
    assert databricks_utils.is_databricks()
