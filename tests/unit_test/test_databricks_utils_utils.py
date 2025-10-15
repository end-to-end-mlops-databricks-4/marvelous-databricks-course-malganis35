"""Unit tests for src/hotel_reservation/utils/databricks_utils.py."""

import json
import logging
import os
import subprocess
import sys
import types
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Never

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


@pytest.fixture
def mock_spark_session(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, type]:
    """Fixture that mocks SparkSession with a builder and getActiveSession method."""

    class DummySpark:
        """Dummy Spark replacement."""

        def __init__(self, mode: str = "local") -> None:
            self.mode = mode

    class DummySparkBuilder:
        """Dummy Spark builder."""

        def __init__(self, should_fail: bool = False) -> None:
            self.should_fail = should_fail
            self.called = False

        def getOrCreate(self) -> DummySpark:
            """Return a DummySpark or raise failure."""
            self.called = True
            if self.should_fail:
                raise Exception("Local Spark failure")
            return DummySpark(mode="local")

    dummy_builder = DummySparkBuilder(should_fail=False)
    fake_spark_class = type(
        "S",
        (),
        {
            "builder": dummy_builder,
            "getActiveSession": staticmethod(lambda: None),
        },
    )

    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark_class)
    return dummy_builder, DummySpark


# --------------------------------------------------------------------------- #
#                                   Dummies                                   #
# --------------------------------------------------------------------------- #
class DummySpark:
    """Dummy Spark session placeholder."""

    def __init__(self, mode: str = "local") -> None:
        self.mode = mode

    def range(self, start: int, end: int) -> "DummySpark":
        """Simulate Spark range query (used for health check)."""
        return self

    def collect(self) -> list[int]:
        """Simulate Spark collect call."""
        return [1]


class DummyRemoteBuilder:
    """Simulate Databricks remote builder."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def getOrCreate(self) -> DummySpark:
        """Return a DummySpark for the given mode."""
        return DummySpark(mode=self.mode)


class DummyDatabricksBuilder:
    """Mocked DatabricksSession.builder behavior."""

    def __init__(self) -> None:
        self.last_mode = "serverless"  # valeur par défaut

    def remote(self, cluster_id: str | None = None, serverless: bool = False) -> DummyRemoteBuilder:
        """Simulate remote builder creation for serverless or cluster mode."""
        if cluster_id:
            self.last_mode = f"cluster:{cluster_id}"
            return DummyRemoteBuilder(mode=self.last_mode)
        if serverless:
            self.last_mode = "serverless"
            return DummyRemoteBuilder(mode=self.last_mode)
        self.last_mode = "default"
        return DummyRemoteBuilder(mode=self.last_mode)

    def getOrCreate(self) -> DummySpark:
        """Simulate builder.getOrCreate() — returns last known mode."""
        return DummySpark(mode=self.last_mode)


class DummySparkBuilder:
    """Simulate SparkSession.builder with optional failure."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.called = False

    def getOrCreate(self) -> DummySpark:
        """Return DummySpark or raise failure."""
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
    fake_spark = type(
        "S",
        (),
        {"builder": dummy_builder, "getActiveSession": staticmethod(lambda: None)},
    )
    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark)

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
    fake_spark = type(
        "S",
        (),
        {"builder": dummy_builder, "getActiveSession": staticmethod(lambda: None)},
    )
    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark)
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "serverless"

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "Serverless" in caplog.text
    assert "Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_cluster(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test fallback to Databricks cluster mode on local Spark failure."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    fake_spark = type(
        "S",
        (),
        {"builder": dummy_builder, "getActiveSession": staticmethod(lambda: None)},
    )
    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark)
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
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test defaulting to serverless when no compute mode is defined."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    fake_spark = type(
        "S",
        (),
        {"builder": dummy_builder, "getActiveSession": staticmethod(lambda: None)},
    )
    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark)
    monkeypatch.setattr(
        databricks_utils,
        "DatabricksSession",
        type("D", (), {"builder": DummyDatabricksBuilder()}),
    )

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    # ✅ Vérifie que le Spark créé est bien en mode "serverless"
    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"

    # ✅ Vérifie seulement que la session a bien été initialisée, pas le texte exact du log
    assert any(
        phrase in caplog.text
        for phrase in [
            "Spark session initialized successfully",
            "Spark session created",
            "Spark initialized",
        ]
    ), f"Expected a log about successful Spark initialization. Got:\n{caplog.text}"


def test_create_spark_session_local_failure_invalid_compute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test invalid compute mode triggers fallback to serverless."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    fake_spark = type(
        "S",
        (),
        {"builder": dummy_builder, "getActiveSession": staticmethod(lambda: None)},
    )
    monkeypatch.setattr(databricks_utils, "SparkSession", fake_spark)
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "foobar"

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "serverless" in caplog.text


def test_get_databricks_token_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test successful token generation via Databricks CLI."""
    dummy_output = {"token_value": "abc123", "expiry": "2030-01-01T00:00:00Z"}

    class DummyCompletedProcess:
        """Dummy subprocess result."""

        def __init__(self) -> None:
            self.stdout = json.dumps(dummy_output)
            self.returncode = 0

    def dummy_run(args: Sequence[str], **kwargs: object) -> DummyCompletedProcess:
        """Fake subprocess.run returning valid output."""
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", dummy_run)

    with caplog.at_level("INFO"):
        result = databricks_utils.get_databricks_token("https://dummy-host")

    assert result == dummy_output
    assert "Temporary token acquired" in caplog.text


def test_get_databricks_token_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that subprocess.run failure raises a CalledProcessError."""

    def dummy_run(args: Sequence[str], **kwargs: object) -> Never:
        """Fake subprocess.run that raises CalledProcessError."""
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
