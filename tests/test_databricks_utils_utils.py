import sys
import types
import os
import logging
import pytest
from loguru import logger

# --- Simule le module databricks.connect avant import ---
fake_databricks_module = types.ModuleType("databricks")
fake_connect_submodule = types.ModuleType("databricks.connect")

class DummyDatabricksSession:
    builder = None  # sera remplacé par monkeypatch

fake_connect_submodule.DatabricksSession = DummyDatabricksSession
fake_databricks_module.connect = fake_connect_submodule
sys.modules["databricks"] = fake_databricks_module
sys.modules["databricks.connect"] = fake_connect_submodule

# Maintenant on peut importer sans erreur
from src.mlops_course.utils import databricks_utils


# --- Connect Loguru to standard logging for caplog ---
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ["DATABRICKS_COMPUTE", "DATABRICKS_CLUSTER_ID"]:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ["DATABRICKS_COMPUTE", "DATABRICKS_CLUSTER_ID"]:
        monkeypatch.delenv(var, raising=False)


# --- Dummies ---
class DummySpark:
    def __init__(self, mode="local"):
        self.mode = mode


class DummyRemoteBuilder:
    def __init__(self, mode):
        self.mode = mode

    def getOrCreate(self):
        return DummySpark(mode=self.mode)


class DummyDatabricksBuilder:
    def remote(self, cluster_id=None, serverless=False):
        if cluster_id:
            return DummyRemoteBuilder(mode=f"cluster:{cluster_id}")
        if serverless:
            return DummyRemoteBuilder(mode="serverless")
        return DummyRemoteBuilder(mode="default")


class DummySparkBuilder:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.called = False

    def getOrCreate(self):
        self.called = True
        if self.should_fail:
            raise Exception("Local Spark failure")
        return DummySpark(mode="local")


# --- Tests ---
def test_create_spark_session_local_success(monkeypatch, caplog):
    dummy_builder = DummySparkBuilder(should_fail=False)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "local"
    assert "Local Spark session initialized successfully" in caplog.text


def test_create_spark_session_local_failure_serverless(monkeypatch, caplog):
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


def test_create_spark_session_local_failure_cluster(monkeypatch, caplog):
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


def test_create_spark_session_local_failure_no_compute(monkeypatch, caplog):
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"
    assert "defaulting to serverless" in caplog.text or "Serverless" in caplog.text
    assert "Spark session initialized successfully" in caplog.text

def test_create_spark_session_local_failure_invalid_compute(monkeypatch, caplog):
    """Local Spark fails → compute mode invalid triggers 'defaulting to serverless'."""
    dummy_builder = DummySparkBuilder(should_fail=True)
    monkeypatch.setattr(databricks_utils, "SparkSession", type("S", (), {"builder": dummy_builder}))
    monkeypatch.setattr(databricks_utils, "DatabricksSession", type("D", (), {"builder": DummyDatabricksBuilder()}))

    os.environ["DATABRICKS_COMPUTE"] = "foobar"  # invalid mode

    with caplog.at_level(logging.INFO):
        spark = databricks_utils.create_spark_session()

    assert isinstance(spark, DummySpark)
    assert spark.mode == "serverless"  # fallback triggered
    assert "No compute specified" in caplog.text or "defaulting to serverless" in caplog.text
