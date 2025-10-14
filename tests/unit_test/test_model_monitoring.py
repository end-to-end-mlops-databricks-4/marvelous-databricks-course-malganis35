"""Unit tests for the model monitoring module with PySpark mocks."""

# ruff: noqa: E402, I001

import sys
import types

# --- Fake PySpark modules so pytest can import without real Spark ---
pyspark = types.ModuleType("pyspark")

# Create dummy submodules
sql_module = types.ModuleType("pyspark.sql")
functions_module = types.ModuleType("pyspark.sql.functions")
types_module = types.ModuleType("pyspark.sql.types")


# --- Fake SparkSession class ---
class FakeSparkSession:
    """Fake SparkSession used for unit testing without real Spark."""

    pass


sql_module.SparkSession = FakeSparkSession


# --- Fake PySpark SQL type classes ---
class FakeType:
    """Fake PySpark SQL type placeholder."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize fake type."""
        pass


types_module.ArrayType = FakeType
types_module.DoubleType = FakeType
types_module.IntegerType = FakeType
types_module.StringType = FakeType
types_module.StructField = FakeType
types_module.StructType = FakeType


# --- Fake PySpark SQL functions ---
def fake_function(*args: object, **kwargs: object) -> str:
    """Return a MagicMock-like placeholder for any Spark SQL function."""
    return f"<fake pyspark function call: args={args}, kwargs={kwargs}>"


functions_module.from_json = fake_function
functions_module.col = fake_function
functions_module.explode = fake_function
functions_module.lit = fake_function
functions_module.when = fake_function

# Link submodules correctly (this fixes the ImportError)
pyspark.sql = sql_module
sql_module.functions = functions_module
sql_module.types = types_module

# Register all dummy modules in sys.modules
sys.modules["pyspark"] = pyspark
sys.modules["pyspark.sql"] = sql_module
sys.modules["pyspark.sql.functions"] = functions_module
sys.modules["pyspark.sql.types"] = types_module

# ---------------------------------------------------------------------
# Import class under test (ignore E402 intentionally)
# ---------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import NotFound

from hotel_reservation.visualization.monitoring import (
    create_monitoring_table,
    create_or_refresh_monitoring,
)


@pytest.fixture(autouse=True)
def patch_pyspark_functions(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch pyspark.sql.functions used in monitoring.py.

    This avoids `'str' object has no attribute 'cast'` errors during unit testing.
    """
    mock_col = MagicMock()
    mock_col.cast.return_value = mock_col
    mock_col.__mul__.return_value = mock_col
    mock_col.alias.return_value = mock_col
    mock_col.__getitem__.return_value = mock_col  # for F.col(...)[0]

    # Patch only the actual functions present in pyspark.sql.functions
    monkeypatch.setattr("hotel_reservation.visualization.monitoring.F.col", lambda _: mock_col)
    monkeypatch.setattr("hotel_reservation.visualization.monitoring.F.lit", lambda _: mock_col)
    monkeypatch.setattr("hotel_reservation.visualization.monitoring.F.when", lambda *args, **kwargs: mock_col)
    monkeypatch.setattr("hotel_reservation.visualization.monitoring.F.from_json", lambda *args, **kwargs: mock_col)
    monkeypatch.setattr("hotel_reservation.visualization.monitoring.F.explode", lambda *args, **kwargs: mock_col)

    return mock_col


@pytest.fixture
def mock_config() -> object:
    """Return a dummy configuration fixture for testing."""

    class Config:
        """Configuration stub."""

        catalog_name = "test_catalog"
        schema_name = "test_schema"

    return Config()


@pytest.fixture
def mock_spark() -> MagicMock:
    """Mock a SparkSession and related DataFrames."""
    spark = MagicMock()
    mock_df = MagicMock()
    mock_df.count.return_value = 5
    mock_df.withColumn.return_value = mock_df
    mock_df.select.return_value = mock_df
    mock_df.dropna.return_value = mock_df
    mock_df.write.format.return_value.mode.return_value.saveAsTable.return_value = None

    spark.sql.return_value = mock_df
    spark.table.return_value = mock_df

    return spark


@pytest.fixture
def mock_workspace() -> MagicMock:
    """Mock the WorkspaceClient."""
    ws = MagicMock()
    ws.quality_monitors.get = MagicMock()
    ws.quality_monitors.run_refresh = MagicMock()
    ws.quality_monitors.create = MagicMock()
    return ws


# -------------------------------------------------------------------------
# Tests for create_or_refresh_monitoring
# -------------------------------------------------------------------------


def test_create_or_refresh_monitoring_no_records(
    mock_config: object, mock_spark: MagicMock, mock_workspace: MagicMock
) -> None:
    """Test: no records -> function stops without writing or creating anything."""
    mock_df = MagicMock()
    mock_df.count.return_value = 0
    mock_spark.sql.return_value = mock_df

    create_or_refresh_monitoring(mock_config, mock_spark, mock_workspace)

    mock_spark.sql.assert_called_once()
    mock_df.write.format.assert_not_called()
    mock_workspace.quality_monitors.get.assert_not_called()


def test_create_or_refresh_monitoring_success_refresh(
    mock_config: object, mock_spark: MagicMock, mock_workspace: MagicMock
) -> None:
    """Test: records exist and monitor already exists (refresh case)."""
    mock_spark.sql.return_value.count.return_value = 10
    create_or_refresh_monitoring(mock_config, mock_spark, mock_workspace)

    mock_spark.sql.assert_called_once()
    mock_spark.table.assert_called_once_with("test_catalog.test_schema.model_monitoring")
    mock_workspace.quality_monitors.get.assert_called_once()
    mock_workspace.quality_monitors.run_refresh.assert_called_once()


@patch("hotel_reservation.visualization.monitoring.create_monitoring_table")
def test_create_or_refresh_monitoring_creates_table(
    mock_create_monitoring_table: MagicMock,
    mock_config: object,
    mock_spark: MagicMock,
    mock_workspace: MagicMock,
) -> None:
    """Test: NotFound raised → monitoring table is created."""
    mock_spark.sql.return_value.count.return_value = 5
    mock_workspace.quality_monitors.get.side_effect = NotFound("not found")

    create_or_refresh_monitoring(mock_config, mock_spark, mock_workspace)

    mock_create_monitoring_table.assert_called_once_with(config=mock_config, spark=mock_spark, workspace=mock_workspace)
    mock_workspace.quality_monitors.run_refresh.assert_not_called()


def test_create_or_refresh_monitoring_all_null_predictions(
    mock_config: object, mock_spark: MagicMock, mock_workspace: MagicMock
) -> None:
    """Test: all predictions are null → logs a warning but still writes."""
    mock_df = MagicMock()
    call_counts = {"n": 0}

    def fake_count() -> int:
        call_counts["n"] += 1
        if call_counts["n"] == 1:
            return 5
        elif call_counts["n"] == 2:
            return 0
        return 5

    mock_df.count.side_effect = fake_count
    mock_df.withColumn.return_value = mock_df
    mock_df.select.return_value = mock_df
    mock_df.dropna.return_value = mock_df
    mock_df.write.format.return_value.mode.return_value.saveAsTable.return_value = None

    mock_spark.sql.return_value = mock_df
    mock_spark.table.return_value = mock_df

    create_or_refresh_monitoring(mock_config, mock_spark, mock_workspace)

    mock_df.write.format.assert_called_once()
    mock_workspace.quality_monitors.get.assert_called_once()


# -------------------------------------------------------------------------
# Tests for create_monitoring_table
# -------------------------------------------------------------------------


def test_create_monitoring_table_success(mock_config: object, mock_spark: MagicMock, mock_workspace: MagicMock) -> None:
    """Test: successful creation of the monitoring table."""
    create_monitoring_table(mock_config, mock_spark, mock_workspace)

    mock_workspace.quality_monitors.create.assert_called_once()
    mock_spark.sql.assert_called_once_with(
        "ALTER TABLE test_catalog.test_schema.model_monitoring SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )


def test_create_monitoring_table_parameters(
    mock_config: object, mock_spark: MagicMock, mock_workspace: MagicMock
) -> None:
    """Verify correct parameters are passed to quality_monitors.create."""
    create_monitoring_table(mock_config, mock_spark, mock_workspace)
    args, kwargs = mock_workspace.quality_monitors.create.call_args

    assert kwargs["table_name"] == "test_catalog.test_schema.model_monitoring"
    assert kwargs["output_schema_name"] == "test_catalog.test_schema"
    assert "MonitorInferenceLog" in str(kwargs["inference_log"].__class__)
