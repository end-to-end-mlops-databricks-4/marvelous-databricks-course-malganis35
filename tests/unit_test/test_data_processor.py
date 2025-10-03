"""Unit tests for the DataProcessor class in hotel_reservation.feature.data_processor."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Patch pyspark before importing DataProcessor
# ---------------------------------------------------------------------
mock_pyspark = types.ModuleType("pyspark")
mock_sql = types.ModuleType("pyspark.sql")
mock_sql.SparkSession = MagicMock()

# Create mock pyspark.sql.functions
mock_functions = types.ModuleType("pyspark.sql.functions")
mock_functions.current_timestamp = MagicMock()
mock_functions.to_utc_timestamp = MagicMock()

mock_pyspark.sql = mock_sql
sys.modules["pyspark"] = mock_pyspark
sys.modules["pyspark.sql"] = mock_sql
sys.modules["pyspark.sql.functions"] = mock_functions

# ---------------------------------------------------------------------
# Import now the module (local imports must come after patch)
# ---------------------------------------------------------------------
from hotel_reservation.feature.data_processor import (  # noqa: E402
    DataProcessor,
    generate_synthetic_data,
    generate_test_data,
)
from hotel_reservation.utils.config import ProjectConfig, Tags  # noqa: E402


@pytest.fixture
def sample_config() -> ProjectConfig:
    """Provide a minimal valid ProjectConfig instance for testing."""
    return ProjectConfig(
        num_features=["num1"],
        cat_features=["cat1"],
        target="booking_status",
        catalog_name="catalog",
        schema_name="schema",
        parameters={"C": 1.0},
        raw_data_file="dummy.csv",
        train_table="train_table",
        test_table="test_table",
        experiment_name_basic="/dummy/exp/basic",
        experiment_name_custom="/dummy/exp/custom",
        model_name="dummy_model",
        model_name_fe="dummy_model",
        model_type="logistic-regression",
        feature_table_name="mock_feature_table",
        feature_function_name="mock_feature_function",
        experiment_name_fe="/Users/mock/fe",
        endpoint_name="endpoint_model",
        endpoint_name_fe="endpoint_fe_model",
    )


@pytest.fixture
def sample_tags() -> Tags:
    """Fixture providing a valid Tags object for testing."""
    return Tags()


@pytest.fixture
def mock_config() -> ProjectConfig:
    """Provide a mock ProjectConfig with predefined values."""
    return ProjectConfig(
        env="dev",
        catalog_name="my_catalog",
        schema_name="my_schema",
        train_table="train_table",
        test_table="test_table",
        num_features=[],
        cat_features=[],
        target="booking_status",
        parameters={},
        raw_data_file="dummy.csv",
        experiment_name_basic="/Users/mock/basic",
        experiment_name_custom="/Users/mock/custom",
        model_name="mock_model",
        model_name_fe="dummy_model",
        model_type="logistic-regression",
        feature_table_name="mock_feature_table",
        feature_function_name="mock_feature_function",
        experiment_name_fe="/Users/mock/fe",
        endpoint_name="endpoint_model",
        endpoint_name_fe="endpoint_fe_model",
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide a sample pandas DataFrame mimicking hotel reservation data."""
    return pd.DataFrame(
        {
            "Booking_ID": ["ID1", "ID2"],
            "no_of_weekend_nights": [1, 2],
            "no_of_week_nights": [2, 3],
            "no_of_children": [0, 1],
            "arrival_year": [2023, 2023],
            "arrival_month": [5, 5],
            "arrival_date": [10, 11],
            "booking_status": ["Canceled", "Not_Canceled"],
            "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2"],
            "room_type_reserved": ["Room A", "Room B"],
            "market_segment_type": ["Online", "Offline"],
            "lead_time": [10, 20],
            "avg_price_per_room": [100.0, 150.0],
            "no_of_special_requests": [1, 2],
        }
    )


@pytest.fixture
def processor(sample_dataframe: pd.DataFrame, sample_config: ProjectConfig) -> DataProcessor:
    """Create a DataProcessor instance with mocked Spark session."""
    spark_mock = MagicMock()
    return DataProcessor(sample_dataframe.copy(), sample_config, spark_mock)


def test_preprocess_pipeline_runs(processor: DataProcessor) -> None:
    """Test that the preprocessing pipeline runs and creates expected columns."""
    df_processed = processor.preprocess()
    assert "arrival_date" not in df_processed.columns
    assert len(df_processed) == len(df_processed.drop_duplicates())


def test_split_data_shapes(processor: DataProcessor) -> None:
    """Test that the split_data method produces disjoint, complete splits."""
    processor.preprocess()
    train, test = processor.split_data(test_size=0.5, random_state=1)
    assert len(train) + len(test) == len(processor.df)
    assert not train.equals(test)


@patch("hotel_reservation.feature.data_processor.to_utc_timestamp")
@patch("hotel_reservation.feature.data_processor.current_timestamp")
def test_save_to_catalog_calls_spark_write(
    mock_current_ts: MagicMock, mock_to_utc_ts: MagicMock, processor: DataProcessor
) -> None:
    """Test that save_to_catalog calls Spark's saveAsTable twice."""
    mock_current_ts.return_value = MagicMock(name="current_timestamp")
    mock_to_utc_ts.return_value = MagicMock(name="utc_timestamp")

    processor.spark.createDataFrame.return_value.withColumn.return_value.write.mode.return_value.saveAsTable = (
        MagicMock()
    )

    df_train, df_test = processor.split_data()
    processor.save_to_catalog(df_train, df_test)

    calls = processor.spark.createDataFrame.return_value.withColumn.return_value.write.mode.return_value.saveAsTable.call_args_list
    assert len(calls) == 2


def test_log_and_scale_with_missing_columns(processor: DataProcessor) -> None:
    """Test _log_and_scale_numeric handles missing numeric columns gracefully."""
    processor.df.drop(columns=["lead_time", "avg_price_per_room"], inplace=True)
    processor._create_features()
    processor._log_and_scale_numeric()
    for col in ["total_nights", "no_of_special_requests"]:
        if col in processor.df.columns:
            assert np.isclose(processor.df[col].mean(), 0, atol=1e-6)


def test_enable_change_data_feed_calls_spark_sql(mock_config: ProjectConfig) -> None:
    """Test that enable_change_data_feed issues correct ALTER TABLE SQL."""
    mock_spark = MagicMock()
    processor = DataProcessor(pandas_df=pd.DataFrame(), config=mock_config, spark=mock_spark)

    processor.enable_change_data_feed()

    expected_train_sql = (
        f"ALTER TABLE {mock_config.catalog_name}.{mock_config.schema_name}.{mock_config.train_table} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )
    expected_test_sql = (
        f"ALTER TABLE {mock_config.catalog_name}.{mock_config.schema_name}.{mock_config.test_table} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )

    mock_spark.sql.assert_any_call(expected_train_sql)
    mock_spark.sql.assert_any_call(expected_test_sql)
    assert mock_spark.sql.call_count == 2


def test_generate_synthetic_data_basic() -> None:
    """Test that generate_synthetic_data creates a valid synthetic dataset."""
    df = pd.DataFrame(
        {
            "lead_time": [10, 20, 30],
            "avg_price_per_room": [100.0, 200.0, 150.0],
            "arrival_year": [2023, 2023, 2024],
            "booking_status": ["Canceled", "Not_Canceled", "Canceled"],
        }
    )

    synthetic = generate_synthetic_data(df, drift=False, num_rows=10)
    assert list(synthetic.columns) == list(df.columns)
    assert len(synthetic) == 10
    assert np.issubdtype(synthetic["arrival_year"].dtype, np.integer)
    assert synthetic["avg_price_per_room"].dtype == np.float64


def test_generate_synthetic_data_with_drift_changes_values() -> None:
    """Test that drift=True modifies numeric and year columns as expected."""
    df = pd.DataFrame(
        {
            "lead_time": [5, 10, 15],
            "avg_price_per_room": [50.0, 100.0, 150.0],
            "arrival_year": [2020, 2021, 2022],
        }
    )

    synthetic_drift = generate_synthetic_data(df, drift=True, num_rows=5)
    assert (synthetic_drift["avg_price_per_room"] > df["avg_price_per_room"].max()).any()
    assert (synthetic_drift["lead_time"] > df["lead_time"].max()).any()
    assert synthetic_drift["arrival_year"].between(pd.Timestamp.now().year - 2, pd.Timestamp.now().year).all()


def test_generate_test_data_delegates_to_generate_synthetic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that generate_test_data delegates correctly to generate_synthetic_data."""
    df = pd.DataFrame({"lead_time": [1, 2, 3]})
    mock_func = MagicMock(return_value=pd.DataFrame({"lead_time": [10, 20]}))
    monkeypatch.setattr("hotel_reservation.feature.data_processor.generate_synthetic_data", mock_func)

    result = generate_test_data(df, drift=True, num_rows=2)
    mock_func.assert_called_once_with(df, True, 2)
    assert isinstance(result, pd.DataFrame)
