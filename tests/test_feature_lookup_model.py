"""Unit tests for FeatureLookUpModel in hotel_reservation.model.feature_lookup_model."""

from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Patch all heavy external dependencies before import
# ---------------------------------------------------------------------
mock_pyspark = types.ModuleType("pyspark")
mock_sql = types.ModuleType("pyspark.sql")
mock_sql.SparkSession = MagicMock()
mock_sql.DataFrame = MagicMock()

# Mock pyspark.sql.utils to avoid import errors from mlflow internals
mock_sql_utils = types.ModuleType("pyspark.sql.utils")
mock_sql_utils.AnalysisException = Exception

sys.modules["pyspark"] = mock_pyspark
sys.modules["pyspark.sql"] = mock_sql
sys.modules["pyspark.sql.utils"] = mock_sql_utils

# Databricks SDK + feature_engineering mocks
mock_databricks = types.ModuleType("databricks")
mock_fe_module = types.ModuleType("databricks.feature_engineering")
mock_fe_module.FeatureEngineeringClient = MagicMock()
mock_fe_module.FeatureLookup = MagicMock()
mock_fe_module.FeatureFunction = MagicMock()
sys.modules["databricks"] = mock_databricks
sys.modules["databricks.feature_engineering"] = mock_fe_module
mock_sdk = types.ModuleType("databricks.sdk")
mock_sdk.WorkspaceClient = MagicMock()
sys.modules["databricks.sdk"] = mock_sdk

# Mock MLflow globally
mock_mlflow = MagicMock()
sys.modules["mlflow"] = mock_mlflow
mock_mlflow.data = MagicMock()
mock_mlflow.models = MagicMock()
mock_mlflow.models.evaluate = MagicMock()
mock_mlflow.start_run = MagicMock()
mock_mlflow.set_experiment = MagicMock()
mock_mlflow.register_model = MagicMock()
mock_mlflow.tracking = MagicMock()
mock_mlflow.tracking.MlflowClient = MagicMock()
mock_mlflow.exceptions = types.SimpleNamespace(MlflowException=Exception)  # ✅ real exception class

# ---------------------------------------------------------------------
# Import class under test
# ---------------------------------------------------------------------
from hotel_reservation.model.feature_lookup_model import FeatureLookUpModel, Result
from hotel_reservation.utils.config import ProjectConfig, Tags


@pytest.fixture
def mock_config() -> ProjectConfig:
    """Provide a lightweight mock configuration."""
    return ProjectConfig(
        num_features=["lead_time"],
        cat_features=["room_type_reserved"],
        target="booking_status",
        catalog_name="mock_catalog",
        schema_name="mock_schema",
        parameters={"max_iter": 100},
        raw_data_file="mock.csv",
        train_table="train_table",
        test_table="test_table",
        feature_table_name="features_table",
        feature_function_name="feature_func",
        experiment_name_fe="/experiments/mock",
        experiment_name_basic="/experiments/basic",
        experiment_name_custom="/experiments/custom",
        model_name="mock_model",
        model_type="logistic-regression",
    )


@pytest.fixture
def mock_tags() -> Tags:
    """Mock git/job tags."""
    return Tags(git_sha="abc123", branch="main", job_run_id="job_42")


@pytest.fixture
def mock_spark() -> MagicMock:
    """Mock SparkSession."""
    spark = MagicMock()
    spark.sql.return_value = MagicMock()
    spark.table.return_value = MagicMock()
    spark.table.return_value.drop.return_value = MagicMock()
    return spark


@pytest.fixture
def model(mock_config: ProjectConfig, mock_tags: Tags, mock_spark: MagicMock) -> FeatureLookUpModel:
    """Instantiate the model with mocked dependencies."""
    return FeatureLookUpModel(config=mock_config, tags=mock_tags, spark=mock_spark)


# ---------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------
def test_create_feature_table_executes_sql(model: FeatureLookUpModel):
    """Ensure CREATE/ALTER/INSERT SQL queries are called."""
    model.create_feature_table()
    assert model.spark.sql.call_count >= 4


def test_define_feature_function_creates_udf(model: FeatureLookUpModel):
    """Ensure feature function creation SQL executes."""
    model.define_feature_function()
    model.spark.sql.assert_called_once()
    called_query = model.spark.sql.call_args[0][0]
    assert "CREATE OR REPLACE FUNCTION" in called_query


def test_load_data_calls_spark_table_and_drops_columns(model: FeatureLookUpModel):
    """Test load_data uses Spark table and performs column transformations."""
    mock_train_df = MagicMock()
    mock_test_df = MagicMock()
    model.spark.table.side_effect = [mock_train_df, mock_test_df]
    model.load_data()
    assert model.spark.table.call_count == 2
    assert hasattr(model, "train_set")
    assert hasattr(model, "test_set")


def test_feature_engineering_creates_training_set(model: FeatureLookUpModel):
    """Test that feature_engineering uses FeatureEngineeringClient to create training set."""
    model.train_set = MagicMock()  # ✅ ensures train_set exists
    mock_fe_client = model.fe
    mock_training_set = MagicMock()
    mock_fe_client.create_training_set.return_value = mock_training_set
    mock_training_set.load_df.return_value.toPandas.return_value = pd.DataFrame(
        {
            "lead_time": [10, 20],
            "room_type_reserved": ["A", "B"],
            "booking_status": [0, 1],
            "no_of_weekend_nights": [1, 2],
            "no_of_week_nights": [2, 3],
            "total_nights": [3, 5],  # ✅ add to avoid KeyError
        }
    )
    model.test_set = pd.DataFrame(
        {
            "lead_time": [30, 40],
            "room_type_reserved": ["C", "D"],
            "booking_status": [1, 0],
            "no_of_weekend_nights": [2, 1],
            "no_of_week_nights": [3, 2],
        }
    )

    model.feature_engineering()

    mock_fe_client.create_training_set.assert_called_once()
    assert hasattr(model, "X_train")
    assert "total_nights" in model.X_test.columns


@patch("hotel_reservation.model.feature_lookup_model.mlflow")
def test_train_logs_metrics(mock_mlflow: MagicMock, model: FeatureLookUpModel):
    """Ensure train() fits model, logs metrics and parameters to MLflow."""
    model.X_train = pd.DataFrame({"lead_time": [1, 2], "room_type_reserved": ["A", "B"], "total_nights": [3, 4]})
    model.y_train = pd.Series([0, 1])
    model.X_test = model.X_train.copy()
    model.y_test = pd.Series([0, 1])
    model.training_set = MagicMock()
    model.training_set.load_df.return_value.toPandas.return_value = model.X_train.copy()
    model.test_set = model.X_test.copy()

    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "mock_run"
    mock_mlflow.data.from_pandas.return_value = MagicMock()
    mock_mlflow.sklearn = MagicMock()
    mock_fe_client = model.fe

    model.train()

    mock_mlflow.set_experiment.assert_called_once()
    assert hasattr(model, "metrics")
    assert "accuracy" in model.metrics
    mock_fe_client.log_model.assert_called_once()


@patch("hotel_reservation.model.feature_lookup_model.MlflowClient")
@patch("hotel_reservation.model.feature_lookup_model.mlflow")
def test_register_model_sets_alias(mock_mlflow: MagicMock, mock_client: MagicMock, model: FeatureLookUpModel):
    """Ensure register_model registers and sets alias."""
    model.run_id = "mock_run_id"
    mock_registered = MagicMock()
    mock_registered.version = 1
    mock_mlflow.register_model.return_value = mock_registered

    # ✅ mock MlflowClient to prevent real URI parsing
    client_instance = MagicMock()
    mock_client.return_value = client_instance

    version = model.register_model()

    assert version == 1
    mock_mlflow.register_model.assert_called_once()
    client_instance.set_registered_model_alias.assert_called_once()


def test_load_latest_model_and_predict_returns_predictions(model: FeatureLookUpModel):
    """Ensure load_latest_model_and_predict calls score_batch."""
    model.fe.score_batch.return_value = pd.DataFrame({"prediction": [1, 0]})
    dummy_df = pd.DataFrame({"a": [1, 2]})
    result = model.load_latest_model_and_predict(dummy_df)
    model.fe.score_batch.assert_called_once()
    assert "prediction" in result.columns


def test_update_feature_table_runs_two_sql_queries(model: FeatureLookUpModel):
    """Ensure update_feature_table executes two SQL queries."""
    model.update_feature_table()
    assert model.spark.sql.call_count == 2


@patch("hotel_reservation.model.feature_lookup_model.mlflow")
def test_model_improved_handles_no_existing_model(mock_mlflow: MagicMock, model: FeatureLookUpModel):
    """If no existing model, F1 = 0 and should return True."""
    model.metrics = {"f1_score": 0.9}
    client_mock = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client_mock
    # ✅ ensure exception is a real subclass of Exception
    mock_mlflow.exceptions.MlflowException = Exception

    client_mock.get_model_version_by_alias.side_effect = Exception("No model found")

    improved = model.model_improved(test_set=pd.DataFrame())

    assert improved is True
