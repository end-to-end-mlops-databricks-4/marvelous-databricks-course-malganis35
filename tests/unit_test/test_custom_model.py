"""Unit tests for CustomModel with full lint compliance.

- Mocks pyspark and delta.tables.DeltaTable so tests run without Spark.
- Covers training, logging, registration, prediction, dataset/metadata retrieval,
  and both branches of model_improved().
"""

from __future__ import annotations

import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

# ---------------------------------------------------------------------
# Patch pyspark and delta modules before importing CustomModel
# ---------------------------------------------------------------------
mock_pyspark = types.ModuleType("pyspark")
mock_sql = types.ModuleType("pyspark.sql")
mock_sql.SparkSession = MagicMock()
mock_pyspark.sql = mock_sql
sys.modules["pyspark"] = mock_pyspark
sys.modules["pyspark.sql"] = mock_sql

# === Mock delta.tables.DeltaTable ===
mock_delta_module = types.ModuleType("delta")
mock_tables_module = types.ModuleType("delta.tables")


class MockDeltaTable:
    """Mock for delta.tables.DeltaTable supporting .forName().history().select().first() chain."""

    @staticmethod
    def forName(spark: MagicMock, table_name: str) -> type:
        """Return a mock DeltaTable instance compatible with the API chain used in code."""

        class MockDelta:
            """Mock object returned by DeltaTable.forName()."""

            @staticmethod
            def history() -> type:
                class MockHistory:
                    @staticmethod
                    def select(column: str) -> type:
                        class MockSelect:
                            @staticmethod
                            def first() -> list[int]:
                                return [0]

                        return MockSelect()

                return MockHistory()

        return MockDelta()


mock_tables_module.DeltaTable = MockDeltaTable
mock_delta_module.tables = mock_tables_module
sys.modules["delta"] = mock_delta_module
sys.modules["delta.tables"] = mock_tables_module

# ---------------------------------------------------------------------
# Import after patching
# ---------------------------------------------------------------------
from src.hotel_reservation.model.custom_model import CustomModel  # noqa: E402

# ---------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")


# ---------------------------------------------------------------------
# Mock Config / Tags
# ---------------------------------------------------------------------
class MockConfig:
    """Define a Class for Mock configuration."""

    num_features: list[str] = ["age", "income"]
    cat_features: list[str] = ["country"]
    target: str = "label"
    parameters: dict[str, int] = {"max_iter": 100}
    catalog_name: str = "catalog"
    schema_name: str = "schema"
    train_table: str = "train"
    test_table: str = "test"
    experiment_name_custom: str = "exp_custom"
    model_name_custom: str = "hotel_model_custom"
    model_type: str = "logreg"


class MockTags:
    """Define a Class for Mock Tags configuration."""

    def model_dump(self) -> dict[str, str]:
        """Return mock tags as a dictionary for MLflow logging."""
        return {"project": "hotel_reservation", "stage": "dev"}


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def mock_config() -> MockConfig:
    """Return a mock configuration object."""
    return MockConfig()


@pytest.fixture
def mock_tags() -> MockTags:
    """Return mock tags configuration."""
    return MockTags()


@pytest.fixture
def mock_spark() -> MagicMock:
    """Return a mocked SparkSession with sample hotel reservation data."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = pd.DataFrame(
        {
            "age": [25, 30, 45],
            "income": [40000, 50000, 60000],
            "country": ["FR", "US", "DE"],
            "label": [0, 1, 0],
        }
    )
    spark = MagicMock()
    spark.table.return_value = mock_df
    return spark


@pytest.fixture
def model(mock_config: MockConfig, mock_tags: MockTags, mock_spark: MagicMock) -> CustomModel:
    """Return an instance of CustomModel with mocked dependencies."""
    return CustomModel(
        config=mock_config,
        tags=mock_tags,
        spark=mock_spark,
        code_paths=["src/hotel_reservation/model/custom_model.py"],  # Added mock code_paths
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_load_data(model: CustomModel) -> None:
    """Ensure load_data() pulls data and creates train/test splits."""
    model.load_data()
    assert isinstance(model.X_train, pd.DataFrame)
    assert model.y_train.name == "label"
    model.spark.table.assert_called()


def test_prepare_features(model: CustomModel) -> None:
    """Ensure feature preparation builds preprocessing pipeline."""
    model.prepare_features()
    steps = [name for name, _ in model.pipeline.steps]
    assert "preprocessor" in steps
    assert "classifier" in steps


def test_train(model: CustomModel) -> None:
    """Ensure training runs successfully."""
    model.X_train = pd.DataFrame({"age": [25, 30], "income": [40000, 50000], "country": ["FR", "US"]})
    model.y_train = pd.Series([0, 1])
    model.prepare_features()
    model.train()
    assert hasattr(model.pipeline, "predict")


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_log_model(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Verify metrics and model are logged with wrapped pyfunc."""
    model.X_train = pd.DataFrame({"age": [25, 30], "income": [40000, 50000], "country": ["FR", "US"]})
    model.X_test = pd.DataFrame({"age": [28, 35], "income": [42000, 52000], "country": ["FR", "US"]})
    model.y_test = pd.Series([0, 1])
    model.pipeline = MagicMock()
    model.pipeline.predict.return_value = np.array([0, 1])

    mock_run = MagicMock()
    mock_run.info.run_id = "123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    model.train_set_spark = MagicMock()
    model.test_set_spark = MagicMock()
    model.data_version = "0"

    model.log_model()

    mock_mlflow.pyfunc.log_model.assert_called_once()
    mock_mlflow.log_metric.assert_any_call("f1_score", 1.0)
    assert isinstance(model.metrics, dict)


@patch("src.hotel_reservation.model.custom_model.mlflow")
@patch("src.hotel_reservation.model.custom_model.MlflowClient")
def test_register_model(mock_client_cls: MagicMock, mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Ensure model is registered and alias 'latest-model' set."""
    model.run_id = "123"
    model.model_name = "catalog.schema.hotel_model_custom"
    mock_mlflow.register_model.return_value.version = 3
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    model.register_model()

    mock_mlflow.register_model.assert_called_once()
    mock_client.set_registered_model_alias.assert_called_with(name=model.model_name, alias="latest-model", version=3)


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_load_latest_model_and_predict(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Ensure latest model is loaded and returns DataFrame."""
    fake_model = MagicMock()
    fake_model.predict.return_value = [
        {"prediction": 1, "label": "Not_Canceled", "probability": 0.9},
        {"prediction": 0, "label": "Cancelled", "probability": 0.1},
    ]
    mock_mlflow.pyfunc.load_model.return_value = fake_model

    df_input = pd.DataFrame({"age": [22, 33], "income": [35000, 65000], "country": ["FR", "US"]})
    preds = model.load_latest_model_and_predict(df_input)

    assert isinstance(preds, pd.DataFrame)
    assert "prediction" in preds.columns
    mock_mlflow.pyfunc.load_model.assert_called_once()


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_retrieve_current_run_dataset(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Ensure dataset source loader is invoked and returns dataset."""
    mock_dataset_source = MagicMock()
    mock_dataset_source.load.return_value = "mocked_dataset"
    mock_mlflow.get_run.return_value.inputs.dataset_inputs = [MagicMock(dataset="mocked_dataset_info")]
    mock_mlflow.data.get_source.return_value = mock_dataset_source

    model.run_id = "123"
    result = model.retrieve_current_run_dataset()

    mock_mlflow.get_run.assert_called_with("123")
    mock_mlflow.data.get_source.assert_called_once_with("mocked_dataset_info")
    assert result == "mocked_dataset"


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_retrieve_current_run_metadata(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Ensure metrics and params dicts are extracted."""
    mock_run = MagicMock()
    mock_run.data.to_dictionary.return_value = {
        "metrics": {"accuracy": 0.9, "f1_score": 0.88},
        "params": {"max_iter": "100"},
    }
    mock_mlflow.get_run.return_value = mock_run
    model.run_id = "456"

    metrics, params = model.retrieve_current_run_metadata()

    assert metrics["accuracy"] == 0.9
    assert params["max_iter"] == "100"


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_model_improved_true(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Return True when new model performs better."""
    model.metrics = {"f1_score": 0.8}  # current model is better
    model.y_test = pd.Series([0, 1])
    model.X_test = pd.DataFrame({"age": [1, 2], "income": [1, 2], "country": ["FR", "US"]})

    # Fake baseline model gives worse F1 (predicts wrong class)
    fake_model = MagicMock()
    fake_model.predict.return_value = [{"prediction": 1}, {"prediction": 1}]  # bad baseline
    mock_mlflow.pyfunc.load_model.return_value = fake_model

    with patch.object(model, "_get_baseline_model_uri", return_value="models:/test_model/1"):
        improved = model.model_improved()
        assert improved is True


@patch("src.hotel_reservation.model.custom_model.mlflow")
def test_model_improved_false(mock_mlflow: MagicMock, model: CustomModel) -> None:
    """Return False when current model is worse."""
    model.metrics = {"f1_score": 0.5}  # current model worse
    model.y_test = pd.Series([0, 1])
    model.X_test = pd.DataFrame({"age": [1, 2], "income": [1, 2], "country": ["FR", "US"]})

    # Fake baseline model predicts perfectly (better F1)
    fake_model = MagicMock()
    fake_model.predict.return_value = [{"prediction": 0}, {"prediction": 1}]  # perfect baseline
    mock_mlflow.pyfunc.load_model.return_value = fake_model

    with patch.object(model, "_get_baseline_model_uri", return_value="models:/test_model/1"):
        improved = model.model_improved()
        assert improved is False
