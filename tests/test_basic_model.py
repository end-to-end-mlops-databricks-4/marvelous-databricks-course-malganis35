"""Unit tests for BasicModel with full lint compliance.

- Mocks `delta.tables.DeltaTable` so tests run without delta-spark.
- Covers training, logging, registration, prediction, dataset/metadata retrieval,
  and both branches of `model_improved()`.
"""

# ---------------------------------------------------------------------
# Patch delta.tables.DeltaTable before importing BasicModel
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

# Create a synthetic `delta` package with `tables.DeltaTable`
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
                """Return a mock history object."""

                class MockHistory:
                    """Mock for DeltaTable.history() result."""

                    @staticmethod
                    def select(column: str) -> type:
                        """Return a mock select object."""

                        class MockSelect:
                            """Mock for DataFrame-like select result."""

                            @staticmethod
                            def first() -> list[int]:
                                """Return list with a single integer version to support [0] indexing."""
                                return [0]

                        return MockSelect()

                return MockHistory()

        return MockDelta()


# Wire up the mocked modules
mock_tables_module.DeltaTable = MockDeltaTable
mock_delta_module.tables = mock_tables_module
sys.modules["delta"] = mock_delta_module
sys.modules["delta.tables"] = mock_tables_module

# ---------------------------------------------------------------------
# Imports after DeltaTable patch
# ---------------------------------------------------------------------
from src.mlops_course.model.basic_model import BasicModel  # noqa: E402

# ---------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")


# ---------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------
class MockConfig:
    """Mock configuration for BasicModel tests."""

    num_features: list[str] = ["age", "income"]
    cat_features: list[str] = ["country"]
    target: str = "label"
    parameters: dict[str, int] = {"max_iter": 100}
    catalog_name: str = "catalog"
    schema_name: str = "schema"
    train_table: str = "train"
    test_table: str = "test"
    experiment_name_basic: str = "exp_basic"
    model_name: str = "hotel_model"
    model_type: str = "logreg"


class MockTags:
    """Mock MLflow tags object."""

    def model_dump(self) -> dict[str, str]:
        """Return a mock tags dictionary."""
        return {"project": "mlops_course", "stage": "dev"}


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def mock_config() -> MockConfig:
    """Return mock configuration instance."""
    return MockConfig()


@pytest.fixture
def mock_tags() -> MockTags:
    """Return mock tags instance."""
    return MockTags()


@pytest.fixture
def mock_spark() -> MagicMock:
    """Return mock SparkSession returning a small pandas DataFrame via toPandas()."""
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
def model(mock_config: MockConfig, mock_tags: MockTags, mock_spark: MagicMock) -> BasicModel:
    """Build a BasicModel instance injected with mocks."""
    return BasicModel(config=mock_config, tags=mock_tags, spark=mock_spark)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_load_data(model: BasicModel) -> None:
    """Ensure load_data() pulls data and creates train/test splits."""
    model.load_data()

    assert hasattr(model, "X_train")
    assert hasattr(model, "y_train")
    assert isinstance(model.X_train, pd.DataFrame)
    assert "age" in model.X_train.columns
    assert model.y_train.name == "label"
    model.spark.table.assert_called()


def test_prepare_features(model: BasicModel) -> None:
    """Ensure feature preparation builds preprocessing pipeline."""
    model.prepare_features()
    steps = [s[0] for s in model.pipeline.steps]
    assert "preprocessor" in steps
    assert "classifier" in steps


def test_train(model: BasicModel) -> None:
    """Ensure training runs and pipeline becomes predictive."""
    model.X_train = pd.DataFrame({"age": [25, 30], "income": [40000, 50000], "country": ["FR", "US"]})
    model.y_train = pd.Series([0, 1])
    model.prepare_features()
    model.train()
    assert hasattr(model.pipeline, "predict")


@patch("src.mlops_course.model.basic_model.mlflow")
def test_log_model(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Verify metrics and model are logged to MLflow."""
    model.X_train = pd.DataFrame({"age": [25.0, 30.0], "income": [40000.0, 50000.0], "country": ["FR", "US"]})
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

    mock_mlflow.log_metric.assert_any_call("accuracy", 1.0)
    mock_mlflow.sklearn.log_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
@patch("src.mlops_course.model.basic_model.MlflowClient")
def test_register_model(mock_client_cls: MagicMock, mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Ensure model is registered and alias is set to latest-model."""
    model.run_id = "123"
    model.model_name = "catalog.schema.hotel_model"
    mock_mlflow.register_model.return_value.version = 2

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    model.register_model()
    mock_mlflow.register_model.assert_called_once()
    mock_client.set_registered_model_alias.assert_called_with(name=model.model_name, alias="latest-model", version=2)


@patch("src.mlops_course.model.basic_model.mlflow")
def test_load_latest_model_and_predict(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Ensure prediction path loads the model and returns numpy array."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1, 0])
    mock_mlflow.sklearn.load_model.return_value = fake_model

    df_input = pd.DataFrame({"age": [22, 33], "income": [35000, 65000], "country": ["FR", "US"]})

    preds = model.load_latest_model_and_predict(df_input)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    mock_mlflow.sklearn.load_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
def test_retrieve_current_run_dataset(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Ensure dataset source loader is invoked and result returned."""
    mock_dataset_source = MagicMock()
    mock_dataset_source.load.return_value = "mocked_dataset"
    mock_mlflow.get_run.return_value.inputs.dataset_inputs = [MagicMock(dataset="mocked_dataset_info")]
    mock_mlflow.data.get_source.return_value = mock_dataset_source

    model.run_id = "123"
    result = model.retrieve_current_run_dataset()

    mock_mlflow.get_run.assert_called_with("123")
    mock_mlflow.data.get_source.assert_called_once_with("mocked_dataset_info")
    assert result == "mocked_dataset"


@patch("src.mlops_course.model.basic_model.mlflow")
def test_retrieve_current_run_metadata(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Ensure metrics and params dicts are extracted from run."""
    mock_run = MagicMock()
    mock_run.data.to_dictionary.return_value = {
        "metrics": {"accuracy": 0.9, "f1_score": 0.88},
        "params": {"max_iter": "100"},
    }
    mock_mlflow.get_run.return_value = mock_run

    model.run_id = "456"
    metrics, params = model.retrieve_current_run_metadata()

    mock_mlflow.get_run.assert_called_with("456")
    assert metrics["accuracy"] == 0.9
    assert params["max_iter"] == "100"


@patch("src.mlops_course.model.basic_model.mlflow")
@patch("src.mlops_course.model.basic_model.MlflowClient")
def test_model_improved(mock_client_cls: MagicMock, mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Return True when current model F1 is >= previous model F1."""
    model.metrics = {"f1_score": 0.8}
    model.eval_data = pd.DataFrame({"age": [25], "income": [40000], "country": ["FR"], "label": [1]})
    model.model_name = "catalog.schema.hotel_model"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value.model_id = "fake_model_id"
    mock_client_cls.return_value = mock_client

    mock_result = MagicMock()
    mock_result.metrics = {"f1_score": 0.7}
    mock_mlflow.models.evaluate.return_value = mock_result

    improved = model.model_improved()

    assert improved is True
    mock_mlflow.models.evaluate.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
@patch("src.mlops_course.model.basic_model.MlflowClient")
def test_model_not_improved(mock_client_cls: MagicMock, mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Return False when current model F1 is < previous model F1."""
    model.metrics = {"f1_score": 0.6}
    model.eval_data = pd.DataFrame({"age": [25], "income": [40000], "country": ["FR"], "label": [1]})
    model.model_name = "catalog.schema.hotel_model"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value.model_id = "fake_model_id"
    mock_client_cls.return_value = mock_client

    mock_result = MagicMock()
    mock_result.metrics = {"f1_score": 0.9}  # previous better than current
    mock_mlflow.models.evaluate.return_value = mock_result

    improved = model.model_improved()

    assert improved is False
    mock_mlflow.models.evaluate.assert_called_once()
