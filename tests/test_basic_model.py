from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

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
    """Typed mock class representing a fake project configuration."""

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
    """Typed mock class representing MLflow tags."""

    def model_dump(self) -> dict[str, str]:
        """Return fake tags as a dictionary."""
        return {"project": "mlops_course", "stage": "dev"}


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def mock_config() -> MockConfig:
    """Return a mocked configuration object."""
    return MockConfig()


@pytest.fixture
def mock_tags() -> MockTags:
    """Return a mocked MLflow tags object."""
    return MockTags()


@pytest.fixture
def mock_spark() -> MagicMock:
    """Return a mocked SparkSession producing fake DataFrames."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = pd.DataFrame(
        {
            "age": [25, 30, 45],
            "income": [40000, 50000, 60000],
            "country": ["FR", "US", "DE"],
            "label": [0, 1, 0],
        }
    )

    mock_spark = MagicMock()
    mock_spark.table.return_value = mock_df
    return mock_spark


@pytest.fixture
def model(mock_config: MockConfig, mock_tags: MockTags, mock_spark: MagicMock) -> BasicModel:
    """Create a BasicModel instance initialized with mocks."""
    return BasicModel(config=mock_config, tags=mock_tags, spark=mock_spark)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_load_data(model: BasicModel) -> None:
    """Test the load_data() method."""
    model.load_data()

    assert hasattr(model, "X_train")
    assert hasattr(model, "y_train")
    assert isinstance(model.X_train, pd.DataFrame)
    assert "age" in model.X_train.columns
    assert model.y_train.name == "label"
    model.spark.table.assert_called()  # Verify Spark table call


def test_prepare_features(model: BasicModel) -> None:
    """Test the feature preparation and pipeline creation."""
    model.cat_features = ["country"]
    model.num_features = ["age", "income"]

    model.prepare_features()

    assert model.pipeline is not None
    steps = [s[0] for s in model.pipeline.steps]
    assert "preprocessor" in steps
    assert "classifier" in steps


def test_train(model: BasicModel) -> None:
    """Test model training logic."""
    model.X_train = pd.DataFrame(
        {
            "age": [25, 30],
            "income": [40000, 50000],
            "country": ["FR", "US"],
        }
    )
    model.y_train = pd.Series([0, 1])

    model.prepare_features()
    model.train()

    assert hasattr(model.pipeline, "predict")  # Ensure training worked


@patch("src.mlops_course.model.basic_model.mlflow")
def test_log_model(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Test log_model() method using MLflow mocks."""
    model.X_train = pd.DataFrame(
        {
            "age": [25.0, 30.0],
            "income": [40000.0, 50000.0],
            "country": ["FR", "US"],
        }
    )
    model.X_test = pd.DataFrame(
        {
            "age": [28, 35],
            "income": [42000, 52000],
            "country": ["FR", "US"],
        }
    )
    model.y_test = pd.Series([0, 1])
    model.pipeline = MagicMock()
    model.pipeline.predict.return_value = np.array([0, 1])

    # Simulate an MLflow run
    mock_run = MagicMock()
    mock_run.info.run_id = "123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    model.train_set_spark = MagicMock()
    model.data_version = "0"  # Added manually since load_data() is not called

    model.log_model()

    mock_mlflow.log_metric.assert_any_call("accuracy", 1.0)
    mock_mlflow.sklearn.log_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
@patch("src.mlops_course.model.basic_model.MlflowClient")
def test_register_model(mock_client_cls: MagicMock, mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Test model registration in MLflow."""
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
    """Test loading the latest model and making predictions."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1, 0])
    mock_mlflow.sklearn.load_model.return_value = fake_model

    df_input = pd.DataFrame(
        {
            "age": [22, 33],
            "income": [35000, 65000],
            "country": ["FR", "US"],
        }
    )

    preds = model.load_latest_model_and_predict(df_input)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    mock_mlflow.sklearn.load_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
def test_retrieve_current_run_dataset(mock_mlflow: MagicMock, model: BasicModel) -> None:
    """Test retrieve_current_run_dataset()."""
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
    """Test retrieve_current_run_metadata()."""
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
