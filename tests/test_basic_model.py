import warnings
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.exceptions import ConvergenceWarning
import scipy

# Disable irrelevant warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")

from src.mlops_course.model.basic_model import BasicModel


@pytest.fixture
def mock_config():
    """Mock the project configuration object."""
    class MockConfig:
        num_features = ["age", "income"]
        cat_features = ["country"]
        target = "label"
        parameters = {"max_iter": 100}
        catalog_name = "catalog"
        schema_name = "schema"
        train_table = "train"
        test_table = "test"
        experiment_name_basic = "exp_basic"
        model_name = "hotel_model"
        model_type = "logreg"

    return MockConfig()


@pytest.fixture
def mock_tags():
    """Mock the MLflow tags object."""
    class MockTags:
        def model_dump(self):
            return {"project": "mlops_course", "stage": "dev"}

    return MockTags()


@pytest.fixture
def mock_spark():
    """Mock a SparkSession returning fake DataFrames."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = pd.DataFrame({
        "age": [25, 30, 45],
        "income": [40000, 50000, 60000],
        "country": ["FR", "US", "DE"],
        "label": [0, 1, 0],
    })

    mock_spark = MagicMock()
    mock_spark.table.return_value = mock_df
    return mock_spark


@pytest.fixture
def model(mock_config, mock_tags, mock_spark):
    """Instance of BasicModel initialized with mocks."""
    return BasicModel(config=mock_config, tags=mock_tags, spark=mock_spark)


def test_load_data(model):
    """Test the load_data() method."""
    model.load_data()

    assert hasattr(model, "X_train")
    assert hasattr(model, "y_train")
    assert isinstance(model.X_train, pd.DataFrame)
    assert "age" in model.X_train.columns
    assert model.y_train.name == "label"
    model.spark.table.assert_called()  # Verify Spark table call


def test_prepare_features(model):
    """Test the pipeline preparation method."""
    model.cat_features = ["country"]
    model.num_features = ["age", "income"]

    model.prepare_features()

    assert model.pipeline is not None
    steps = [s[0] for s in model.pipeline.steps]
    assert "preprocessor" in steps
    assert "classifier" in steps


def test_train(model):
    """Test the model training process."""
    model.X_train = pd.DataFrame({
        "age": [25, 30],
        "income": [40000, 50000],
        "country": ["FR", "US"],
    })
    model.y_train = pd.Series([0, 1])

    model.prepare_features()
    model.train()

    # Verify that the model was successfully trained
    assert hasattr(model.pipeline, "predict")


@patch("src.mlops_course.model.basic_model.mlflow")
def test_log_model(mock_mlflow, model):
    """Test the log_model() method using MLflow mocks."""
    model.X_train = pd.DataFrame({
        "age": [25.0, 30.0],
        "income": [40000.0, 50000.0],
        "country": ["FR", "US"],
    })
    model.X_test = pd.DataFrame({
        "age": [28, 35],
        "income": [42000, 52000],
        "country": ["FR", "US"],
    })
    model.y_test = pd.Series([0, 1])
    model.pipeline = MagicMock()
    model.pipeline.predict.return_value = np.array([0, 1])

    # Simulate an MLflow run
    mock_run = MagicMock()
    mock_run.info.run_id = "123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    model.train_set_spark = MagicMock()
    model.data_version = "0"  # ✅ Added manually since load_data() is not called

    model.log_model()

    mock_mlflow.log_metric.assert_any_call("accuracy", 1.0)
    mock_mlflow.sklearn.log_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
@patch("src.mlops_course.model.basic_model.MlflowClient")
def test_register_model(mock_client_cls, mock_mlflow, model):
    """Test the model registration process in MLflow."""
    model.run_id = "123"
    model.model_name = "catalog.schema.hotel_model"
    mock_mlflow.register_model.return_value.version = 2

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    model.register_model()

    mock_mlflow.register_model.assert_called_once()
    mock_client.set_registered_model_alias.assert_called_with(
        name=model.model_name, alias="latest-model", version=2
    )


@patch("src.mlops_course.model.basic_model.mlflow")
def test_load_latest_model_and_predict(mock_mlflow, model):
    """Test model loading and prediction from MLflow."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1, 0])
    mock_mlflow.sklearn.load_model.return_value = fake_model

    df_input = pd.DataFrame({
        "age": [22, 33],
        "income": [35000, 65000],
        "country": ["FR", "US"],
    })

    preds = model.load_latest_model_and_predict(df_input)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    mock_mlflow.sklearn.load_model.assert_called_once()


@patch("src.mlops_course.model.basic_model.mlflow")
def test_retrieve_current_run_dataset(mock_mlflow, model):
    """Test retrieve_current_run_dataset()."""
    mock_dataset_source = MagicMock()
    mock_dataset_source.load.return_value = "mocked_dataset"
    mock_mlflow.get_run.return_value.inputs.dataset_inputs = [
        MagicMock(dataset="mocked_dataset_info")
    ]
    mock_mlflow.data.get_source.return_value = mock_dataset_source

    model.run_id = "123"
    result = model.retrieve_current_run_dataset()

    mock_mlflow.get_run.assert_called_with("123")
    mock_mlflow.data.get_source.assert_called_once_with("mocked_dataset_info")
    assert result == "mocked_dataset"


@patch("src.mlops_course.model.basic_model.mlflow")
def test_retrieve_current_run_metadata(mock_mlflow, model):
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
