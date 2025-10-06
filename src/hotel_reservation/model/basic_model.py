"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.timer import timeit


class Result:
    """Container for storing model evaluation metrics."""

    def __init__(self) -> None:
        """Initialize metrics dictionary."""
        self.metrics = {}


class BasicModel:
    """A basic model class for hotel_reservation prediction using LogisticRegression.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.train_table = self.config.train_table
        self.test_table = self.config.test_table
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.{self.config.model_name}"
        self.model_type = self.config.model_type
        self.tags = tags.model_dump()

    @timeit
    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.train_table}")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.test_table}")
        self.test_set = self.test_set_spark.toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.num_features + self.cat_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.{self.train_table}")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.{self.test_table}")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])

        logger.info("✅ Data successfully loaded.")

    @timeit
    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LogisticRegression Classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LogisticRegression(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    @timeit
    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    @timeit
    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.{self.train_table}",
                version=self.data_version,
            )
            mlflow.log_input(train_dataset, context="training")

            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.{self.test_table}",
                version=self.data_version,
            )
            mlflow.log_input(test_dataset, context="testing")

            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path=f"{self.model_type}-pipeline-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            # Evaluate classification metrics
            result = Result()
            result.metrics["accuracy"] = accuracy_score(self.y_test, y_pred)
            result.metrics["precision"] = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            result.metrics["recall"] = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            result.metrics["f1_score"] = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

            logger.info(f"📊 Accuracy: {result.metrics['accuracy']}")
            logger.info(f"📊 Precision: {result.metrics['precision']}")
            logger.info(f"📊 Recall: {result.metrics['recall']}")
            logger.info(f"📊 F1 Score: {result.metrics['f1_score']}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "Logistic Regression with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", result.metrics["accuracy"])
            mlflow.log_metric("precision", result.metrics["precision"])
            mlflow.log_metric("recall", result.metrics["recall"])
            mlflow.log_metric("f1_score", result.metrics["f1_score"])

            self.metrics = result.metrics

    @timeit
    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/{self.model_type}-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        
        return latest_version

    @timeit
    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("Dataset source loaded.")
        return dataset_source.load()

    @timeit
    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("Dataset metadata loaded.")
        return metrics, params

    @timeit
    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions

    @timeit
    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        try:
            latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
            latest_model_uri = f"models:/{latest_model_version.model_id}"

            result = mlflow.models.evaluate(
                latest_model_uri,
                self.eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            metrics_old = result.metrics
            logger.info(f"Latest model F1-score: {metrics_old['f1_score']}")
        except Exception:
            logger.info("No model exist yet. Set F1-score to Zero")
            metrics_old = {}
            metrics_old["f1_score"] = 0

        logger.info(f"Current model F1-score: {self.metrics['f1_score']}")
        if self.metrics["f1_score"] >= metrics_old["f1_score"]:
            logger.info("💥 Current model performs better. Returning True.")
            return True
        else:
            logger.info("⛔ Current model does not improve over latest. Returning False.")
            return False
