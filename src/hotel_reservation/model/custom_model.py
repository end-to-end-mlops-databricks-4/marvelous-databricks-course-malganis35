"""Custom model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.timer import timeit

import yaml
from pathlib import Path

class Result:
    """Container for storing model evaluation metrics."""

    def __init__(self) -> None:
        """Initialize metrics dictionary."""
        self.metrics = {}


class SklearnModelWithProba(mlflow.pyfunc.PythonModel):
    """Wrapper MLflow model that outputs both prediction and probability."""

    def __init__(self, sklearn_model: BaseEstimator) -> None:
        """Initialize the wrapper with a scikit-learn model."""
        self.model = sklearn_model

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> list[dict[str, float | str | int]]:
        """Return both prediction label and probability."""
        proba = self.model.predict_proba(model_input)[:, 1]
        preds = (proba >= 0.5).astype(int)
        results: list[dict[str, float | str | int]] = []
        for p, pr in zip(preds, proba, strict=False):
            results.append(
                {
                    "prediction": int(p),
                    "label": "Cancelled" if p == 0 else "Not_Canceled",
                    "probability": float(pr),
                }
            )
        return results


class CustomModel:
    """A basic model class for hotel_reservation prediction using LogisticRegression.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
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
        self.experiment_name = self.config.experiment_name_custom
        self.model_name = f"{self.catalog_name}.{self.schema_name}.{self.config.model_name_custom}"
        self.model_type = self.config.model_type
        self.tags = tags.model_dump()
        self.code_paths = code_paths

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
        
        ####################################################
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")
        ####################################################
        
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

            wrapped_model = SklearnModelWithProba(self.pipeline)

            #############################################################
            # ✅ Define custom conda environment for MLflow logging
            # conda_env = {
            #     "channels": ["conda-forge"],
            #     "dependencies": [
            #         "python=3.11.9",
            #         "pip<=25.2",
            #         {
            #             "pip": [
            #                 "mlflow==3.1.1",
            #                 "cloudpickle==3.1.1",
            #                 "databricks-connect==16.3.5",
            #                 "numpy==1.26.4",
            #                 "pandas==2.3.0",
            #                 "psutil==7.1.0",
            #                 "scikit-learn==1.7.0",
            #                 "scipy==1.16.0",
            #                 "-e ."
            #             ]
            #         },
            #     ],
            #     "name": "mlflow-env",
            # }
            
            # # Write it to file so MLflow can version it
            # env_path = Path("conda_env_custom.yaml")
            # with open(env_path, "w") as f:
            #     yaml.safe_dump(conda_env, f)
            # logger.info(f"✅ Custom conda environment written to {env_path}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)
            
            #############################################################
            
            mlflow.pyfunc.log_model(
                artifact_path=f"{self.model_type}-pipeline-custom-model",
                python_model=wrapped_model,
                signature=signature,
                input_example=self.X_test[0:1],
                ##################################
                # conda_env=str(env_path),
                # code_paths=["src"],
                conda_env=conda_env,
                code_paths=self.code_paths,
                ##################################
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
            model_uri=f"runs:/{self.run_id}/{self.model_type}-pipeline-custom-model",
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
        """Load the latest model from MLflow (alias=latest-model) and make predictions."""
        logger.info("Loading model from MLflow alias 'latest-model'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)  # ✅ Correction ici

        logger.info("Model successfully loaded.")

        # Convert Spark → Pandas if needed
        if hasattr(input_data, "toPandas"):
            logger.info("Converting Spark DataFrame to Pandas for prediction...")
            input_data = input_data.toPandas()

        # Make predictions
        results = model.predict(input_data)

        # Convertir la liste de dicts en DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def _get_baseline_model_uri(self) -> str | None:
        """Return the URI of the baseline model used for comparison.

        The method attempts to find the most relevant reference model:
        - Uses the alias @latest-model if it exists,
        - Otherwise falls back to the latest registered version,
        - Returns None if no registered model exists yet.
        """
        client = MlflowClient(registry_uri=mlflow.get_registry_uri())

        try:
            alias_info = client.get_model_version_by_alias(
                name=self.model_name,
                alias="latest-model",
            )
            v = alias_info.version
            logger.info(f"Baseline: using alias @latest-model (version {v})")
            return f"models:/{self.model_name}/{v}"  # 👈 important: on fixe la version exacte
        except Exception as e:
            logger.info(f"No alias 'latest-model' (fallback to latest version). Reason: {e}")

        try:
            latest_versions = client.get_latest_versions(self.model_name)
            if latest_versions:
                v = latest_versions[0].version
                logger.info(f"Baseline: using last registered version: {v}")
                return f"models:/{self.model_name}/{v}"
        except RestException as e:
            logger.info(f"No registered versions found. Reason: {e}")

        logger.info("No registered model found. Treat as first run.")
        return None

    @timeit
    def model_improved(self) -> bool:
        """Compare the current model (metrics already computed in log_model).

        Compares the model against the best registered model (alias @latest-model if available,
        otherwise the latest registered version). Returns True if the new model's
        F1-score is greater than or equal to the baseline.
        """
        logger.info(f"Active registry URI before fetching alias: {mlflow.get_registry_uri()}")

        baseline_uri = self._get_baseline_model_uri()
        new_f1 = float(self.metrics["f1_score"])

        if baseline_uri is None:
            # Premier run : pas de modèle de référence
            logger.info(f"No baseline model. Current model F1-score: {new_f1} → will register.")
            return True

        # Évaluer manuellement (sans mlflow.evaluate) car le modèle retourne un dict
        baseline_model = mlflow.pyfunc.load_model(baseline_uri)
        # Faire les prédictions du modèle baseline
        preds_df = pd.DataFrame(baseline_model.predict(self.X_test))
        if "prediction" not in preds_df.columns:
            raise ValueError("Expected key 'prediction' in baseline model output.")

        y_pred = preds_df["prediction"]
        y_true = self.y_test.copy()

        # Harmoniser les labels de y_true (string → int)
        y_true = y_true.replace(
            {
                "Canceled": 0,
                "Not_Canceled": 1,
            }
        )
        y_true = y_true.astype(int)

        old_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        logger.info(f"Baseline F1-score: {old_f1}")
        logger.info(f"Current  F1-score: {new_f1}")

        improved = new_f1 >= old_f1
        if improved:
            logger.info("💥 Current model performs better or equal. Returning True.")
        else:
            logger.info("⛔ Current model is worse. Returning False.")
        return improved
