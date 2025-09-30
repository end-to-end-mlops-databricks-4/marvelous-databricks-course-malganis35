"""FeatureLookUp model implementation."""

from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
# from lightgbm import LGBMRegressor
from delta.tables import DeltaTable
from sklearn.linear_model import LogisticRegression
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.feature_table_name = self.config.feature_table_name
        self.feature_function_name = self.config.feature_function_name
        self.train_table = self.config.train_table
        self.test_table = self.config.test_table
        
        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.{self.feature_table_name}"
        self.feature_function_name = f"{self.catalog_name}.{self.schema_name}.{self.feature_function_name}"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    @timeit
    def create_feature_table(self) -> None:
        """Create or update the hotel_reservations_features table and populate it.

        This table stores features related to hotel_reservations.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog_name}.{self.schema_name}.{self.train_table}"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog_name}.{self.schema_name}.{self.test_table}"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the number of persons.

        This function adds no_of_weekend_nights + no_of_week_nights = total_nights
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(total_nights INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        total_nights = no_of_weekend_nights + no_of_week_nights
        return total_nights
        $$
        """)
        logger.info("✅ Feature function defined.")

    @timeit
    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'no_of_weekend_nights' and 'no_of_week_nights' to integer type.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.train_table}").drop(
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.test_table}").toPandas()

        self.train_set = self.train_set.withColumn("no_of_weekend_nights", self.train_set["no_of_weekend_nights"].cast("int"))
        self.train_set = self.train_set.withColumn("no_of_week_nights", self.train_set["no_of_week_nights"].cast("int"))
        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    @timeit
    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="total_nights",
                    input_bindings={"no_of_weekend_nights": "no_of_weekend_nights", 
                                    "no_of_week_nights": "no_of_week_nights", 
                                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        current_year = datetime.now().year
        self.test_set["total_nights"] = self.test_set["no_of_weekend_nights"] + self.test_set["no_of_week_nights"]

        self.X_train = self.training_df[self.num_features + self.cat_features + ["total_nights"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["total_nights"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    @timeit
    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            
            self.run_id = run.info.run_id
            
            logger.debug("Start the model ...")
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            
            logger.debug("Log the model ...")
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
            
            self.fe.log_model(
                sk_model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path=f"{self.model_type}-pipeline-model-fe",
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
    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    @timeit
    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        logger.info("Loading model from MLflow alias 'production'...")
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe@latest-model"
        
        logger.info("Model successfully loaded.")

        # Make predictions
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        
        # Return predictions as a DataFrame
        return predictions

    @timeit
    def update_feature_table(self) -> None:
        """Update the hotel_reservations_features table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.{self.train_table}
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
            FROM {self.config.catalog_name}.{self.config.schema_name}.{self.train_table}
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.{self.test_table}
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
            FROM {self.config.catalog_name}.{self.config.schema_name}.{self.test_table}
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Hotel Reservations features table updated successfully.")

    def model_improved(self, test_set: DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        latest_model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe@latest-model"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics
        logger.info(f"Latest model F1-score: {metrics_old['f1_score']}")
        logger.info(f"Current model F1-score: {self.metrics['f1_score']}")
        if self.metrics["f1_score"] >= metrics_old["f1_score"]:
            logger.info("💥 Current model performs better. Returning True.")
            return True
        else:
            logger.info("⛔ Current model does not improve over latest. Returning False.")
            return False
