# Databricks notebook source
# %pip install house_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# %restart_python

# COMMAND ----------

# Configure tracking uri
import argparse
import os
import sys

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from hotel_reservation.marvelous.common import is_databricks
from hotel_reservation.model.feature_lookup_model import FeatureLookUpModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

## COMMAND ----------
# Global user setup
if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        root_path = ".."
        config = "project_config.yml"
        env = ".env"
        git_sha = "abcd12345"
        job_run_id = "local_test_run"
        branch = "dev"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="branch of the project")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------
if not is_databricks():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
# spark = SparkSession.builder.getOrCreate()
spark = create_spark_session()

tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define house age feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("OverallQual", "GrLivArea", "GarageCars", config.target)


# COMMAND ----------

X_test = (
    X_test.withColumn("LotArea", col("LotArea").cast("int"))
    .withColumn("OverallCond", col("OverallCond").cast("int"))
    .withColumn("YearBuilt", col("YearBuilt").cast("int"))
    .withColumn("YearRemodAdd", col("YearRemodAdd").cast("int"))
    .withColumn("TotalBsmtSF", col("TotalBsmtSF").cast("int"))
)


# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------
