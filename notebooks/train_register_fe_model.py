# Databricks notebook source
# %pip install house_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# %restart_python

# COMMAND ----------

# Generate a temporary token: databricks auth token --host https://dbc-f122dc18-1b68.cloud.databricks.com

# Configure tracking uri
import argparse
import os
import sys

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.model.feature_lookup_model import FeatureLookUpModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks

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
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

    # Get a temporary token for Databricks Connect & SDK
    token_data = get_databricks_token(DATABRICKS_HOST)
    db_token = token_data["access_token"]

    # Define the variable for Databricks SDK
    os.environ["DATABRICKS_TOKEN"] = db_token
    os.environ["DBR_TOKEN"] = db_token
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DBR_HOST"] = DATABRICKS_HOST

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
# Register the model
fe_model.register_model()

# COMMAND ----------
# # Evaluate old and new model
# try:
#     model_improved = fe_model.model_improved()
#     logger.info(f"Model evaluation completed, model improved: {model_improved}")
# except Exception as e:
#     logger.error(f"Model evaluation encountered an issue: {e}")
#     model_improved = False


# # COMMAND ----------
# if model_improved:
#     # Register the model
#     fe_model.register_model()
#     logger.info("Model registration completed.")
# else:
#     logger.info("Model not registered as it did not improve or has encountered an error.")
