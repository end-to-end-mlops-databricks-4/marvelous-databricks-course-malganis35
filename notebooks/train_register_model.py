"""Main entry point for train a model and register to MLFlow on hotel reservation data."""

# Databricks notebook source

import argparse
import os
import sys

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.model.basic_model import BasicModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session, is_databricks

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
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# COMMAND ----------
# Load data and prepare features
basic_model.load_data()
basic_model.prepare_features()
logger.info("Loaded data, prepared features.")

# COMMAND ----------
# Train
basic_model.train()
logger.info("Model training completed.")

# COMMAND ----------
# log the model in MLFlow Experiment
basic_model.log_model()
logger.info("Model is logged in MLFlow Experiments.")

# COMMAND ----------
# Evaluate old and new model
model_improved = basic_model.model_improved()
logger.info(f"Model evaluation completed, model improved: {model_improved}")

# COMMAND ----------
if model_improved:
    # Register the model
    basic_model.register_model()
    logger.info("Model registration completed.")
else:
    logger.info("Model not registered as it did not improve.")

# COMMAND ----------
