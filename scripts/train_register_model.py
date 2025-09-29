"""Main entry point for train a model and register to MLFlow on hotel reservation data."""

# Databricks notebook source

import os

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from mlops_course.marvelous.common import is_databricks
from mlops_course.model.basic_model import BasicModel
from mlops_course.utils.config import ProjectConfig, Tags
from mlops_course.utils.databricks_utils import create_spark_session

## COMMAND ----------
# Global user setup

ENV_FILE = "../.env"
CONFIG_FILE = "../project_config.yml"
ENVIRONMENT_CHOICE = "dev"

# COMMAND ----------
if not is_databricks():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=ENVIRONMENT_CHOICE)
# spark = SparkSession.builder.getOrCreate()
spark = create_spark_session()

tags_dict = {"git_sha": "abcd12345", "branch": "week2", "job_run_id": ""}
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
logger.info("Model evaluation completed, model improved: %s", model_improved)

# COMMAND ----------
if model_improved:
    # Register the model
    basic_model.register_model()
    logger.info("Model registration completed.")
    
# COMMAND ----------
