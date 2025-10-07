"""Main entry point for train a model and register to MLFlow on hotel reservation data."""

# Databricks notebook source

import argparse

import pretty_errors  # noqa: F401
from loguru import logger
from pyspark.dbutils import DBUtils

from hotel_reservation import __version__ as hotel_reservation_v
from hotel_reservation.model.custom_model import CustomModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session

# COMMAND ----------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)

args = parser.parse_args()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# COMMAND ----------

# Create the spark session
spark = create_spark_session()
dbutils = DBUtils(spark)


# COMMAND ----------
# Initialize model
# Initialize model with the config path
custom_model = CustomModel(
    config=config, 
    tags=tags, 
    spark=spark,
    code_paths=[f"{root_path}/dist/hotel_reservation-{hotel_reservation_v}-py3-none-any.whl"],
)
logger.info("Model initialized.")

# COMMAND ----------
# Load data and prepare features
custom_model.load_data()
custom_model.prepare_features()
logger.info("Loaded data, prepared features.")

# COMMAND ----------
# Train
custom_model.train()
logger.info("Model training completed.")

# COMMAND ----------
# log the model in MLFlow Experiment
custom_model.log_model()
logger.info("Model is logged in MLFlow Experiments.")

# COMMAND ----------
# Evaluate old and new model
model_improved = custom_model.model_improved()
logger.info(f"Model evaluation completed, model improved: {model_improved}")

is_test = args.is_test

# when running test, always register and deploy
if is_test == args.is_test:
    model_improved = True

# COMMAND ----------
if model_improved:
    # Register the model
    latest_version = custom_model.register_model()
    logger.info("Model registration completed.")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    logger.info("Model not registered as it did not improve.")
    dbutils.jobs.taskValues.set(key="model_updated", value=0)

# COMMAND ----------
