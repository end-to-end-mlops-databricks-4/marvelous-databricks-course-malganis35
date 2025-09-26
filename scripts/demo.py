"""Demo script to showcase hotel_resa package functionality."""
# To run this script, use:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/demo.py

# %% Databricks notebook source
import pretty_errors  # noqa: F401
from loguru import logger

from mlops_course.utils.databricks_utils import create_spark_session
from mlops_course.utils.env_loader import load_environment

ENV_FILE = "./.env"

logger.info("🔧 Loading environment and Databricks configuration...")
load_environment(ENV_FILE)
logger.info("🔧 Initialize Spark Session...")
spark = create_spark_session()

logger.info("List tables in samples.nyctaxi")
df = spark.read.table("samples.nyctaxi.trips")

logger.info("Show 5 rows")
df.show(5)

logger.info("Dataframe columns:")
logger.info(list(df.columns))

# COMMAND ----------
