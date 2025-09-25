"""Demo script to showcase hotel_resa package functionality."""
# To run this script, use:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/demo.py

# %% Databricks notebook source
import pretty_errors  # noqa: F401
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from loguru import logger

try:
    spark = SparkSession.builder.getOrCreate()
except:
    logger.warning("Falling back to DatabricksSession")
    spark = DatabricksSession.builder.getOrCreate()

df = spark.read.table("samples.nyctaxi.trips")
df.show(5)

# COMMAND ----------

print(list(df.columns))

# COMMAND ----------
