"""Demo script to showcase hotel_resa package functionality."""
# To run this script, use:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/demo.py

# %% Databricks notebook source
import argparse
import sys

import pretty_errors  # noqa: F401
from loguru import logger

from hotel_reservation.utils.databricks_utils import create_spark_session
from hotel_reservation.utils.env_loader import load_environment

if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        root_path = ".."
        env = ".env"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--env", type=str, default=".env")
    args = parser.parse_args()

root_path = args.root_path
ENV_FILE = f"{root_path}/{args.env}"

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
