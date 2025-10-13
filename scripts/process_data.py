"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import argparse
import sys

import pretty_errors  # noqa: F401
import yaml
from loguru import logger

from hotel_reservation.feature.data_processor import DataProcessor
from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session
from hotel_reservation.utils.env_loader import load_environment

# COMMAND ----------

if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        root_path = ".."
        config = "project_config.yml"
        env = ".env"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", help="branch of the project")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"
ENVIRONMENT_CHOICE = args.branch

# COMMAND ----------

# Load variables from .env file
logger.info("Load environment variables and Databricks config")
load_environment(ENV_FILE)

logger.info("Load configuration from YAML file")
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=ENVIRONMENT_CHOICE)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

spark = create_spark_session()

logger.info("Load the hotel reservations dataset from the catalog")
df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/{config.raw_data_file}", header=True, inferSchema=True
).toPandas()
logger.info(f"Dataset shape: {df.shape}")

# COMMAND ----------

logger.info("Preprocess the data")
data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

logger.info("Split the data into train and test sets")
X_train, X_test = data_processor.split_data()

logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# COMMAND ----------
