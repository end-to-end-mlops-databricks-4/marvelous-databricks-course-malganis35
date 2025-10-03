"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import argparse
import sys

import pretty_errors  # noqa: F401
import yaml
from loguru import logger

from hotel_reservation.feature.data_processor import DataProcessor, generate_synthetic_data, generate_test_data
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
        branch = "dev"
        is_test = 0
        write_mode = "append"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", help="branch of the project")
    parser.add_argument("--is_test", type=int, default=0, help="synthetic data (0) and test data (1)")
    parser.add_argument(
        "--write_mode",
        type=str,
        default="append",
        choices=["overwrite", "append", "upsert"],
        help="How to write data to the catalog: overwrite, append, or upsert",
    )
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"
ENVIRONMENT_CHOICE = args.branch
is_test = args.is_test
write_mode = args.write_mode

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
logger.info("Generating data ...")
if is_test == 0:
    # Generate synthetic data.
    # This is mimicking a new data arrival. In real world, this would be a new batch of data.
    # df is passed to infer schema
    new_data = generate_synthetic_data(df, num_rows=10)
    logger.success("✅ Synthetic data generated successfully")
else:
    # Generate synthetic data
    # This is mimicking a new data arrival. This is a valid example for integration testing.
    new_data = generate_test_data(df, num_rows=10)
    logger.success("✅ Test data generated successfully.")

logger.debug("===== Information about the dataset: =====")
logger.debug(f"Size of the dataset: {new_data.shape}")
logger.debug("Top 10 lines of the dataset:")
logger.debug(new_data.head(10))

# COMMAND ----------

logger.info("Initialize DataProcessor ...")
data_processor = DataProcessor(new_data, config, spark)

logger.info("Preprocess the data ...")
data_processor.preprocess()

logger.info("Split the data into train and test sets")
X_train, X_test = data_processor.split_data()

logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test, write_mode=write_mode)

# COMMAND ----------
