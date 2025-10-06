"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import argparse

import pretty_errors  # noqa: F401
import yaml
from loguru import logger

from hotel_reservation.feature.data_processor import DataProcessor, generate_synthetic_data, generate_test_data
from hotel_reservation.utils.config import ProjectConfig
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
    "--write_mode",
    action="store",
    default="overwrite",
    type=str,
    choices=["overwrite", "append", "upsert"],
    help="How to write data to the catalog: overwrite, append, or upsert",
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

write_mode = args.write_mode

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

spark = create_spark_session()

logger.info("Load the hotel reservations dataset from the catalog")
df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/{config.raw_data_file}", header=True, inferSchema=True
).toPandas()
logger.info(f"Dataset shape: {df.shape}")


logger.debug("===== Information about the dataset: =====")
logger.debug(f"Size of the dataset: {df.shape}")
logger.debug("Top 10 lines of the dataset:")
logger.debug(df.head(10))

# COMMAND ----------

logger.info("Initialize DataProcessor ...")
data_processor = DataProcessor(df, config, spark)

logger.info("Preprocess the data ...")
data_processor.preprocess()

logger.info("Split the data into train and test sets")
X_train, X_test = data_processor.split_data()

logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test, write_mode=write_mode)

# COMMAND ----------
