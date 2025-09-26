"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import pretty_errors  # noqa: F401
import yaml
from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.feature.data_processor import DataProcessor
from mlops_course.utils.config import ProjectConfig

# COMMAND ----------
# Adjust the path: if running from this script, use "../project_config.yml" or if running from the command line in the root folder, use "./project_config.yml"
CONFIG_FILE = "./project_config.yml"
ENVIRONMENT_CHOICE="dev"

# COMMAND ----------

logger.info("Load configuration from YAML file")
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=ENVIRONMENT_CHOICE)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

try:
    logger.info("Intialize Spark session")
    spark = SparkSession.builder.getOrCreate()
except:
    logger.warning("Falling back to DatabricksSession")
    spark = DatabricksSession.builder.getOrCreate()

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
