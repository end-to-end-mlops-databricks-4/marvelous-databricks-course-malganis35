"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import pretty_errors  # noqa: F401
import yaml
from loguru import logger

from mlops_course.feature.data_processor import DataProcessor
from mlops_course.utils.config import ProjectConfig
from mlops_course.utils.databricks_utils import create_spark_session
from mlops_course.utils.env_loader import load_environment

# COMMAND ----------
# Adjust the path: if running from this script, use "../project_config.yml" or if running from the command line in the root folder, use "./project_config.yml"
CONFIG_FILE = "./project_config.yml"
ENVIRONMENT_CHOICE = "dev"
ENV_FILE = "./.env"

# COMMAND ----------

# Load variables from .env file
logger.info("Load environment variables and Databricks config")
load_environment(ENV_FILE)

logger.info("Load configuration from YAML file")
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=ENVIRONMENT_CHOICE)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# try:
#     logger.info("Intialize Spark session")
#     spark = SparkSession.builder.getOrCreate()
# except Exception as e:
#     logger.warning(f"Falling back to DatabricksSession due to: {e}")
#     spark = DatabricksSession.builder.getOrCreate()

# from databricks.connect import DatabricksSession

# compute_mode = os.getenv("DATABRICKS_COMPUTE", "serverless")
# cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

# builder = DatabricksSession.builder

# if cluster_id and compute_mode.lower() == "cluster":
#     logger.info(f"Using Databricks cluster: {cluster_id}")
#     spark = builder.remote(cluster_id=cluster_id).getOrCreate()
# elif compute_mode.lower() == "serverless":
#     logger.info("Using Databricks Serverless compute")
#     spark = builder.remote(serverless=True).getOrCreate()
# else:
#     logger.info("No compute specified, defaulting to serverless")
#     spark = builder.remote(serverless=True).getOrCreate()

# logger.info("✅ Spark session initialized successfully")

# Créer la session Spark connectée à Databricks
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
