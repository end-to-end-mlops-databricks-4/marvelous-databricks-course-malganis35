"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse

import pretty_errors  # noqa: F401
from loguru import logger
from pyspark.dbutils import DBUtils

from hotel_reservation.serving.model_serving import ModelServing
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
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)

parser.add_argument(
    "--model_version",
    action="store",
    default="auto",
    type=str,
    required=False,
    help="Specify model version to deploy. Use 'auto' to take the latest from training task.",
)

args = parser.parse_args()
root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config.yml"

# COMMAND ----------

# Create the spark session
spark = create_spark_session()
dbutils = DBUtils(spark)
if args.model_version == "auto":
    logger.info("Model Version is set to 'auto'. Retrieving version from previous training task.")
    model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
else:
    model_version = args.model_version
logger.info(f"Model version selected for deployment: {model_version}")

# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"{config.endpoint_name_custom}-{args.env}"

# COMMAND ----------

model_name_to_deploy = f"{config.catalog_name}.{config.schema_name}.{config.model_name_custom}"

# COMMAND ----------

# Main script to serve the endpoint of the model
serving = ModelServing(model_name=model_name_to_deploy, endpoint_name=endpoint_name)

if model_version == "auto":
    logger.info("Model version 'auto' detected. Fetching the latest ready version in Unity Catalog...")
    entity_version_latest_ready = serving.get_latest_ready_version()
else:
    entity_version_latest_ready = model_version

logger.info(f"Version of the model that will be deployed: {entity_version_latest_ready}")

logger.info("Checking that the endpoint is not busy")
serving.wait_until_ready()

serving.deploy_or_update_serving_endpoint(
    version=entity_version_latest_ready,
    environment_vars={
        "aws_access_key_id": "{{secrets/mlops/aws_access_key_id}}",
        "aws_secret_access_key": "{{secrets/mlops/aws_access_key}}",
        "region_name": "eu-west-1",
    },
)

logger.info("Checking when the endpoint is ready")
serving.wait_until_ready()
