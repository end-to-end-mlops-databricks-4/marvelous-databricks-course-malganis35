"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse
import os
import sys

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.serving.model_serving import ModelServing
from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks

# COMMAND ----------

# Global setup
if "ipykernel" in sys.modules:
    # Running interactively (e.g. in Jupyter), mock CLI arguments
    class Args:
        """Mock arguments used when running interactively."""

        root_path = ".."
        config = "project_config.yml"
        env = ".env"
        branch = "dev"
        model_version = "auto"

    args = Args()
else:
    # Standard CLI usage
    parser = argparse.ArgumentParser(description="Deploy model as Databricks serving endpoint.")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="Project branch to use")
    parser.add_argument("--model_version", type=str, default="auto")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------

# Load environment and setup MLflow tracking
if not is_databricks():
    logger.info("Code is not executed on Databricks. Setting the local environment variables ...")
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


# COMMAND ----------

# Generate a temporary Databricks access token using the CLI
token_data = get_databricks_token(DATABRICKS_HOST)
db_token = token_data["access_token"]

# Set both env var pairs for all consumers
os.environ["DBR_TOKEN"] = db_token  # used by custom requests
os.environ["DATABRICKS_TOKEN"] = db_token  # required by Databricks SDK / Connect
os.environ["DBR_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST

logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")

# COMMAND ----------

# Load configuration and Spark session
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
spark = create_spark_session()

model_name_to_deploy = f"{config.catalog_name}.{config.schema_name}.{config.model_name}"

# COMMAND ----------

# Main script to serve the endpoint of the model
serving = ModelServing(model_name=model_name_to_deploy, endpoint_name="hotel-reservation-basic-model-serving-db")

if args.model_version == "auto":
    logger.info("Model Version is set to default 'auto'. Finding the last version of the model in Unity Catalog")
    entity_version_latest_ready = serving.get_latest_ready_version()
    logger.info(f"Version of the model that will be deployed: {entity_version_latest_ready}")
else:
    entity_version_latest_ready = args.model_version
    logger.info(f"Version of the model defined by the user and that will be deployed: {entity_version_latest_ready}")

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
