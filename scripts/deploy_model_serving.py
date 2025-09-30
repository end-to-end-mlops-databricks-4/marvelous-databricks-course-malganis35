"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse
import sys
import os
import time
import requests

import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger
import mlflow
from typing import Dict, List
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from hotel_reservation.marvelous.common import is_databricks

from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.serving.model_serving import ModelServing
from hotel_reservation.utils.databricks_utils import create_spark_session

# COMMAND ----------
# Global user setup
if "ipykernel" in sys.modules:
    # Running interactively (e.g. in Jupyter), mock CLI arguments
    class Args:
        """Mock arguments used when running interactively."""
        root_path = ".."
        config = "project_config.yml"
        env = ".env"
        branch = "dev"
    args = Args()

else:
    # Standard CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="Project branch to use")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------
if not is_databricks():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# COMMAND ----------
# Generate a temporary Databricks access token using the CLI
import subprocess
import json

logger.info("🔑 Automatically generating a Databricks temporary token via CLI...")
result = subprocess.run(
    ["databricks", "auth", "token", "--host", DATABRICKS_HOST, "--output", "JSON"],
    capture_output=True,
    text=True,
    check=True
)

# Parse JSON output
token_data = json.loads(result.stdout)
db_token = token_data["access_token"]

# Set both env var pairs for all consumers
os.environ["DBR_TOKEN"] = db_token              # used by custom requests
os.environ["DATABRICKS_TOKEN"] = db_token       # required by Databricks SDK / Connect
os.environ["DBR_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST

logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")

# COMMAND ----------
# Load configuration and Spark session
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
spark = create_spark_session()

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model_name

# COMMAND ----------
# Check for existing model versions in MLflow
client = mlflow.MlflowClient()
model_name_to_deploy = f"{catalog_name}.{schema_name}.{model_name}"

# Retrieve all READY versions
ready_versions = [
    int(mv.version)
    for mv in client.search_model_versions(f"name='{model_name_to_deploy}'")
    if mv.status == "READY"
]

if not ready_versions:
    raise ValueError(f"No READY version found for model {model_name_to_deploy}")

entity_version_latest_ready = str(max(ready_versions))
logger.info(f"✅ Latest READY model version: {entity_version_latest_ready}")

# COMMAND ----------
"""Model serving deployment section."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
model_name = f"{catalog_name}.{schema_name}.{model_name}"
endpoint_name = "hotel-reservation-basic-model-serving-db"
entity_version = entity_version_latest_ready
# os.environ["DBR_HOST"] = w.config.host
# os.environ["DBR_TOKEN"] = db_token

served_entities = [
    ServedEntityInput(
        entity_name=model_name_to_deploy,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=entity_version,
        environment_vars={
            "aws_access_key_id": "{{secrets/mlops/aws_access_key_id}}",
            "aws_secret_access_key": "{{secrets/mlops/aws_access_key}}",
            "region_name": "eu-west-1",
        }
    )
]

# Check if the serving endpoint already exists
existing_endpoints = [e.name for e in w.serving_endpoints.list()]

if endpoint_name in existing_endpoints:
    logger.info(f"🔄 Endpoint '{endpoint_name}' already exists — updating configuration...")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities,
    )
    logger.info(f"✅ Endpoint '{endpoint_name}' updated with model version {entity_version}")
else:
    logger.info(f"🚀 Creating new endpoint '{endpoint_name}'...")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(served_entities=served_entities),
    )
    logger.success(f"✅ Endpoint '{endpoint_name}' created with model version {entity_version}")

# COMMAND ----------
# Wait until endpoint becomes READY
def wait_until_endpoint_ready(workspace, endpoint_name, timeout=600, check_interval=10):
    """Wait for the Databricks endpoint to reach READY state."""
    start_time = time.time()
    logger.info(f"⏳ Waiting for endpoint '{endpoint_name}' to become READY...")

    while time.time() - start_time < timeout:
        endpoint = workspace.serving_endpoints.get(endpoint_name)
        state = endpoint.state.ready
        state_str = state.value if hasattr(state, "value") else state
        config_update = getattr(endpoint.state, "config_update", None)

        logger.info(f"➡️  Current state: {state_str}")
        if config_update and hasattr(config_update, "state"):
            logger.info(f"   - Update details: {config_update.state}")

        if state_str == "READY":
            logger.success(f"✅ Endpoint '{endpoint_name}' is READY to serve requests!")
            return  # Exit function cleanly

        if state_str == "NOT_READY" and config_update and getattr(config_update, "state", "") == "UPDATE_FAILED":
            raise RuntimeError(f"❌ Deployment failed for endpoint '{endpoint_name}'")

        time.sleep(check_interval)

    raise TimeoutError(f"❌ Timeout: endpoint '{endpoint_name}' did not become READY after {timeout} seconds.")

# Wait for endpoint
wait_until_endpoint_ready(w, endpoint_name)

# COMMAND ----------
# Create a sample request body
required_columns = [
    "arrival_month",
    "arrival_year",
    "avg_price_per_room",
    "lead_time",
    "no_of_adults",
    "no_of_children",
    "no_of_previous_bookings_not_canceled",
    "no_of_previous_cancellations",
    "no_of_special_requests",
    "no_of_week_nights",
    "no_of_weekend_nights",
    "repeated_guest",
    "required_car_parking_space",
    "market_segment_type",
    "room_type_reserved",
    "type_of_meal_plan",
]

train_set = spark.table(f"{catalog_name}.{schema_name}.{config.train_table}").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each record in the request body should be a list of JSON objects like:

[{
    "arrival_month": xxx,
    "arrival_year": xxx,
    "avg_price_per_room": xxx,
    "lead_time": xxx,
    "no_of_adults": xxx,
    "no_of_children": xxx,
    "no_of_previous_bookings_not_canceled": xxx,
    "no_of_previous_cancellations": xxx,
    "no_of_special_requests": xxx,
    "no_of_week_nights": xxx,
    "no_of_weekend_nights": xxx,
    "repeated_guest": xxx,
    "required_car_parking_space": xxx,
    "market_segment_type": xxx,
    "room_type_reserved": xxx,
    "type_of_meal_plan": xxx,
}]
"""

def call_endpoint(record):
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/hotel-reservation-basic-model-serving-db/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

status_code, response_text = call_endpoint(dataframe_records[0])
logger.debug(f"Response Status: {status_code}")
logger.debug(f"Response Text: {response_text}")

# COMMAND ----------
# Simple load test
total = 10
for i in range(total):
    logger.debug(f"➡️ Sending request {i+1}/{range(total)}")
    status_code, response_text = call_endpoint(dataframe_records[i])
    logger.debug(f"Response Status: {status_code}")
    logger.debug(f"Response Text: {response_text}")
    time.sleep(0.2)
