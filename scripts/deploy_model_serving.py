"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import mlflow
import pretty_errors  # noqa: F401
import requests
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.marvelous.common import is_databricks
from hotel_reservation.serving.model_serving import ModelServing
from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session

# COMMAND ----------

# -------------------------------------------------------------------------
# Global setup
# -------------------------------------------------------------------------
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
    parser = argparse.ArgumentParser(description="Deploy model as Databricks serving endpoint.")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="Project branch to use")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------

# -------------------------------------------------------------------------
# Load environment and setup MLflow tracking
# -------------------------------------------------------------------------
if not is_databricks():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# COMMAND ----------

# -------------------------------------------------------------------------
# Generate a temporary Databricks access token using the CLI
# -------------------------------------------------------------------------
logger.info("🔑 Automatically generating a Databricks temporary token via CLI...")

result = subprocess.run(
    ["databricks", "auth", "token", "--host", DATABRICKS_HOST, "--output", "JSON"],
    capture_output=True,
    text=True,
    check=True,
)

token_data = json.loads(result.stdout)
db_token = token_data["access_token"]

# Set both env var pairs for all consumers
os.environ["DBR_TOKEN"] = db_token  # used by custom requests
os.environ["DATABRICKS_TOKEN"] = db_token  # required by Databricks SDK / Connect
os.environ["DBR_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST

logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")

# COMMAND ----------

# -------------------------------------------------------------------------
# Load configuration and Spark session
# -------------------------------------------------------------------------
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
spark = create_spark_session()

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model_name

# COMMAND ----------

# -------------------------------------------------------------------------
# Find latest READY model version
# -------------------------------------------------------------------------
client = mlflow.MlflowClient()
model_name_to_deploy = f"{catalog_name}.{schema_name}.{model_name}"

ready_versions = [
    int(mv.version) for mv in client.search_model_versions(f"name='{model_name_to_deploy}'") if mv.status == "READY"
]

if not ready_versions:
    raise ValueError(f"No READY version found for model {model_name_to_deploy}")

entity_version_latest_ready = str(max(ready_versions))
logger.info(f"✅ Latest READY model version: {entity_version_latest_ready}")

# COMMAND ----------

serving = ModelServing(model_name=model_name_to_deploy, endpoint_name="hotel-reservation-basic-model-serving-db")

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


# COMMAND ----------

# -------------------------------------------------------------------------
# Prepare sample request data
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Endpoint call function
# -------------------------------------------------------------------------
def call_endpoint(record: list[dict[str, Any]]) -> tuple[int, str]:
    """Call the Databricks model serving endpoint with a given input record."""
    serving_endpoint = (
        f"{os.environ['DBR_HOST']}/serving-endpoints/hotel-reservation-basic-model-serving-db/invocations"
    )

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# Test with one sample
status_code, response_text = call_endpoint(dataframe_records[0])
logger.debug(f"Response Status: {status_code}")
logger.debug(f"Response Text: {response_text}")

# COMMAND ----------

# -------------------------------------------------------------------------
# Simple load test
# -------------------------------------------------------------------------
total = 10
for i in range(total):
    logger.debug(f"➡️ Sending request {i + 1}/{total}")
    status_code, response_text = call_endpoint(dataframe_records[i])
    logger.debug(f"Response Status: {status_code}")
    logger.debug(f"Response Text: {response_text}")
    time.sleep(0.2)
