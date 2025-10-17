# Databricks notebook source

# COMMAND ----------

import argparse
import datetime
import itertools
import os
import sys
import time
from typing import Any

import pretty_errors  # noqa: F401
import requests
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils

from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks
from hotel_reservation.visualization.monitoring import create_or_refresh_monitoring

## COMMAND ----------
# Global user setup
if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        root_path = ".."
        config = "project_config.yml"
        env = ".env"
        branch = "dev"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="branch of the project")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
# spark = SparkSession.builder.getOrCreate()
spark = create_spark_session()


# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.test_table}").toPandas()

# COMMAND ----------
if is_databricks():
    from pyspark.dbutils import DBUtils

    dbutils = DBUtils(spark)
    os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
else:
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    # DBR_TOKEN and DBR_HOST should be set in your .env file

    # Generate a temporary Databricks access token using the CLI
    if os.getenv("DATABRICKS_TOKEN"):
        logger.debug("Existing databricks token in .env file")
        db_token = os.getenv("DATABRICKS_TOKEN")
    else:
        logger.debug("No databricks token in .env file. Getting a temporary token ...")
        token_data = get_databricks_token(DATABRICKS_HOST)
        db_token = token_data["access_token"]
        logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")

    # Set both env var pairs for all consumers
    os.environ["DBR_TOKEN"] = db_token  # used by custom requests
    os.environ["DATABRICKS_TOKEN"] = db_token  # required by Databricks SDK / Connect
    os.environ["DBR_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST

    assert os.environ.get("DBR_TOKEN"), "DBR_TOKEN must be set in your environment or .env file."
    assert os.environ.get("DBR_HOST"), "DBR_HOST must be set in your environment or .env file."


# COMMAND ----------

workspace = WorkspaceClient()

# Required columns for inference
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

# COMMAND ----------

# Sample records for testing
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")

# %%


# 1. Using https endpoint
def send_request_https(
    dataframe_record: dict[str, Any],
    config: ProjectConfig,
    token: str,
) -> requests.Response:
    """Send an inference request to the model serving endpoint using HTTPS.

    Args:
        dataframe_record: A single record (row) of input data as a dictionary.
        config: The project configuration object.
        token: The Databricks access token.

    Returns:
        The `requests.Response` object returned by the endpoint.

    """
    model_serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{config.endpoint_name}/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# 2. Using workspace client
def send_request_workspace(
    dataframe_record: dict[str, Any],
    config: ProjectConfig,
) -> dict[str, Any]:
    """Send an inference request using the Databricks WorkspaceClient.

    Args:
        dataframe_record: A single record (row) of input data as a dictionary.
        config: The project configuration object.

    Returns:
        A dictionary response from the Databricks workspace endpoint query.

    """
    response = workspace.serving_endpoints.query(name=config.endpoint_name, dataframe_records=[dataframe_record])
    return response


# COMMAND ----------

end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
# Test the endpoint
for index, record in enumerate(itertools.cycle(sampled_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for data, index {index}")
    response = send_request_https(record, config, token=os.environ["DBR_TOKEN"])
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

workspace = WorkspaceClient()

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

# COMMAND ----------
