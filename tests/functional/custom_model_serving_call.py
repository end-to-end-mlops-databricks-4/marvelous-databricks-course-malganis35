"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse
import os
import sys
import time

import pandas as pd
import pretty_errors  # noqa: F401
import requests
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks

# COMMAND ----------

# Global setup
if "ipykernel" in sys.modules:
    # Running interactively (e.g. in Jupyter), mock CLI arguments
    class Args:
        """Mock arguments used when running interactively."""

        root_path = "../.."
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

# Load environment and setup MLflow tracking
if not is_databricks():
    logger.info("Code is not executed on Databricks. Setting the local environment variables ...")
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# COMMAND ----------

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

# COMMAND ----------

# Load configuration and Spark session
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
spark = create_spark_session()

# COMMAND ----------

# Prepare sample request data
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

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.train_table}").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
# dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
# logger.info(dataframe_records[0])

# COMMAND ----------


def call_endpoint(dataset: pd.DataFrame) -> tuple[int, dict]:
    """Send a Pandas DataFrame to the Databricks Serving endpoint.

    Sends the request and returns the HTTP status code and the JSON response.
    """
    url = "https://dbc-c36d09ec-dbbe.cloud.databricks.com/serving-endpoints/hotel-reservation-custom-model-serving-db/invocations"
    headers = {
        "Authorization": f"Bearer {os.environ.get('DATABRICKS_TOKEN')}",
        "Content-Type": "application/json",
    }

    # Vérification du type d'entrée
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("❌ dataset must be a pandas DataFrame")

    # Conversion en format attendu par Databricks Serving
    payload = {"dataframe_split": dataset.to_dict(orient="split")}

    # Appel du endpoint
    response = requests.post(url, headers=headers, json=payload)

    # Gestion des erreurs HTTP
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    # Retourne la réponse JSON brute
    return response.status_code, response.json()


# Test on 1 line
sample_df = pd.DataFrame([sampled_records[0]])

# Endpoint call
status_code, response_text = call_endpoint(sample_df)

# Print the json
logger.debug(f"Response Status: {status_code}")
logger.debug(f"Response Text: {response_text}")

# COMMAND ----------

# Simple load test
total = 10
for i in range(total):
    logger.debug(f"➡️ Sending request {i + 1}/{total}")
    status_code, response_text = call_endpoint(pd.DataFrame([sampled_records[i]]))
    logger.debug(f"Response Status: {status_code}")
    logger.debug(f"Response Text: {response_text}")
    time.sleep(0.2)
