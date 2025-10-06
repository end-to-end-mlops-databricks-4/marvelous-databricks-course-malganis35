"""Main entry point for deploying a model as a serving endpoint in Databricks."""

# COMMAND ----------

import argparse
import os
import sys
from typing import Any

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

# Load environment and setup MLflow tracking
if not is_databricks():
    logger.info("Code is not executed on Databricks. Setting the local environment variables ...")
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# COMMAND ----------

# Generate a temporary Databricks access token using the CLI
token_data = get_databricks_token(DATABRICKS_HOST)
db_token = token_data["access_token"]

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
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])

# COMMAND ----------


# Endpoint call function
def call_endpoint(record: list[dict[str, Any]]) -> dict[str, Any]:
    """Call the Databricks model serving endpoint and parse the response robustly."""
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{config.endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"}

    logger.debug(f"Calling endpoint: {serving_endpoint}")

    response = requests.post(serving_endpoint, headers=headers, json={"dataframe_records": record})

    if response.status_code != 200:
        logger.error(f"❌ Endpoint error {response.status_code}: {response.text}")
        return {"status": "error", "code": response.status_code, "response": response.text}

    # Parse response safely
    try:
        result_json = response.json()
        logger.debug(f"Raw response: {result_json}")

        # Case 1️⃣ — direct predictions field
        if "predictions" in result_json:
            preds = result_json["predictions"]

        # Case 2️⃣ — MLflow Serving “outputs” key
        elif "outputs" in result_json:
            preds = result_json["outputs"]

        else:
            preds = result_json

        # Case 3️⃣ — Predictions contain dicts (probability + label)
        if isinstance(preds[0], dict):
            pred = preds[0]
            label = pred.get("label", str(pred.get("prediction", "unknown")))
            probability = pred.get("probability", None)
            return {"label": label, "probability": probability}

        # Case 4️⃣ — Simple list of numeric predictions
        elif isinstance(preds[0], list):
            prob = preds[0][1] if len(preds[0]) > 1 else preds[0][0]
            pred_class = int(prob >= 0.5)
            label = "cancelled" if pred_class == 1 else "not_cancelled"
            return {"label": label, "probability": prob}

        # Case 5️⃣ — Simple scalar or string
        else:
            value = preds[0]
            if isinstance(value, str):
                label = value
                probability = None
            else:
                probability = float(value)
                label = "cancelled" if probability >= 0.5 else "not_cancelled"
            return {"label": label, "probability": probability}

    except Exception as e:
        logger.error(f"⚠️ Unable to parse response: {e}")
        return {"status": "error", "raw_response": response.text}


# Test with one sample
result = call_endpoint(dataframe_records[0])
logger.info(f"✅ Parsed response: {result}")

# COMMAND ----------

# Simple load test
# total = 10
# for i in range(total):
#     logger.debug(f"➡️ Sending request {i + 1}/{total}")
#     status_code, response_text = call_endpoint(dataframe_records[i])
#     logger.debug(f"Response Status: {status_code}")
#     logger.debug(f"Response Text: {response_text}")
#     time.sleep(0.2)
