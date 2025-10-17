"""Main entry point for preprocessing hotel reservation data."""
# Run the script:
# DATABRICKS_CONFIG_PROFILE=cohort4 uv run scripts/process_data.py

# %% Databricks notebook source

import argparse
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pretty_errors  # noqa: F401
import requests
import yaml
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql.functions import lit
from pyspark.sql.utils import AnalysisException

from hotel_reservation.feature.data_processor import DataProcessor
from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks

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
    default="upsert",
    type=str,
    choices=["overwrite", "append", "upsert"],
    help="How to write data to the catalog: overwrite, append, or upsert",
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
PREDICTION_COL = "booking_status_pred_custom"
PROBABILITY_COL = "booking_status_proba_custom"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

write_mode = args.write_mode

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

spark = create_spark_session()


# COMMAND ----------
df = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.batch_inference_table}")  # .toPandas()

lst_col_check = [PREDICTION_COL, PROBABILITY_COL]

for col in lst_col_check:
    # Vérifier si la colonne 'booking_status_pred' existe
    if col not in df.columns:
        logger.warning(f"Column {col} is absent from the dataset. Add a NULL column.")
        df = df.withColumn(col, lit(None).cast("string"))


    full_table_name = f"{config.catalog_name}.{config.schema_name}.{config.batch_inference_table}"

    try:
        spark.sql(f"ALTER TABLE {full_table_name} ADD COLUMNS ({col} STRING)")
        logger.success(f"✅ Column {col} added to {full_table_name}")
    except AnalysisException as e:
        if "already exists" in str(e):
            logger.info(f"✅ Column {col} is already present.")
        else:
            raise

    # Recharge la table avec la nouvelle colonne
    df = spark.table(full_table_name)

# Maintenant, la colonne existe toujours, même si elle est vide
df_to_predict = df.filter(df.booking_status_pred_custom.isNull())

# COMMAND ----------
if is_databricks():
    from pyspark.dbutils import DBUtils

    dbutils = DBUtils(spark)
    os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
    os.environ["DBR_HOST"] = (
        os.environ["DBR_HOST"] if os.environ["DBR_HOST"].startswith("https://") else f"https://{os.environ['DBR_HOST']}"
    )
    logger.info(f"Databricks host URL: {os.environ['DBR_HOST']}")
else:
    load_dotenv(dotenv_path=args.env, override=True)
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

# Take the records that do not have a prediction yet
# sampled_records = prediction_set[required_columns].to_dict(orient="records")
records_to_predict = df_to_predict.toPandas().to_dict(orient="records")

# %%


def clean_record_for_json(record: dict) -> dict:
    """Convert Pandas/Numpy/Timestamp types to JSON serializable types."""
    clean_record = {}
    for k, v in record.items():
        if isinstance(v, np.int64 | np.int32):
            clean_record[k] = int(v)
        elif isinstance(v, np.float64 | np.float32):
            clean_record[k] = float(v)
        elif isinstance(v, pd.Timestamp | datetime):
            clean_record[k] = v.isoformat()
        elif pd.isna(v):
            clean_record[k] = None
        else:
            clean_record[k] = v
    return clean_record


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
    endpoint_name = f"{config.endpoint_name_custom}-{args.env}"
    model_serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    logger.debug(f"Sending request to endpoint: {model_serving_endpoint}")

    dataframe_record = clean_record_for_json(dataframe_record)

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

predictions = []

# Send to the endpoint
for index, record in enumerate(records_to_predict):
    print(f"Sending request for data, index {index + 1}/{len(records_to_predict)}")
    response = send_request_https(record, config, token=os.environ["DBR_TOKEN"])
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    prediction_value = None
    probability_value = None

    if response.status_code == 200:
        try:
            result = response.json()

            # Extract the values
            predictions_list = result.get("predictions", [])
            if predictions_list and isinstance(predictions_list[0], dict):
                prediction_value = predictions_list[0].get("label")
                probability_value = predictions_list[0].get("probability")

        except Exception as e:
            logger.error(f"Erreur lors du parsing de la réponse : {e}")

    else:
        logger.error(f"Erreur HTTP {response.status_code}: {response.text}")

    print(f"Prediction: {prediction_value}")
    print(f"Probability: {probability_value}")

    record[PREDICTION_COL] = prediction_value
    record[PROBABILITY_COL] = probability_value
    predictions.append(record)

    time.sleep(0.05)

# COMMAND ----------
# --- Save back to Databricks ---
pred_df = pd.DataFrame(predictions)

data_processor = DataProcessor(pred_df, config, spark)
data_processor.save_to_catalog(pred_df, pred_df, write_mode=args.write_mode, job_type="inference")

logger.success("✅ Prediction save in Unity Catalog")
