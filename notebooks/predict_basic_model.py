# COMMAND ----------

# Generate a temporary token: databricks auth token --host https://dbc-f122dc18-1b68.cloud.databricks.com

# Configure tracking uri
import argparse
import os
import sys

import mlflow
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.model.basic_model import BasicModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session, is_databricks

## COMMAND ----------
# Global user setup
if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        root_path = ".."
        config = "project_config.yml"
        env = ".env"
        git_sha = "abcd12345"
        job_run_id = "local_test_run"
        branch = "dev"

    args = Args()
else:
    # Normal CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--config", type=str, default="project_config.yml")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
    parser.add_argument("--branch", type=str, default="dev", required=True, help="branch of the project")
    args = parser.parse_args()

root_path = args.root_path
CONFIG_FILE = f"{root_path}/{args.config}"
ENV_FILE = f"{root_path}/{args.env}"

# COMMAND ----------
if not is_databricks():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=args.branch)
# spark = SparkSession.builder.getOrCreate()
spark = create_spark_session()

tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)


# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.test_table}").limit(10)

# Drop feature lookup columns and target
# X_test = test_set.drop("no_of_weekend_nights", "no_of_week_nights", config.target)
X_test = test_set

# COMMAND ----------

basic_model = BasicModel(config=config, tags=tags, spark=spark)

# %% Make predictions
predictions = basic_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------
