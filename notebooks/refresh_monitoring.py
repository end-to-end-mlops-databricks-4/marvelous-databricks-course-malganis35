# Databricks notebook source

# COMMAND ----------

import argparse
import datetime
import itertools
import os
import sys
import time
from typing import Any
import yaml

import pretty_errors  # noqa: F401
import requests
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils

from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks
from hotel_reservation.visualization.monitoring import create_or_refresh_monitoring
from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.databricks_utils import create_spark_session
from hotel_reservation.utils.env_loader import load_environment

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
ENVIRONMENT_CHOICE = args.branch

# COMMAND ----------

# Load variables from .env file
logger.info("Load environment variables and Databricks config")
load_environment(ENV_FILE)

logger.info("Load configuration from YAML file")
config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=ENVIRONMENT_CHOICE)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = create_spark_session()


# COMMAND ----------

workspace = WorkspaceClient()

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

# COMMAND ----------
