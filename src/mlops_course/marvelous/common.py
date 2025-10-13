"""Common modules."""

import argparse
import os
from collections.abc import Sequence

from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbr_token() -> str:
    """Retrieve the Databricks API token.

    This function obtains the API token from the Databricks notebook context.

    :return: The Databricks API token as a string.
    :raises ValueError: If not running in a Databricks environment.
    Important note: Never use your personal databricks token in real application. Create Service Principal instead.
    This is just for testing purposes
    """
    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    else:
        raise ValueError("This function is only supported on Databricks.")


def get_dbr_host() -> str:
    """Retrieve the Databricks workspace URL.

    This function obtains the workspace URL from Spark configuration.

    :return: The Databricks workspace URL as a string.
    :raises ValueError: If not running in a Databricks environment.
    """
    ws = WorkspaceClient()
    return ws.config.host


def create_parser(args: Sequence[str] = None) -> argparse.Namespace:
    """Create and configure an argument parser for MLOps on Databricks.

    This function sets up a parser with subparsers for different MLOps operations.

    :param args: Optional sequence of command-line argument strings
    :return: Parsed argument namespace
    """
    parser = argparse.ArgumentParser(description="Parser for MLOps on Databricks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--root_path", type=str, required=True, help="Path of root on DAB")
    common_args.add_argument("--env", type=str, required=True, help="Path of env file on DAB")
    common_args.add_argument("--is_test", type=int, required=True, help="1 if integration test is running")

    # Data ingestion subparser
    subparsers.add_parser("data_ingestion", parents=[common_args], help="Data ingestion options")

    # Model training & registering subparser
    model_parser = subparsers.add_parser(
        "model_train_register",
        parents=[common_args],
        help="Model training and registering options",
    )
    model_parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    model_parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
    model_parser.add_argument("--branch", type=str, required=True, help="branch of the project")

    # Deployment subparser
    subparsers.add_parser("deployment", parents=[common_args], help="Deployment options")

    # Post commit check subparser
    post_commit_check = subparsers.add_parser("post_commit_check", help="Deployment options")
    post_commit_check.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    post_commit_check.add_argument(
        "--job_run_id", type=str, required=True, help="run id of the run of the databricks job"
    )
    post_commit_check.add_argument("--job_id", type=str, required=True, help="job id of the databricks job")
    post_commit_check.add_argument("--repo", type=str, required=True, help="repo of the project")
    post_commit_check.add_argument("--org", type=str, required=True, help="org of the project")

    # Monitoring subparser
    subparsers.add_parser("monitor", parents=[common_args], help="Monitoring options")

    return parser.parse_args(args)
