# Run the code :
# uv run scripts/run_create_mlflow_workspace.py --env-file ./.env --config-file ./project_config.yml --environment dev

import argparse
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist
from dotenv import load_dotenv
from loguru import logger
from mlflow.tracking import MlflowClient

from mlops_course.utils.config import ProjectConfig


def main(env_file: str, config_file: str, environment: str) -> None:
    """Prepare and set up an MLflow experiment on Databricks."""
    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_file)

    # ✅ Retrieve the profile only from .env (ignore CLI args)
    profile = os.getenv("PROFILE")
    if not profile:
        logger.error("❌ PROFILE is not set in .env file")
        raise ValueError("PROFILE is not set in .env file")

    logger.info(f"Using Databricks profile from .env: {profile}")

    # Load the project configuration
    config = ProjectConfig.from_yaml(config_path=config_file, env=environment)
    logger.info(f"Loaded project configuration for environment: {environment}")

    # Connect to Databricks
    w = WorkspaceClient(profile=profile)
    client = MlflowClient()
    logger.debug("Connected to Databricks and initialized MlflowClient")

    # Get the experiment path from config
    experiment_path = config.experiment_name_basic
    logger.info(f"Experiment path resolved from config: {experiment_path}")

    # Ensure parent directory exists (e.g. /Shared/experiments)
    exp_dir = "/".join(experiment_path.split("/")[:-1])  # => "/Shared/experiments"
    try:
        w.workspace.get_status(exp_dir)
        logger.debug(f"Directory already exists: {exp_dir}")
    except ResourceDoesNotExist:
        w.workspace.mkdirs(exp_dir)
        logger.success(f"Directory created: {exp_dir}")

    # Check if experiment exists and is deleted
    exp = client.get_experiment_by_name(experiment_path)
    if exp and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        logger.warning(f"Restored deleted experiment: {experiment_path}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_path)
    logger.success(f"MLflow experiment ready: {experiment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and configure MLflow experiment on Databricks")
    parser.add_argument("--env-file", default="./.env", help="Path to .env file (default: ./.env)")
    parser.add_argument("--config-file", default="./project_config.yml", help="Path to project_config.yml")
    parser.add_argument("--environment", default="dev", choices=["dev", "acc", "prd"], help="Environment to use")

    args = parser.parse_args()

    main(
        env_file=args.env_file,
        config_file=args.config_file,
        environment=args.environment,
    )
