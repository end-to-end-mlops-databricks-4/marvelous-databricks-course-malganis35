# Run this script
# uv run scripts/run_cleanup_mlflow_experiments.py --env-file ./.env --config-file ./project_config.yml --environment dev

import argparse
import os

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.tracking import MlflowClient

from mlops_course.utils.config import ProjectConfig


def main(env_file: str, config_file: str, environment: str, yes: bool = False) -> None:
    """Delete (mark as deleted) MLflow experiments defined in project_config.yml on Databricks."""
    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_file, override=True)

    # ✅ Retrieve the profile only from .env (ignore CLI args)
    profile = os.getenv("PROFILE")
    if not profile:
        logger.error("❌ PROFILE is not set in .env file")
        raise ValueError("PROFILE is not set in .env file")

    logger.info(f"Using Databricks profile from .env: {profile}")

    # Load project config
    config = ProjectConfig.from_yaml(config_path=config_file, env=environment)
    logger.info(f"Loaded project config for environment: {environment}")

    # Force MLflow tracking on Databricks
    mlflow.set_tracking_uri(f"databricks://{profile}")
    client = MlflowClient()
    logger.debug("MLflow client initialized with Databricks tracking URI")

    # Experiments to delete
    experiment_paths = [config.experiment_name_basic, config.experiment_name_custom]
    logger.info("Experiments that will be deleted (marked as deleted):")
    for path in experiment_paths:
        logger.info(f"- {path}")

    if not yes:
        confirm = input("⚠️ Are you sure you want to delete these experiments? (y/N): ")
        if confirm.lower() != "y":
            logger.warning("❌ Aborted by user.")
            return

    # Delete each experiment if it exists
    for path in experiment_paths:
        exp = client.get_experiment_by_name(path)
        if exp:
            client.delete_experiment(exp.experiment_id)
            logger.success(f"Deleted (marked as deleted): {path} (id={exp.experiment_id})")
        else:
            logger.warning(f"Experiment not found: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete MLflow experiments (mark as deleted) on Databricks")
    parser.add_argument("--env-file", default="./.env", help="Path to .env file (default: ./.env)")
    parser.add_argument("--config-file", default="./project_config.yml", help="Path to project_config.yml")
    parser.add_argument("--environment", default="dev", choices=["dev", "acc", "prd"], help="Environment to use")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    main(
        env_file=args.env_file,
        config_file=args.config_file,
        environment=args.environment,
        yes=args.yes,
    )
