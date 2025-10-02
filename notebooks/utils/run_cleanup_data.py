# run the script
# On one environment: uv run scripts/run_cleanup_data.py --env dev --env-file .env --config project_config.yml
# On all environment: uv run scripts/run_cleanup_data.py --env all --env-file .env --config project_config.yml
import argparse

from databricks.sdk import WorkspaceClient
from loguru import logger

from hotel_reservation.data.cleanup import delete_schema, delete_volume
from hotel_reservation.data.config_loader import load_env, load_project_config


def main() -> None:
    """Databricks cleanup script for schemas and volumes.

    This script cleans up Databricks schemas and volumes based on the provided environment and configuration.

    Args:
        --env (str): Target environment (or 'all' for all environments). Defaults to "dev".
        --env-file (str): Path to .env file with credentials. Defaults to ".env".
        --config (str): YAML configuration file. Defaults to "project_config.yml".

    Returns:
        None

    Notes:
        The script loads credentials from the provided .env file, initializes a Databricks client, and then iterates over the selected environments.
        For each environment, it loads the project configuration, retrieves the catalog, schema, and volume names, and then deletes the volume and schema.

    """
    parser = argparse.ArgumentParser(description="Databricks cleanup (schemas & volumes)")
    parser.add_argument(
        "--env",
        default="dev",
        choices=["dev", "acc", "prd", "all"],
        help="Target environment (or 'all' for all environments)",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file with credentials")
    parser.add_argument("--config", default="project_config.yml", help="YAML configuration file")
    args = parser.parse_args()

    # Load host/token
    host, token, profile = load_env(args.env_file)
    logger.info(f"Loaded credentials from {args.env_file}")

    # Prepare environments to process
    envs = ["dev", "acc", "prd"] if args.env == "all" else [args.env]
    logger.info(f"Environments selected for cleanup: {envs}")

    # Initialize Databricks client
    if profile:
        w = WorkspaceClient(profile=profile)
        logger.debug(f"Databricks WorkspaceClient initialized with profile {profile}")
    elif host and token:
        w = WorkspaceClient(host=host, token=token)
        logger.debug("Databricks WorkspaceClient initialized with host/token")
    else:
        w = WorkspaceClient()  # fallback auto-detection
        logger.debug("Databricks WorkspaceClient initialized with auto-detection")

    for env in envs:
        env_config, _ = load_project_config(args.config, env)
        catalog = env_config["catalog_name"]
        schema = env_config["schema_name"]
        volume = env_config["volume_name"]

        logger.info(f"=== Cleaning {env} ({catalog}.{schema}.{volume}) ===")

        # 1. Delete volume
        delete_volume(w, catalog, schema, volume)
        logger.success(f"Volume {catalog}.{schema}.{volume} deleted (if existed)")

        # 2. Delete schema
        delete_schema(w, catalog, schema)
        logger.success(f"Schema {catalog}.{schema} deleted (if existed)")

    logger.success("Cleanup completed.")

    return None


if __name__ == "__main__":
    main()
