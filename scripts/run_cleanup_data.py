# run the script
# On one environment: uv run scripts/run_cleanup_data.py --env dev --env-file .env --config project_config.yml
# On all environment: uv run scripts/run_cleanup_data.py --env all --env-file .env --config project_config.yml
import argparse
from databricks.sdk import WorkspaceClient
from mlops_course.data.config_loader import load_env, load_project_config
from mlops_course.data.cleanup import delete_volume, delete_schema
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Databricks cleanup (schemas & volumes)")
    parser.add_argument(
        "--env", default="dev", choices=["dev", "acc", "prd", "all"],
        help="Target environment (or 'all' for all environments)"
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file with credentials")
    parser.add_argument("--config", default="project_config.yml", help="YAML configuration file")
    args = parser.parse_args()

    # Load host/token
    host, token = load_env(args.env_file)
    logger.info(f"Loaded credentials from {args.env_file}")

    # Prepare environments to process
    envs = ["dev", "acc", "prd"] if args.env == "all" else [args.env]
    logger.info(f"Environments selected for cleanup: {envs}")

    # Initialize Databricks client
    w = WorkspaceClient(host=host, token=token)
    logger.debug("Databricks WorkspaceClient initialized")

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


if __name__ == "__main__":
    main()
