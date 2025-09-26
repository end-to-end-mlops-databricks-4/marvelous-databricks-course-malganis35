"""Utility to load environment variables for Databricks and project configuration.

Usage:
    from mlops_course.utils.env_loader import load_environment

    load_environment(".env")  # or load_environment() to use default
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def load_environment(env_file: str | None = None) -> None:
    """Load environment variables from a .env file and set Databricks profile defaults.

    Args:
        env_file (str | None): Path to the .env file. Defaults to ".env" in project root.

    """
    # Determine env path
    env_path = Path(env_file or os.getenv("ENV_FILE", ".env"))

    # Load .env if exists
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"✅ Loaded environment from: {env_path.resolve()}")
    else:
        logger.warning(f"⚠️ .env file not found at {env_path.resolve()}")

    # Always override Databricks profile with PROFILE from .env
    if "PROFILE" in os.environ:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = os.getenv("PROFILE")

    # Log for verification
    logger.info(f"✅ Databricks profile set to: {os.getenv('DATABRICKS_CONFIG_PROFILE')}")
    logger.info(f"Databricks host: {os.getenv('DATABRICKS_HOST', '(not set)')}")
