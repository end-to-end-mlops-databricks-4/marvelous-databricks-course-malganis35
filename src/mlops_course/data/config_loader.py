import os

import yaml
from dotenv import dotenv_values, load_dotenv
from loguru import logger


def load_env(env_file: str = ".env") -> tuple[str | None, str | None, str | None]:
    """Load Databricks authentication details from a .env file.

    The function reads environment variables from the specified `.env` file using
    `python-dotenv`. It supports two authentication modes:
    - **Profile mode:** when a `PROFILE` variable is defined.
    - **Token mode:** when both `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are defined.

    Args:
        env_file (str): Path to the `.env` file. Defaults to ".env".

    Returns:
        tuple[str | None, str | None, str | None]:
        A tuple containing `(DATABRICKS_HOST, DATABRICKS_TOKEN, PROFILE)`, where
        unused values are returned as `None`.

    Raises:
        OSError: If neither `PROFILE` nor both `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
            are defined in the given `.env` file.

    """
    # Résolution du chemin absolu
    env_path = os.path.abspath(env_file)
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found at {env_path}")

    # Charge les variables sans dépendre du cwd
    load_dotenv(dotenv_path=env_path, override=True)

    # Lecture directe du fichier (pour vérifier le contenu réel)
    env_vars = dotenv_values(env_path)
    host = env_vars.get("DATABRICKS_HOST") or os.getenv("DATABRICKS_HOST")
    token = env_vars.get("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_TOKEN")
    profile = env_vars.get("PROFILE") or os.getenv("PROFILE")

    # Logging debug utile
    logger.debug(f"Loaded env from {env_path}: host={host}, token={'***' if token else None}, profile={profile}")

    # Logique de priorité : PROFILE > TOKEN
    if profile:
        return None, None, profile
    if host and token:
        return host, token, None

    raise OSError(f"Missing PROFILE or DATABRICKS_HOST/TOKEN in {env_path}")
    return None, None, None  # mode auto-detect


def load_project_config(path: str, env: str) -> tuple[dict, dict]:
    """Load environment variables from a .env file.

    Specifically extracts the DATABRICKS_HOST and DATABRICKS_TOKEN for Databricks authentication.

    It extracts the environment-specific configuration (`env_config`) and the global configuration (`global_config`).

    Args:
        path (str): The path to the YAML project configuration file.
        env (str): The environment for which the configuration is being loaded.

    Returns:
        tuple: A tuple containing the environment-specific configuration (`env_config`) and the global configuration (`global_config`).

    Raises:
        ValueError: If the specified environment is not found in the configuration file.

    """
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if env not in config:
        raise ValueError(f"Environment {env} not found in {path}")

    env_config = config[env]
    global_config = {k: v for k, v in config.items() if k not in ["dev", "acc", "prd"]}

    return env_config, global_config
