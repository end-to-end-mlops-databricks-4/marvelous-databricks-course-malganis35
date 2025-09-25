import os
import yaml
from dotenv import load_dotenv


def load_env(env_file: str = ".env"):
    """
    Loads environment variables from a .env file, specifically extracting the 
    DATABRICKS_HOST and DATABRICKS_TOKEN for Databricks authentication.
    
    Args:
        env_file (str): Path to the .env file. Defaults to ".env".
    
    Returns:
        tuple: A tuple containing the DATABRICKS_HOST and DATABRICKS_TOKEN values.
        
    Raises:
        EnvironmentError: If DATABRICKS_HOST or DATABRICKS_TOKEN are not defined in the env_file.
    """
    load_dotenv(dotenv_path=env_file)

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    profile = os.getenv("PROFILE")

    if profile:
        return None, None, profile  # mode profil
    if host and token:
        return host, token, None    # mode token
    raise EnvironmentError(
        f"Ni PROFILE ni DATABRICKS_HOST/TOKEN définis dans {env_file}"
    )


def load_project_config(path: str, env: str):
    """
    Load and parse the YAML project configuration from the specified path.

    It extracts the environment-specific configuration (`env_config`) and the global configuration (`global_config`).

    Args:
        path (str): The path to the YAML project configuration file.
        env (str): The environment for which the configuration is being loaded.

    Returns:
        tuple: A tuple containing the environment-specific configuration (`env_config`) and the global configuration (`global_config`).

    Raises:
        ValueError: If the specified environment is not found in the configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if env not in config:
        raise ValueError(f"Environment {env} not found in {path}")

    env_config = config[env]
    global_config = {k: v for k, v in config.items() if k not in ["dev", "acc", "prd"]}

    return env_config, global_config
