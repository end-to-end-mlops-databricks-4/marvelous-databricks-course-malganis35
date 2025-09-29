import os

from databricks.sdk import WorkspaceClient
from loguru import logger

from .databricks_utils import check_catalog_exists, ensure_schema, ensure_volume


def load_files_from_source(config: dict) -> list[str]:
    """Load files from a specified source configuration.

    Supports loading from a local directory or a Kaggle dataset, based on the
    `source_type` key in the configuration dictionary.

    Args:
        config (dict): A configuration dictionary that specifies:
            - **source_type** (str): Either `"local"` or `"kaggle"`.
            - **files** (list[str]): List of filenames to load.
            - **local_path** (str, optional): Base path for local files (if `source_type` is `"local"`).
            - **kaggle_dataset** (str, optional): Kaggle dataset identifier (if `source_type` is `"kaggle"`).

    Returns:
        list[str]: A list of file paths corresponding to the loaded files.

    Raises:
        ValueError: If `source_type` is not `"local"` or `"kaggle"`.
        FileNotFoundError: If any specified file does not exist.

    """
    source_type = config["source_type"]
    files = config["files"]

    if source_type == "local":
        base_path = config["local_path"]
        file_paths = [os.path.join(base_path, f) for f in files]
    elif source_type == "kaggle":
        import kagglehub

        dataset = config["kaggle_dataset"]
        path = kagglehub.dataset_download(dataset)
        file_paths = [os.path.join(path, f) for f in files]
    else:
        raise ValueError("source_type must be 'local' or 'kaggle'")

    for f in file_paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    return file_paths


def upload_files(host: str, token: str, env_config: dict, files: list, profile: str = None) -> None:
    """Upload files into the specified Databricks Volume.

    Args:
        host (str): Databricks host URL
        token (str): Databricks API token
        env_config (dict): Environment configuration dictionary containing 'catalog_name', 'schema_name', and 'volume_name' keys.
        files (list): List of local file paths to upload.
        profile (str): Databricks profile

    Returns:
        uploaded (list): List of uploaded file paths in Databricks DBFS.

    """
    if profile:
        w = WorkspaceClient(profile=profile)
    elif host and token:
        w = WorkspaceClient(host=host, token=token)
    else:
        w = WorkspaceClient()  # fallback : auto-detect (CLI, cloud login, etc.)

    catalog = env_config["catalog_name"]
    schema = env_config["schema_name"]
    volume = env_config["volume_name"]

    # Ensure catalog, schema, and volume exist
    check_catalog_exists(w, catalog)
    ensure_schema(w, catalog, schema)
    ensure_volume(w, catalog, schema, volume)

    uploaded = []
    for local_file in files:
        filename = os.path.basename(local_file)
        target_path = f"dbfs:/Volumes/{catalog}/{schema}/{volume}/{filename}"

        logger.info(f"Uploading {local_file} -> {target_path}")
        with open(local_file, "rb") as f:
            w.dbfs.upload(target_path, f, overwrite=True)

        uploaded.append(target_path)

    return uploaded
