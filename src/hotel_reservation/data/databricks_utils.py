from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from loguru import logger


def check_catalog_exists(w: WorkspaceClient, catalog_name: str) -> None:
    """Check if a Databricks catalog exists and is accessible.

    Args:
        w (Databricks client instance): The Databricks client instance.
        catalog_name (str): The name of the catalog to check.

    Raises:
        SystemExit(1): If the catalog does not exist or is inaccessible.

    """
    try:
        w.catalogs.get(catalog_name)
        logger.info(f"Catalog {catalog_name} exists.")
    except Exception as e:
        logger.error(f"Catalog {catalog_name} not found or inaccessible: {e}")
        raise SystemExit(1) from e
    return None


def ensure_schema(w: WorkspaceClient, catalog_name: str, schema_name: str) -> None:
    """Check if a schema exists in the specified Databricks catalog. If it doesn't exist, the function creates it.

    Args:
        w: A Databricks client object.
        catalog_name: The name of the Databricks catalog.
        schema_name: The name of the schema to create or check.

    Returns:
        None

    """
    try:
        w.schemas.get(schema_name, catalog_name=catalog_name)
        logger.info(f"Schema {schema_name} already exists in {catalog_name}.")
    except Exception:
        logger.info(f"Creating schema {schema_name} in {catalog_name}...")
        try:
            w.schemas.create(name=schema_name, catalog_name=catalog_name, comment="ML Schema")
            logger.info(f"Schema {schema_name} created in {catalog_name}.")
        except Exception as e:
            logger.error(f"Failed to create schema {schema_name} in {catalog_name}: {e}")
    return None


def ensure_volume(w: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str) -> None:
    """Ensure the existence of a Databricks volume in the specified catalog.schema.

    Args:
        w (Databricks client): The Databricks client instance.
        catalog_name (str): Name of the Databricks catalog.
        schema_name (str): Name of the Databricks schema.
        volume_name (str): Name of the volume to ensure exists.

    Returns:
        None

    Notes:
    This function will create the volume if it does not exist in the specified catalog.schema.

    """
    try:
        w.volumes.get(volume_name, catalog_name=catalog_name, schema_name=schema_name)
        logger.info(f"Volume {volume_name} already exists in {catalog_name}.{schema_name}.")
    except Exception:
        logger.info(f"Creating volume {volume_name} in {catalog_name}.{schema_name}...")
        try:
            w.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_type=VolumeType.MANAGED,
                comment="Data volume",
            )
            logger.info(f"Volume {volume_name} created in {catalog_name}.{schema_name}.")
        except Exception as e:
            logger.error(f"Failed to create volume {volume_name} in {catalog_name}.{schema_name}: {e}")
    return None
