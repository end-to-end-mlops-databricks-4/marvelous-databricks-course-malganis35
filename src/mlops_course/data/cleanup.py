from databricks.sdk import WorkspaceClient
from loguru import logger


def delete_volume(w: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str):
    """
    Deletes a volume if it exists in the given catalog and schema.
    
    Args:
        w: The Workspace Client.
        catalog_name: The name of the Databricks catalog.
        schema_name: The name of the schema the volume belongs to.
        volume_name: The name of the volume to delete.
    
    Returns:
        None
    """
    full_name = f"{catalog_name}.{schema_name}.{volume_name}"
    try:
        # VolumesAPI.delete does not support "force"
        w.volumes.delete(full_name)
        logger.info(f"Deleted volume: {full_name}")
    except Exception as e:
        logger.warning(f"Could not delete volume {full_name}: {e}")


def delete_schema(w: WorkspaceClient, catalog_name: str, schema_name: str):
    """Deletes a schema if it exists (must be empty or with force=True).
    Args:
        w: The Workspace Client.
        catalog_name: The name of the Databricks catalog.
        schema_name: The name of the schema to delete.

    Returns:
        None
    """
    full_name = f"{catalog_name}.{schema_name}"
    try:
        w.schemas.delete(full_name, force=True)
        logger.info(f"Deleted schema: {full_name}")
    except Exception as e:
        logger.warning(f"Could not delete schema {full_name}: {e}")
