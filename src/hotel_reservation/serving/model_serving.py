"""Model serving management module for Databricks.

This module provides a `ModelServing` class to automate the deployment and
management of model serving endpoints on Databricks. It integrates with MLflow
and the Databricks SDK to handle endpoint creation, update, monitoring, and
state management with retry and readiness checks.

Compatible with Databricks SDK >= 0.55.0.
"""

import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceConflict, ResourceDoesNotExist
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger


class ModelServing:
    """Manage Databricks model serving endpoints.

    This class provides methods to deploy, update, and monitor Databricks
    serving endpoints linked to registered MLflow models.

    Args:
        model_name (str): Fully qualified name of the model (e.g., `"catalog.schema.model"`).
        endpoint_name (str): Name of the Databricks serving endpoint.
        catalog_name (str, optional): Unity Catalog name for inference tables.
        schema_name (str, optional): Unity Schema name for inference tables.
        monitoring_table_suffix (str, optional): Prefix for inference tables. Defaults to "endpoint_logs".

    """

    def __init__(
        self,
        model_name: str,
        endpoint_name: str,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        monitoring_table_suffix: str = "endpoint_logs",
    ) -> None:
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.monitoring_table_suffix = f"{model_name}_{monitoring_table_suffix}"

    # -------------------------------------------------------------------------
    # Model Version Utilities
    # -------------------------------------------------------------------------
    def get_latest_model_version(self) -> str:
        """Retrieve the latest model version using the alias `"latest-model"`.

        Returns:
            str: The latest model version number.

        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"📦 Latest model version by alias 'latest-model': {latest_version}")
        return latest_version

    def get_latest_ready_version(self) -> str:
        """Find the latest model version with status `READY` in Unity Catalog.

        Returns:
            str: The latest `READY` version of the model.

        Raises:
            ValueError: If no `READY` version is found for the model.

        """
        client = mlflow.MlflowClient()
        logger.info(f"🔍 Searching for READY versions of model '{self.model_name}'...")

        ready_versions = [
            int(mv.version) for mv in client.search_model_versions(f"name='{self.model_name}'") if mv.status == "READY"
        ]

        if not ready_versions:
            raise ValueError(f"No READY version found for model '{self.model_name}'")

        latest_ready_version = str(max(ready_versions))
        logger.success(f"✅ Latest READY model version: {latest_ready_version}")
        return latest_ready_version

    # -------------------------------------------------------------------------
    # Endpoint State Management
    # -------------------------------------------------------------------------
    def is_updating(self) -> bool:
        """Check whether the endpoint configuration is currently being updated.

        Returns:
            bool: `True` if the endpoint is updating, `False` otherwise.

        """
        endpoint = self.workspace.serving_endpoints.get(self.endpoint_name)
        config_update = getattr(endpoint.state, "config_update", None)
        return bool(config_update and getattr(config_update, "state", None) == "IN_PROGRESS")

    def wait_until_not_updating(self, timeout: int = 300, check_interval: int = 10) -> None:
        """Wait until the endpoint finishes its current update process.

        Args:
            timeout (int, optional): Maximum wait time in seconds. Defaults to `300`.
            check_interval (int, optional): Interval between checks in seconds. Defaults to `10`.

        Raises:
            TimeoutError: If the endpoint remains updating after the timeout period.

        """
        start_time = time.time()
        logger.info(f"⏳ Waiting for endpoint '{self.endpoint_name}' to finish current update...")

        while time.time() - start_time < timeout:
            if not self.is_updating():
                logger.success(f"✅ Endpoint '{self.endpoint_name}' is no longer updating.")
                return
            logger.info("🔁 Endpoint still updating... waiting...")
            time.sleep(check_interval)

        raise TimeoutError(f"❌ Timeout: endpoint '{self.endpoint_name}' still updating after {timeout} seconds.")

    def wait_until_ready(self, timeout: int = 1200, check_interval: int = 10) -> None:
        """Wait until the Databricks serving endpoint reaches the READY state.

        Args:
        timeout (int, optional): Maximum wait time in seconds. Defaults to 600.
        check_interval (int, optional): Interval between status checks in seconds. Defaults to 10.

        Raises:
        RuntimeError: If the endpoint update fails.
        TimeoutError: If the endpoint does not become READY within the timeout.

        """
        start_time = time.time()
        logger.info(f"⏳ Waiting for endpoint '{self.endpoint_name}' to become READY...")

        try:
            endpoint = self.workspace.serving_endpoints.get(self.endpoint_name)
        except ResourceDoesNotExist:
            logger.warning(
                f"⚠️ Endpoint '{self.endpoint_name}' does not exist yet. Skipping wait — it will be created later."
            )
            return

        while time.time() - start_time < timeout:
            endpoint = self.workspace.serving_endpoints.get(self.endpoint_name)
            state = endpoint.state.ready
            state_str = state.value if hasattr(state, "value") else state
            config_update = getattr(endpoint.state, "config_update", None)

            logger.info(f"➡️  Current state: {state_str}")
            if config_update and hasattr(config_update, "state"):
                logger.info(f"   - Update details: {config_update.state}")

            if state_str == "READY":
                logger.success(f"✅ Endpoint '{self.endpoint_name}' is READY to serve requests!")
                return

            if state_str == "NOT_READY" and config_update and getattr(config_update, "state", "") == "UPDATE_FAILED":
                raise RuntimeError(f"❌ Deployment failed for endpoint '{self.endpoint_name}'")

            time.sleep(check_interval)

        raise TimeoutError(f"❌ Timeout: endpoint '{self.endpoint_name}' did not become READY after {timeout} seconds.")

    # -------------------------------------------------------------------------
    # Deployment / Update Logic
    # -------------------------------------------------------------------------
    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        environment_vars: dict | None = None,
        enable_inference_tables: bool = False,
        max_retries: int = 5,
        retry_interval: int = 20,
    ) -> None:
        """Deploy or update a Databricks model serving endpoint with optional AI Gateway inference tables.

        Automatically creates a new endpoint if it does not exist, or updates
        the configuration if it already exists. Includes retry logic to handle
        `ResourceConflict` errors during concurrent updates.

        Args:
            version (str, optional): Model version to deploy. Defaults to `"latest"`.
            workload_size (str, optional): Workload size (e.g., `"Small"`, `"Medium"`, `"Large"`). Defaults to `"Small"`.
            scale_to_zero (bool, optional): Whether to scale endpoint to 0 when idle. Defaults to `True`.
            environment_vars (dict, optional): Environment variables to inject into the serving environment. Defaults to `None`.
            enable_inference_tables (boolean, optional): Enable Inferance Table in AI Gateway. Defaults to `False`.
            max_retries (int, optional): Maximum number of retry attempts on conflict. Defaults to `5`.
            retry_interval (int, optional): Wait time in seconds between retries. Defaults to `20`.

        Raises:
            ResourceConflict: If the endpoint cannot be updated after all retries.

        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
                environment_vars=environment_vars or {},
            )
        ]

        # Optional AI Gateway inference table configuration
        ai_gateway_cfg = None
        if enable_inference_tables:
            if not (self.catalog_name and self.schema_name):
                raise ValueError("To enable inference tables, both catalog_name and schema_name must be provided.")
            ai_gateway_cfg = AiGatewayConfig(
                inference_table_config=AiGatewayInferenceTableConfig(
                    enabled=True,
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                    monitoring_table_suffix=self.monitoring_table_suffix,
                )
            )
            logger.info(
                f"🧠 Enabling AI Gateway inference tables: {self.catalog_name}.{self.schema_name}.{self.monitoring_table_suffix}_*"
            )

        # Create new endpoint
        if not endpoint_exists:
            logger.info(f"🚀 Creating new endpoint '{self.endpoint_name}'...")
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
                ai_gateway=ai_gateway_cfg,
            )
            logger.success(f"✅ Endpoint '{self.endpoint_name}' created with model version {entity_version}")
            return

        # Update existing endpoint with retry handling
        for attempt in range(1, max_retries + 1):
            try:
                if self.is_updating():
                    logger.warning(f"⚠️ Endpoint '{self.endpoint_name}' is updating — waiting before retry...")
                    self.wait_until_not_updating()

                logger.info(f"🔄 Attempt {attempt}/{max_retries}: updating endpoint '{self.endpoint_name}'...")
                self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)

                # If inference tables are enabled, ensure AI Gateway config is applied
                if ai_gateway_cfg:
                    self.workspace.serving_endpoints.put_ai_gateway(
                        name=self.endpoint_name,
                        inference_table_config=ai_gateway_cfg.inference_table_config,
                    )
                    logger.success(f"🧩 AI Gateway inference tables enabled for endpoint '{self.endpoint_name}'")

                logger.success(f"✅ Endpoint '{self.endpoint_name}' updated with model version {entity_version}")
                return

            except ResourceConflict as e:
                if attempt < max_retries:
                    logger.warning(f"⏳ Resource conflict detected — waiting {retry_interval}s before retry...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"❌ Failed after {max_retries} retries — endpoint still busy.")
                    raise e
