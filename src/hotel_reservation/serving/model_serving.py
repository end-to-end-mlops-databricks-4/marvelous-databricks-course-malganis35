"""Modele serving module."""

import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger


class ModelServing:
    """Manages model serving in Databricks.

    This class provides functionality to deploy and update model serving endpoints.
    """

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        environment_vars: dict | None = None,
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks.

        :param version: Version of the model to deploy, defaults to "latest"
        :param workload_size: Workload size (number of concurrent requests), defaults to "Small"
        :param scale_to_zero: If True, endpoint scales to 0 when unused, defaults to True
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

        if not endpoint_exists:
            logger.info(f"🚀 Creating new endpoint '{self.endpoint_name}'...")
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
            logger.success(f"✅ Endpoint '{self.endpoint_name}' created with model version {entity_version}")
        else:
            logger.info(f"🔄 Endpoint '{self.endpoint_name}' already exists — updating configuration...")
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
            logger.success(f"✅ Endpoint '{self.endpoint_name}' updated with model version {entity_version}")

    def wait_until_ready(self, timeout: int = 600, check_interval: int = 10) -> None:
        """Wait for the Databricks serving endpoint to reach READY state."""
        start_time = time.time()
        logger.info(f"⏳ Waiting for endpoint '{self.endpoint_name}' to become READY...")

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
