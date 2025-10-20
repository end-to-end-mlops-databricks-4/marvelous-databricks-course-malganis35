from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import ResourceConflict

from hotel_reservation.serving.model_serving import ModelServing


@pytest.fixture
def mock_workspace() -> MagicMock:
    """Mock the Databricks WorkspaceClient and its serving_endpoints."""
    mock_ws = MagicMock()
    mock_ws.serving_endpoints.get = MagicMock()
    mock_ws.serving_endpoints.list = MagicMock()
    mock_ws.serving_endpoints.create = MagicMock()
    mock_ws.serving_endpoints.update_config = MagicMock()
    return mock_ws


@pytest.fixture
def model_serving(mock_workspace: MagicMock) -> ModelServing:
    """Return a ModelServing instance using a mocked Databricks workspace."""
    with patch("hotel_reservation.serving.model_serving.WorkspaceClient", return_value=mock_workspace):
        return ModelServing(model_name="catalog.schema.model", endpoint_name="test-endpoint")


# -----------------------------------------------------------------------------
# Model version utilities
# -----------------------------------------------------------------------------
@patch("hotel_reservation.serving.model_serving.mlflow.MlflowClient")
def test_get_latest_model_version(mock_mlflow_client: MagicMock, model_serving: ModelServing) -> None:
    """Test that get_latest_model_version retrieves the latest model version correctly."""
    mock_client = mock_mlflow_client.return_value
    mock_client.get_model_version_by_alias.return_value.version = "42"

    version = model_serving.get_latest_model_version()

    assert version == "42"
    mock_client.get_model_version_by_alias.assert_called_once_with("catalog.schema.model", alias="latest-model")


@patch("hotel_reservation.serving.model_serving.mlflow.MlflowClient")
def test_get_latest_ready_version_found(mock_mlflow_client: MagicMock, model_serving: ModelServing) -> None:
    """Test that get_latest_ready_version returns the latest READY version."""
    mock_client = mock_mlflow_client.return_value
    mock_client.search_model_versions.return_value = [
        MagicMock(version="1", status="READY"),
        MagicMock(version="3", status="READY"),
        MagicMock(version="2", status="PENDING"),
    ]

    version = model_serving.get_latest_ready_version()
    assert version == "3"


@patch("hotel_reservation.serving.model_serving.mlflow.MlflowClient")
def test_get_latest_ready_version_not_found(mock_mlflow_client: MagicMock, model_serving: ModelServing) -> None:
    """Test that get_latest_ready_version raises ValueError when no READY version is found."""
    mock_client = mock_mlflow_client.return_value
    mock_client.search_model_versions.return_value = [
        MagicMock(version="1", status="PENDING"),
        MagicMock(version="2", status="FAILED"),
    ]

    with pytest.raises(ValueError, match="No READY version found"):
        model_serving.get_latest_ready_version()


# -----------------------------------------------------------------------------
# Endpoint state management
# -----------------------------------------------------------------------------
def test_is_updating_true(model_serving: ModelServing, mock_workspace: MagicMock) -> None:
    """Test that is_updating returns True when update state is IN_PROGRESS."""
    mock_endpoint = MagicMock()
    mock_endpoint.state.config_update.state = "IN_PROGRESS"
    mock_workspace.serving_endpoints.get.return_value = mock_endpoint

    assert model_serving.is_updating() is True


def test_is_updating_false(model_serving: ModelServing, mock_workspace: MagicMock) -> None:
    """Test that is_updating returns False when update state is COMPLETED."""
    mock_endpoint = MagicMock()
    mock_endpoint.state.config_update.state = "COMPLETED"
    mock_workspace.serving_endpoints.get.return_value = mock_endpoint

    assert model_serving.is_updating() is False


@patch("time.sleep", return_value=None)
def test_wait_until_not_updating_completes(mock_sleep: MagicMock, model_serving: ModelServing) -> None:
    """Test that wait_until_not_updating completes successfully."""
    model_serving.is_updating = MagicMock(side_effect=[True, False])
    model_serving.wait_until_not_updating(timeout=5, check_interval=1)
    assert model_serving.is_updating.call_count == 2


@patch("time.sleep", return_value=None)
def test_wait_until_not_updating_timeout(mock_sleep: MagicMock, model_serving: ModelServing) -> None:
    """Test that wait_until_not_updating raises TimeoutError when stuck updating."""
    model_serving.is_updating = MagicMock(return_value=True)
    with pytest.raises(TimeoutError):
        model_serving.wait_until_not_updating(timeout=1, check_interval=1)


@patch("time.sleep", return_value=None)
def test_wait_until_ready_success(
    mock_sleep: MagicMock, model_serving: ModelServing, mock_workspace: MagicMock
) -> None:
    """Test that wait_until_ready returns when endpoint is READY."""
    mock_endpoint = MagicMock()
    mock_endpoint.state.ready = "READY"
    mock_workspace.serving_endpoints.get.return_value = mock_endpoint

    model_serving.wait_until_ready(timeout=1, check_interval=1)


@patch("time.sleep", return_value=None)
def test_wait_until_ready_fails_with_update_failed(
    mock_sleep: MagicMock, model_serving: ModelServing, mock_workspace: MagicMock
) -> None:
    """Test that wait_until_ready raises RuntimeError when deployment fails."""
    mock_endpoint = MagicMock()
    mock_endpoint.state.ready = "NOT_READY"
    mock_endpoint.state.config_update.state = "UPDATE_FAILED"
    mock_workspace.serving_endpoints.get.return_value = mock_endpoint

    with pytest.raises(RuntimeError, match="Deployment failed"):
        model_serving.wait_until_ready(timeout=1, check_interval=1)


# -----------------------------------------------------------------------------
# Deployment logic
# -----------------------------------------------------------------------------
def test_deploy_creates_new_endpoint(model_serving: ModelServing, mock_workspace: MagicMock) -> None:
    """Test that a new serving endpoint is created if none exists."""
    mock_workspace.serving_endpoints.list.return_value = []
    model_serving.get_latest_model_version = MagicMock(return_value="1")

    model_serving.deploy_or_update_serving_endpoint()

    mock_workspace.serving_endpoints.create.assert_called_once()
    mock_workspace.serving_endpoints.update_config.assert_not_called()


def test_deploy_updates_existing_endpoint(model_serving: ModelServing, mock_workspace: MagicMock) -> None:
    """Test that deploy_or_update_serving_endpoint updates an existing endpoint."""
    mock_item = MagicMock()
    mock_item.name = "test-endpoint"
    mock_workspace.serving_endpoints.list.return_value = [mock_item]

    model_serving.get_latest_model_version = MagicMock(return_value="1")
    model_serving.is_updating = MagicMock(return_value=False)

    model_serving.deploy_or_update_serving_endpoint()

    mock_workspace.serving_endpoints.update_config.assert_called_once()
    mock_workspace.serving_endpoints.create.assert_not_called()


@patch("time.sleep", return_value=None)
def test_deploy_retries_on_resource_conflict(
    mock_sleep: MagicMock, model_serving: ModelServing, mock_workspace: MagicMock
) -> None:
    """Test that deploy_or_update_serving_endpoint retries when ResourceConflict occurs."""
    mock_item = MagicMock()
    mock_item.name = "test-endpoint"
    mock_workspace.serving_endpoints.list.return_value = [mock_item]

    model_serving.get_latest_model_version = MagicMock(return_value="1")
    model_serving.is_updating = MagicMock(return_value=False)

    mock_workspace.serving_endpoints.update_config.side_effect = [
        ResourceConflict("conflict"),
        None,
    ]

    model_serving.deploy_or_update_serving_endpoint(max_retries=2, retry_interval=0)

    assert mock_workspace.serving_endpoints.update_config.call_count == 2


@patch("time.sleep", return_value=None)
def test_deploy_fails_after_max_retries(
    mock_sleep: MagicMock, model_serving: ModelServing, mock_workspace: MagicMock
) -> None:
    """Test that deploy_or_update_serving_endpoint raises ResourceConflict after max retries."""
    mock_item = MagicMock()
    mock_item.name = "test-endpoint"
    mock_workspace.serving_endpoints.list.return_value = [mock_item]

    model_serving.get_latest_model_version = MagicMock(return_value="1")
    model_serving.is_updating = MagicMock(return_value=False)

    mock_workspace.serving_endpoints.update_config.side_effect = ResourceConflict("still conflict")

    with pytest.raises(ResourceConflict):
        model_serving.deploy_or_update_serving_endpoint(max_retries=2, retry_interval=0)
