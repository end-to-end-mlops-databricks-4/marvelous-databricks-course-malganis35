"""Unit tests for src/mlops_course/data/databricks_utils.py."""

import logging

import pytest
from loguru import logger

from src.mlops_course.data.databricks_utils import (
    check_catalog_exists,
    ensure_schema,
    ensure_volume,
)


# --------------------------------------------------------------------------- #
#                   Bridge Loguru → Standard Logging for caplog               #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Forward Loguru logs to Python's logging system for pytest capture."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to the Python logging system."""
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                     Dummy Classes to Mock Databricks SDK                   #
# --------------------------------------------------------------------------- #
class DummyCatalogs:
    """Simulate Databricks Catalogs API."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize with failure simulation flag."""
        self.should_fail = should_fail
        self.get_called_with: str | None = None

    def get(self, catalog_name: str) -> None:
        """Simulate catalog retrieval, raising if configured to fail."""
        self.get_called_with = catalog_name
        if self.should_fail:
            raise Exception("catalog not found")


class DummySchemas:
    """Simulate Databricks Schemas API."""

    def __init__(self, get_fail: bool = False, create_fail: bool = False) -> None:
        """Initialize with failure flags for get/create operations."""
        self.get_fail = get_fail
        self.create_fail = create_fail
        self.get_called_with: tuple[str, str | None] | None = None
        self.created_with: tuple[str, str | None, str | None] | None = None

    def get(self, schema_name: str, catalog_name: str | None = None) -> None:
        """Simulate schema retrieval, raising if configured to fail."""
        self.get_called_with = (schema_name, catalog_name)
        if self.get_fail:
            raise Exception("schema not found")

    def create(self, name: str, catalog_name: str | None = None, comment: str | None = None) -> None:
        """Simulate schema creation, raising if configured to fail."""
        self.created_with = (name, catalog_name, comment)
        if self.create_fail:
            raise Exception("creation failed")


class DummyVolumes:
    """Simulate Databricks Volumes API."""

    def __init__(self, get_fail: bool = False, create_fail: bool = False) -> None:
        """Initialize with failure flags for get/create operations."""
        self.get_fail = get_fail
        self.create_fail = create_fail
        self.get_called_with: tuple[str, str | None, str | None] | None = None
        self.created_with: tuple[str, str | None, str | None, str | None, str | None] | None = None

    def get(self, volume_name: str, catalog_name: str | None = None, schema_name: str | None = None) -> None:
        """Simulate volume retrieval, raising if configured to fail."""
        self.get_called_with = (volume_name, catalog_name, schema_name)
        if self.get_fail:
            raise Exception("volume not found")

    def create(
        self,
        name: str,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Simulate volume creation, raising if configured to fail."""
        self.created_with = (name, catalog_name, schema_name, volume_type, comment)
        if self.create_fail:
            raise Exception("creation failed")


class DummyWorkspaceClient:
    """Simulate Databricks WorkspaceClient with nested API objects."""

    def __init__(
        self,
        catalog_fail: bool = False,
        schema_get_fail: bool = False,
        schema_create_fail: bool = False,
        volume_get_fail: bool = False,
        volume_create_fail: bool = False,
    ) -> None:
        """Initialize dummy client with failure configuration flags."""
        self.catalogs = DummyCatalogs(should_fail=catalog_fail)
        self.schemas = DummySchemas(get_fail=schema_get_fail, create_fail=schema_create_fail)
        self.volumes = DummyVolumes(get_fail=volume_get_fail, create_fail=volume_create_fail)


# --------------------------------------------------------------------------- #
#                                   TESTS                                    #
# --------------------------------------------------------------------------- #
def test_check_catalog_exists_success(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that catalog existence is correctly logged on success."""
    w = DummyWorkspaceClient()
    with caplog.at_level(logging.INFO):
        check_catalog_exists(w, "my_catalog")

    assert "Catalog my_catalog exists." in caplog.text
    assert w.catalogs.get_called_with == "my_catalog"


def test_check_catalog_exists_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that missing catalog triggers SystemExit and error log."""
    w = DummyWorkspaceClient(catalog_fail=True)
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        check_catalog_exists(w, "bad_catalog")

    assert "not found or inaccessible" in caplog.text
    assert w.catalogs.get_called_with == "bad_catalog"


@pytest.mark.parametrize("schema_exists", [True, False])
def test_ensure_schema_creation_or_existing(caplog: pytest.LogCaptureFixture, schema_exists: bool) -> None:
    """Test ensure_schema handles both existing and missing schemas."""
    w = DummyWorkspaceClient(schema_get_fail=not schema_exists)
    with caplog.at_level(logging.INFO):
        ensure_schema(w, "cat", "sch")

    if schema_exists:
        assert "already exists" in caplog.text
        assert w.schemas.get_called_with == ("sch", "cat")
        assert w.schemas.created_with is None
    else:
        assert "Creating schema sch in cat" in caplog.text
        assert w.schemas.created_with == ("sch", "cat", "ML Schema")


def test_ensure_schema_create_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Test ensure_schema logs creation failure correctly."""
    w = DummyWorkspaceClient(schema_get_fail=True, schema_create_fail=True)
    with caplog.at_level(logging.INFO):
        ensure_schema(w, "cat", "sch")

    assert "Failed to create schema sch" in caplog.text
    assert w.schemas.created_with == ("sch", "cat", "ML Schema")


@pytest.mark.parametrize("volume_exists", [True, False])
def test_ensure_volume_creation_or_existing(caplog: pytest.LogCaptureFixture, volume_exists: bool) -> None:
    """Test ensure_volume handles both existing and missing volumes."""
    w = DummyWorkspaceClient(volume_get_fail=not volume_exists)
    with caplog.at_level(logging.INFO):
        ensure_volume(w, "cat", "sch", "vol")

    if volume_exists:
        assert "already exists" in caplog.text
        assert w.volumes.get_called_with == ("vol", "cat", "sch")
        assert w.volumes.created_with is None
    else:
        assert "Creating volume vol in cat.sch" in caplog.text
        assert w.volumes.created_with is not None
        assert w.volumes.created_with[0:3] == ("vol", "cat", "sch")


def test_ensure_volume_create_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Test ensure_volume logs creation failure correctly."""
    w = DummyWorkspaceClient(volume_get_fail=True, volume_create_fail=True)
    with caplog.at_level(logging.INFO):
        ensure_volume(w, "cat", "sch", "vol")

    assert "Failed to create volume vol" in caplog.text
    assert w.volumes.created_with is not None
    assert w.volumes.created_with[0:3] == ("vol", "cat", "sch")
