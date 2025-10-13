"""Unit tests for src/hotel_reservation/data/cleanup.py."""

import logging
from typing import Any

import pytest
from loguru import logger

from src.hotel_reservation.data.cleanup import delete_schema, delete_volume


# --------------------------------------------------------------------------- #
#                     Bridge Loguru → Standard Logging                        #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Forward Loguru logs to the standard logging system for pytest capture."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through the standard logging system."""
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                         Dummy API Simulations                               #
# --------------------------------------------------------------------------- #
class DummyVolumesAPI:
    """Simulated Databricks Volumes API for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize with optional failure mode."""
        self.should_fail = should_fail
        self.deleted: Any = None

    def delete(self, full_name: str) -> None:
        """Simulate deleting a volume or raising an exception."""
        if self.should_fail:
            raise Exception("delete failed")
        self.deleted = full_name


class DummySchemasAPI:
    """Simulated Databricks Schemas API for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize with optional failure mode."""
        self.should_fail = should_fail
        self.deleted: Any = None

    def delete(self, full_name: str, force: bool = False) -> None:
        """Simulate deleting a schema or raising an exception."""
        if self.should_fail:
            raise Exception("delete failed")
        self.deleted = (full_name, force)


class DummyWorkspaceClient:
    """Simulated WorkspaceClient with schema and volume sub-APIs."""

    def __init__(self, volume_fail: bool = False, schema_fail: bool = False) -> None:
        """Initialize dummy client with configurable failure behavior."""
        self.volumes = DummyVolumesAPI(volume_fail)
        self.schemas = DummySchemasAPI(schema_fail)


# --------------------------------------------------------------------------- #
#                                   TESTS                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("should_fail", [False, True])
def test_delete_volume_logs_and_calls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    should_fail: bool,
) -> None:
    """Verify delete_volume logs success or failure and calls API correctly."""
    w = DummyWorkspaceClient(volume_fail=should_fail)
    catalog, schema, volume = "cat", "sch", "vol"
    full_name = f"{catalog}.{schema}.{volume}"

    with caplog.at_level(logging.INFO):
        delete_volume(w, catalog, schema, volume)

    if should_fail:
        assert any("Could not delete volume" in msg for msg in caplog.messages)
        assert w.volumes.deleted is None
    else:
        assert any(f"Deleted volume: {full_name}" in msg for msg in caplog.messages)
        assert w.volumes.deleted == full_name


@pytest.mark.parametrize("should_fail", [False, True])
def test_delete_schema_logs_and_calls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    should_fail: bool,
) -> None:
    """Verify delete_schema logs success or failure and calls API correctly."""
    w = DummyWorkspaceClient(schema_fail=should_fail)
    catalog, schema = "cat", "sch"
    full_name = f"{catalog}.{schema}"

    with caplog.at_level(logging.INFO):
        delete_schema(w, catalog, schema)

    if should_fail:
        assert any("Could not delete schema" in msg for msg in caplog.messages)
        assert w.schemas.deleted is None
    else:
        assert any(f"Deleted schema: {full_name}" in msg for msg in caplog.messages)
        assert w.schemas.deleted == (full_name, True)
