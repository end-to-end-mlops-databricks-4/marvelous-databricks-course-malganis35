import logging
import pytest
from loguru import logger
from src.mlops_course.data.cleanup import delete_volume, delete_schema


# 🔧 Connecte Loguru à logging standard pour que caplog puisse le capturer
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


class DummyVolumesAPI:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.deleted = None

    def delete(self, full_name):
        if self.should_fail:
            raise Exception("delete failed")
        self.deleted = full_name


class DummySchemasAPI:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.deleted = None

    def delete(self, full_name, force=False):
        if self.should_fail:
            raise Exception("delete failed")
        self.deleted = (full_name, force)


class DummyWorkspaceClient:
    def __init__(self, volume_fail=False, schema_fail=False):
        self.volumes = DummyVolumesAPI(volume_fail)
        self.schemas = DummySchemasAPI(schema_fail)


@pytest.mark.parametrize("should_fail", [False, True])
def test_delete_volume_logs_and_calls(monkeypatch, caplog, should_fail):
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
def test_delete_schema_logs_and_calls(monkeypatch, caplog, should_fail):
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
