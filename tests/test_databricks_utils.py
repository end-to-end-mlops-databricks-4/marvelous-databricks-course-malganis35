import logging
import pytest
from loguru import logger
from src.mlops_course.data.databricks_utils import (
    check_catalog_exists,
    ensure_schema,
    ensure_volume,
)


# ---- Setup pour capturer les logs loguru via caplog ----
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# ---- Dummy Classes to simulate Databricks SDK behavior ----
class DummyCatalogs:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.get_called_with = None

    def get(self, catalog_name):
        self.get_called_with = catalog_name
        if self.should_fail:
            raise Exception("catalog not found")


class DummySchemas:
    def __init__(self, get_fail=False, create_fail=False):
        self.get_fail = get_fail
        self.create_fail = create_fail
        self.get_called_with = None
        self.created_with = None

    def get(self, schema_name, catalog_name=None):
        self.get_called_with = (schema_name, catalog_name)
        if self.get_fail:
            raise Exception("schema not found")

    def create(self, name, catalog_name=None, comment=None):
        self.created_with = (name, catalog_name, comment)
        if self.create_fail:
            raise Exception("creation failed")


class DummyVolumes:
    def __init__(self, get_fail=False, create_fail=False):
        self.get_fail = get_fail
        self.create_fail = create_fail
        self.get_called_with = None
        self.created_with = None

    def get(self, volume_name, catalog_name=None, schema_name=None):
        self.get_called_with = (volume_name, catalog_name, schema_name)
        if self.get_fail:
            raise Exception("volume not found")

    def create(self, name, catalog_name=None, schema_name=None, volume_type=None, comment=None):
        self.created_with = (name, catalog_name, schema_name, volume_type, comment)
        if self.create_fail:
            raise Exception("creation failed")


class DummyWorkspaceClient:
    def __init__(self, catalog_fail=False, schema_get_fail=False, schema_create_fail=False,
                 volume_get_fail=False, volume_create_fail=False):
        self.catalogs = DummyCatalogs(should_fail=catalog_fail)
        self.schemas = DummySchemas(get_fail=schema_get_fail, create_fail=schema_create_fail)
        self.volumes = DummyVolumes(get_fail=volume_get_fail, create_fail=volume_create_fail)


# ---- TESTS ----

def test_check_catalog_exists_success(caplog):
    w = DummyWorkspaceClient()
    with caplog.at_level(logging.INFO):
        check_catalog_exists(w, "my_catalog")

    assert "Catalog my_catalog exists." in caplog.text
    assert w.catalogs.get_called_with == "my_catalog"


def test_check_catalog_exists_failure(caplog):
    w = DummyWorkspaceClient(catalog_fail=True)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            check_catalog_exists(w, "bad_catalog")

    assert "not found or inaccessible" in caplog.text
    assert w.catalogs.get_called_with == "bad_catalog"


@pytest.mark.parametrize("schema_exists", [True, False])
def test_ensure_schema_creation_or_existing(caplog, schema_exists):
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


def test_ensure_schema_create_failure(caplog):
    w = DummyWorkspaceClient(schema_get_fail=True, schema_create_fail=True)
    with caplog.at_level(logging.INFO):
        ensure_schema(w, "cat", "sch")

    assert "Failed to create schema sch" in caplog.text
    assert w.schemas.created_with == ("sch", "cat", "ML Schema")


@pytest.mark.parametrize("volume_exists", [True, False])
def test_ensure_volume_creation_or_existing(caplog, volume_exists):
    w = DummyWorkspaceClient(volume_get_fail=not volume_exists)
    with caplog.at_level(logging.INFO):
        ensure_volume(w, "cat", "sch", "vol")

    if volume_exists:
        assert "already exists" in caplog.text
        assert w.volumes.get_called_with == ("vol", "cat", "sch")
        assert w.volumes.created_with is None
    else:
        assert "Creating volume vol in cat.sch" in caplog.text
        assert w.volumes.created_with[0:3] == ("vol", "cat", "sch")


def test_ensure_volume_create_failure(caplog):
    w = DummyWorkspaceClient(volume_get_fail=True, volume_create_fail=True)
    with caplog.at_level(logging.INFO):
        ensure_volume(w, "cat", "sch", "vol")

    assert "Failed to create volume vol" in caplog.text
    assert w.volumes.created_with[0:3] == ("vol", "cat", "sch")
