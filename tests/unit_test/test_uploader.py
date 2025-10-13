"""Unit tests for src/hotel_reservation/data/uploader.py."""

import io
import logging
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

from src.hotel_reservation.data import uploader


# --------------------------------------------------------------------------- #
#                               Loguru integration                            #
# --------------------------------------------------------------------------- #
class PropagateHandler(logging.Handler):
    """Redirect Loguru logs into pytest's caplog."""

    def emit(self, record: logging.LogRecord) -> None:
        """Propagate Loguru log record to standard logging."""
        logging.getLogger(record.name).handle(record)


# Attach the propagation handler
logger.add(PropagateHandler(), format="{message}")


# --------------------------------------------------------------------------- #
#                        Dummy Databricks client simulation                    #
# --------------------------------------------------------------------------- #
class DummyDBFS:
    """Simulated Databricks DBFS client."""

    def __init__(self) -> None:
        """Initialize an upload call tracker."""
        self.upload_calls: list[tuple[str, bytes, bool]] = []

    def upload(self, path: str, file_obj: io.BufferedReader, overwrite: bool = False) -> None:
        """Record file upload content and parameters."""
        content = file_obj.read()
        self.upload_calls.append((path, content, overwrite))


class DummyWorkspaceClient:
    """Simulated WorkspaceClient with DBFS and sub-APIs."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the dummy WorkspaceClient."""
        self.dbfs = DummyDBFS()


def dummy_check_catalog_exists(w: DummyWorkspaceClient, catalog_name: str) -> None:
    """Simulate catalog existence check."""
    logger.info(f"[dummy] Checked catalog {catalog_name}")


def dummy_ensure_schema(w: DummyWorkspaceClient, catalog_name: str, schema_name: str) -> None:
    """Simulate ensuring schema exists."""
    logger.info(f"[dummy] Ensured schema {schema_name} in {catalog_name}")


def dummy_ensure_volume(w: DummyWorkspaceClient, catalog_name: str, schema_name: str, volume_name: str) -> None:
    """Simulate ensuring volume exists."""
    logger.info(f"[dummy] Ensured volume {volume_name} in {catalog_name}.{schema_name}")


# --------------------------------------------------------------------------- #
#                                Pytest fixtures                               #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_databricks_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch Databricks-related functions to avoid real API calls."""
    monkeypatch.setattr(uploader, "WorkspaceClient", DummyWorkspaceClient)
    monkeypatch.setattr(uploader, "check_catalog_exists", dummy_check_catalog_exists)
    monkeypatch.setattr(uploader, "ensure_schema", dummy_ensure_schema)
    monkeypatch.setattr(uploader, "ensure_volume", dummy_ensure_volume)
    yield


# --------------------------------------------------------------------------- #
#                          Tests: load_files_from_source                       #
# --------------------------------------------------------------------------- #
def test_load_files_from_source_local_success(tmp_path: Path) -> None:
    """Ensure local files are correctly loaded when they exist."""
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("x")
    f2.write_text("y")

    config = {
        "source_type": "local",
        "files": ["a.csv", "b.csv"],
        "local_path": str(tmp_path),
    }

    result = uploader.load_files_from_source(config)
    assert all(os.path.exists(p) for p in result)
    assert all(p.endswith(".csv") for p in result)


def test_load_files_from_source_local_missing(tmp_path: Path) -> None:
    """Raise FileNotFoundError if a listed local file does not exist."""
    config = {
        "source_type": "local",
        "files": ["missing.csv"],
        "local_path": str(tmp_path),
    }
    with pytest.raises(FileNotFoundError):
        uploader.load_files_from_source(config)


def test_load_files_from_source_invalid_type() -> None:
    """Raise ValueError for unsupported source_type."""
    config = {"source_type": "unknown", "files": []}
    with pytest.raises(ValueError):
        uploader.load_files_from_source(config)


def test_load_files_from_source_kaggle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Simulate Kaggle dataset download using monkeypatch."""
    fake_dataset_path = tmp_path / "dataset"
    fake_dataset_path.mkdir()
    (fake_dataset_path / "data.csv").write_text("dummy")

    class DummyKaggleHub:
        """Dummy kagglehub simulation."""

        def dataset_download(self, dataset: str) -> str:
            """Return fake dataset directory."""
            return str(fake_dataset_path)

    monkeypatch.setitem(sys.modules, "kagglehub", DummyKaggleHub())

    config = {
        "source_type": "kaggle",
        "files": ["data.csv"],
        "kaggle_dataset": "user/dataset",
    }

    result = uploader.load_files_from_source(config)
    assert len(result) == 1
    assert os.path.exists(result[0])


# --------------------------------------------------------------------------- #
#                              Tests: upload_files                             #
# --------------------------------------------------------------------------- #
def test_upload_files_success(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test end-to-end upload flow using DummyWorkspaceClient."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    with caplog.at_level(logging.INFO):
        result = uploader.upload_files(
            host="https://dummy",
            token="123",
            env_config=env_config,
            files=[str(file_path)],
        )

    assert "Uploading" in caplog.text
    assert result == ["dbfs:/Volumes/cat/sch/vol/sample.txt"]


def test_upload_files_without_host_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test upload_files fallback to WorkspaceClient() with no credentials."""
    f1 = tmp_path / "data.txt"
    f1.write_text("data")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    result = uploader.upload_files("", "", env_config, [str(f1)])
    assert result[0].endswith("data.txt")


def test_upload_files_missing_local_file(tmp_path: Path) -> None:
    """Ensure missing file triggers FileNotFoundError."""
    missing_file = tmp_path / "nope.txt"
    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    with pytest.raises(FileNotFoundError):
        uploader.upload_files("https://dummy", "token", env_config, [str(missing_file)])


def test_upload_files_with_profile(tmp_path: Path) -> None:
    """Verify upload_files uses profile argument instead of host/token."""
    file_path = tmp_path / "demo.txt"
    file_path.write_text("abc")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    result = uploader.upload_files(
        host=None,
        token=None,
        env_config=env_config,
        files=[str(file_path)],
        profile="my_profile",
    )

    assert result[0].endswith("demo.txt")
