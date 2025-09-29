import os
import io
import sys
import logging
import builtins
import pytest
from loguru import logger
from src.mlops_course.data import uploader


# --- Connect Loguru to standard logging for caplog ---
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --- Dummy implementations for Databricks WorkspaceClient and dependencies ---
class DummyDBFS:
    def __init__(self):
        self.upload_calls = []

    def upload(self, path, file_obj, overwrite=False):
        content = file_obj.read()
        self.upload_calls.append((path, content, overwrite))


class DummyWorkspaceClient:
    """Simule un WorkspaceClient avec sous-objets et upload DBFS."""
    def __init__(self, *args, **kwargs):
        self.dbfs = DummyDBFS()


# Dummy versions for check_catalog_exists, ensure_schema, ensure_volume
def dummy_check_catalog_exists(w, catalog_name):
    logger.info(f"[dummy] Checked catalog {catalog_name}")


def dummy_ensure_schema(w, catalog_name, schema_name):
    logger.info(f"[dummy] Ensured schema {schema_name} in {catalog_name}")


def dummy_ensure_volume(w, catalog_name, schema_name, volume_name):
    logger.info(f"[dummy] Ensured volume {volume_name} in {catalog_name}.{schema_name}")


# --- FIXTURES ---
@pytest.fixture(autouse=True)
def patch_databricks_utils(monkeypatch):
    """Monkeypatch les fonctions de databricks_utils pour éviter les vrais appels."""
    monkeypatch.setattr(uploader, "WorkspaceClient", DummyWorkspaceClient)
    monkeypatch.setattr(uploader, "check_catalog_exists", dummy_check_catalog_exists)
    monkeypatch.setattr(uploader, "ensure_schema", dummy_ensure_schema)
    monkeypatch.setattr(uploader, "ensure_volume", dummy_ensure_volume)
    yield


# --- TESTS load_files_from_source ---
def test_load_files_from_source_local_success(tmp_path):
    """Test loading local files successfully."""
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


def test_load_files_from_source_local_missing(tmp_path):
    """Test that missing file raises FileNotFoundError."""
    config = {
        "source_type": "local",
        "files": ["missing.csv"],
        "local_path": str(tmp_path),
    }
    with pytest.raises(FileNotFoundError):
        uploader.load_files_from_source(config)


def test_load_files_from_source_invalid_type():
    """Invalid source_type should raise ValueError."""
    config = {"source_type": "unknown", "files": []}
    with pytest.raises(ValueError):
        uploader.load_files_from_source(config)


def test_load_files_from_source_kaggle(monkeypatch, tmp_path):
    """Test Kaggle dataset loading using monkeypatch."""
    fake_dataset_path = tmp_path / "dataset"
    fake_dataset_path.mkdir()
    (fake_dataset_path / "data.csv").write_text("dummy")

    # Simule kagglehub.dataset_download()
    class DummyKaggleHub:
        def dataset_download(self, dataset):
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


# --- TESTS upload_files ---
def test_upload_files_success(tmp_path, caplog):
    """Test upload_files end-to-end with dummy WorkspaceClient."""
    # Crée un fichier local
    f1 = tmp_path / "sample.txt"
    f1.write_text("content")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    with caplog.at_level(logging.INFO):
        result = uploader.upload_files(
            host="https://dummy",
            token="123",
            env_config=env_config,
            files=[str(f1)],
        )

    # Vérifie les logs
    assert "Uploading" in caplog.text
    assert result == ["dbfs:/Volumes/cat/sch/vol/sample.txt"]


def test_upload_files_without_host_token(monkeypatch, tmp_path):
    """Test upload_files fallback to WorkspaceClient() without host/token/profile."""
    f1 = tmp_path / "data.txt"
    f1.write_text("data")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    # Pas d'erreur, fallback automatique
    result = uploader.upload_files("", "", env_config, [str(f1)])
    assert result[0].endswith("data.txt")


def test_upload_files_missing_local_file(tmp_path):
    """If local file is missing, open() should raise FileNotFoundError."""
    missing_file = tmp_path / "nope.txt"
    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    with pytest.raises(FileNotFoundError):
        uploader.upload_files("https://dummy", "token", env_config, [str(missing_file)])


def test_upload_files_with_profile(tmp_path):
    """Test upload_files with profile argument."""
    f1 = tmp_path / "demo.txt"
    f1.write_text("abc")

    env_config = {"catalog_name": "cat", "schema_name": "sch", "volume_name": "vol"}

    result = uploader.upload_files(
        host=None,
        token=None,
        env_config=env_config,
        files=[str(f1)],
        profile="my_profile"
    )

    assert result[0].endswith("demo.txt")