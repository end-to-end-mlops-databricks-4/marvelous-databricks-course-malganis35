import os
import yaml
import logging
import pytest
from pathlib import Path
from loguru import logger
from src.mlops_course.data import config_loader


# --- Connect Loguru to standard logging for caplog ---
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


# --- Fixture globale : nettoyage de l'environnement ---
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Supprime les variables Databricks avant chaque test."""
    for var in ["PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ["PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
        monkeypatch.delenv(var, raising=False)


# ----------------- TESTS for load_env() -----------------

def test_load_env_with_profile(monkeypatch, tmp_path, caplog):
    """Test when PROFILE is set in the .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROFILE=my_profile\n")

    def fake_load_dotenv(dotenv_path, override):
        os.environ["PROFILE"] = "my_profile"

    def fake_dotenv_values(path):
        return {"PROFILE": "my_profile"}

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with caplog.at_level(logging.DEBUG):
        host, token, profile = config_loader.load_env(str(env_file))

    assert profile == "my_profile"
    assert host is None and token is None
    assert "Loaded env" in caplog.text


def test_load_env_with_token(monkeypatch, tmp_path, caplog):
    """Test when host/token are present (no profile)."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=https://dummy\nDATABRICKS_TOKEN=secret")

    def fake_load_dotenv(dotenv_path, override):
        os.environ["DATABRICKS_HOST"] = "https://dummy"
        os.environ["DATABRICKS_TOKEN"] = "secret"

    def fake_dotenv_values(path):
        return {
            "DATABRICKS_HOST": "https://dummy",
            "DATABRICKS_TOKEN": "secret",
            "PROFILE": None,  # Important: s'assurer que PROFILE est None
        }

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with caplog.at_level(logging.DEBUG):
        host, token, profile = config_loader.load_env(str(env_file))

    assert host == "https://dummy"
    assert token == "secret"
    assert profile is None
    assert "Loaded env" in caplog.text


def test_load_env_missing_file(tmp_path):
    """Test when .env file does not exist."""
    env_file = tmp_path / "nope.env"
    with pytest.raises(FileNotFoundError):
        config_loader.load_env(str(env_file))


def test_load_env_missing_vars(monkeypatch, tmp_path):
    """Test when file exists but missing PROFILE and host/token."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=\nDATABRICKS_TOKEN=\n")

    def fake_load_dotenv(dotenv_path, override):
        pass

    def fake_dotenv_values(path):
        return {}  # Aucun host/token/profile

    monkeypatch.setattr(config_loader, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config_loader, "dotenv_values", fake_dotenv_values)

    with pytest.raises(OSError) as excinfo:
        config_loader.load_env(str(env_file))

    assert "Missing PROFILE" in str(excinfo.value)


# ----------------- TESTS for load_project_config() -----------------

def test_load_project_config_valid(tmp_path):
    """Test normal YAML loading with environment present."""
    yaml_content = {
        "dev": {"catalog_name": "cat_dev", "schema_name": "sch_dev"},
        "global_setting": {"owner": "team-mlops"},
    }
    config_file = tmp_path / "project.yml"
    config_file.write_text(yaml.safe_dump(yaml_content))

    env_config, global_config = config_loader.load_project_config(str(config_file), "dev")

    assert env_config == {"catalog_name": "cat_dev", "schema_name": "sch_dev"}
    assert "global_setting" in global_config
    assert "dev" not in global_config


def test_load_project_config_missing_env(tmp_path):
    """Test ValueError when environment not found."""
    yaml_content = {"dev": {"x": 1}}
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValueError):
        config_loader.load_project_config(str(config_file), "prd")
