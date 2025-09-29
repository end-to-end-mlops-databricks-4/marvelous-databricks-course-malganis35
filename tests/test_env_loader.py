import os
import logging
from pathlib import Path
import pytest
from loguru import logger
from src.mlops_course.utils import env_loader


# --- Connect Loguru to standard logging for caplog ---
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.add(PropagateHandler(), format="{message}")


@pytest.fixture(autouse=True)
def cleanup_env(monkeypatch):
    """Nettoie les variables d'environnement entre les tests."""
    for key in ["PROFILE", "DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST"]:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in ["PROFILE", "DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST"]:
        monkeypatch.delenv(key, raising=False)


def test_load_environment_with_existing_env_file(tmp_path, monkeypatch, caplog):
    """Test that load_environment loads variables from a valid .env file."""
    # Crée un faux .env
    env_file = tmp_path / ".env"
    env_file.write_text("PROFILE=my_profile\nDATABRICKS_HOST=https://dummy")

    # Mock load_dotenv pour s'assurer qu'il est bien appelé
    called = {}

    def fake_load_dotenv(dotenv_path, override):
        called["path"] = dotenv_path
        called["override"] = override
        os.environ["PROFILE"] = "my_profile"
        os.environ["DATABRICKS_HOST"] = "https://dummy"

    monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    # Vérifie que load_dotenv a bien été appelé
    assert called["path"] == env_file
    assert called["override"] is True

    # Vérifie les logs
    assert "Loaded environment" in caplog.text
    assert "Databricks profile set to" in caplog.text

    # Vérifie les variables
    assert os.getenv("DATABRICKS_CONFIG_PROFILE") == "my_profile"
    assert os.getenv("DATABRICKS_HOST") == "https://dummy"


def test_load_environment_with_missing_env_file(tmp_path, caplog):
    """Test that a missing .env file logs a warning but still logs profile info."""
    env_file = tmp_path / "missing.env"

    # Capture tous les niveaux de logs (INFO et WARNING)
    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    # Vérifie qu’un warning a bien été logué
    assert "not found" in caplog.text or "⚠️" in caplog.text

    # Vérifie que le log d’info sur le profil apparaît aussi
    assert "Databricks profile set to" in caplog.text

    # Vérifie qu’on logue aussi le host même si non défini
    assert "Databricks host" in caplog.text


def test_load_environment_without_argument_uses_default(monkeypatch, tmp_path, caplog):
    """Test default behavior when env_file argument is None."""
    default_env = tmp_path / ".env"
    default_env.write_text("PROFILE=default_profile")

    # Force ENV_FILE dans les variables d'environnement
    monkeypatch.setenv("ENV_FILE", str(default_env))

    with caplog.at_level(logging.INFO):
        env_loader.load_environment()

    assert "Loaded environment" in caplog.text
    assert os.getenv("DATABRICKS_CONFIG_PROFILE") == "default_profile"


def test_load_environment_without_profile(monkeypatch, tmp_path, caplog):
    """Test when .env file exists but no PROFILE is set."""
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HOST=https://example")

    def fake_load_dotenv(dotenv_path, override):
        os.environ["DATABRICKS_HOST"] = "https://example"

    monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

    with caplog.at_level(logging.INFO):
        env_loader.load_environment(str(env_file))

    # PROFILE n'est pas défini → DATABRICKS_CONFIG_PROFILE doit être None
    assert os.getenv("DATABRICKS_CONFIG_PROFILE") is None
    assert "Databricks host: https://example" in caplog.text
