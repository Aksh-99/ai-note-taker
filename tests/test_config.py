"""Tests for notetaker.config.

These never touch the real machine's settings: `monkeypatch` (a built-in
pytest fixture) sets or deletes environment variables just for one test and
undoes it afterwards, and `tmp_path` gives each test a fresh empty folder so a
developer's own `.env` file cannot change the results.
"""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from notetaker.cli import main
from notetaker.config import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Run every test in an empty folder with no NOTETAKER_* variables set.

    `autouse=True` applies this fixture to every test in the file
    automatically. Clearing the get_settings cache makes each test read fresh.
    """
    import os

    for name in list(os.environ):
        if name.startswith("NOTETAKER_") or name in {"GROQ_API_KEY", "HF_TOKEN"}:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_free_and_local():
    settings = Settings()
    assert settings.whisper_model == "small"
    assert settings.llm_provider == "ollama"
    assert settings.groq_api_key is None
    assert settings.hf_token is None
    assert settings.sample_rate == 16_000


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("NOTETAKER_WHISPER_MODEL", "base")
    monkeypatch.setenv("NOTETAKER_LLM_TEMPERATURE", "0.7")
    settings = Settings()
    assert settings.whisper_model == "base"
    # pydantic converted the string "0.7" into a float.
    assert settings.llm_temperature == pytest.approx(0.7)


def test_dotenv_file_is_read(tmp_path):
    (tmp_path / ".env").write_text("NOTETAKER_OUTPUT_DIR=my_notes\n")
    assert Settings().output_dir == Path("my_notes")


def test_env_var_beats_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("NOTETAKER_WHISPER_MODEL=tiny\n")
    monkeypatch.setenv("NOTETAKER_WHISPER_MODEL", "medium")
    assert Settings().whisper_model == "medium"


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("NOTETAKER_WHISPER_MODEL", "smal"),  # typo in an allowed choice
        ("NOTETAKER_LLM_PROVIDER", "openai"),  # paid provider not allowed
        ("NOTETAKER_LLM_TEMPERATURE", "5"),  # outside [0, 2]
        ("NOTETAKER_SAMPLE_RATE", "0"),  # must be > 0
    ],
)
def test_invalid_values_fail_loudly(monkeypatch, name, bad_value):
    # `parametrize` runs this one test once per (name, bad_value) pair.
    monkeypatch.setenv(name, bad_value)
    with pytest.raises(ValidationError):
        Settings()


def test_blank_values_mean_none(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LANGUAGE", "")
    monkeypatch.setenv("NOTETAKER_GROQ_API_KEY", "")
    settings = Settings()
    assert settings.language is None
    assert settings.groq_api_key is None


def test_provider_style_key_names_work(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_for_test")
    settings = Settings()
    assert settings.groq_api_key.get_secret_value() == "gsk_fake_for_test"


def test_secrets_never_print(monkeypatch):
    monkeypatch.setenv("NOTETAKER_HF_TOKEN", "hf_fake_for_test")
    settings = Settings()
    assert "hf_fake_for_test" not in repr(settings)
    assert settings.redacted()["hf_token"] == "**********"


def test_get_settings_returns_same_object():
    assert get_settings() is get_settings()


def test_cli_config_masks_secrets(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_for_test")
    assert main(["config"]) == 0
    printed = capsys.readouterr().out
    assert "whisper_model = small" in printed
    assert "gsk_fake_for_test" not in printed


def test_cli_config_reports_bad_value(monkeypatch, capsys):
    monkeypatch.setenv("NOTETAKER_WHISPER_MODEL", "huge")
    assert main(["config"]) == 1
    assert "Invalid configuration" in capsys.readouterr().out


def test_env_example_documents_every_setting():
    """Guard against adding a setting and forgetting to document it."""
    example = (REPO_ROOT / ".env.example").read_text()
    documented = set(re.findall(r"^NOTETAKER_(\w+)=", example, flags=re.MULTILINE))
    expected = {name.upper() for name in Settings.model_fields}
    assert documented == expected


def test_env_example_loads_cleanly(tmp_path):
    # The template itself must be a valid config with no real secrets in it.
    (tmp_path / ".env").write_text((REPO_ROOT / ".env.example").read_text())
    settings = Settings()
    assert settings.groq_api_key is None
    assert settings.hf_token is None
