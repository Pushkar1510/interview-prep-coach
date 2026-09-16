import os

import pytest

from src.config import load_settings


def test_load_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("MEM0_DATA_PATH", str(tmp_path / "mem0"))

    settings = load_settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.groq_api_key is None
    assert settings.llm_provider == "openai"
    assert settings.openai_chat_model == "gpt-4o-mini"
    assert settings.groq_chat_model == "llama-3.3-70b-versatile"
    assert settings.mem0_data_path == str(tmp_path / "mem0")


def test_invalid_provider_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")

    settings = load_settings()

    assert settings.llm_provider == "openai"


def test_require_chat_key_groq_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = load_settings()

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        settings.require_chat_key()


def test_require_openai_for_mem0(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    settings = load_settings()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_openai_for_mem0()
