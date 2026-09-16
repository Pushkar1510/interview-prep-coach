import pytest

from src.config import load_settings


def test_load_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    monkeypatch.setenv("MEM0_DATA_PATH", str(tmp_path / "mem0"))

    settings = load_settings()

    assert settings.openai_api_key == ""
    assert settings.groq_api_key is None
    assert settings.llm_provider == "ollama"
    assert settings.openai_chat_model == "gpt-4o-mini"
    assert settings.groq_chat_model == "llama-3.3-70b-versatile"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_chat_model == "llama3.2:1b"
    assert settings.ollama_embed_model == "nomic-embed-text"
    assert settings.mem0_data_path == str(tmp_path / "mem0")


def test_invalid_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")

    settings = load_settings()

    assert settings.llm_provider == "ollama"


def test_require_chat_key_groq_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = load_settings()

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        settings.require_chat_key()


def test_require_chat_key_openai_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_settings()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_chat_key()


def test_require_mem0_ready_ok_for_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_settings()
    settings.require_mem0_ready()


def test_require_chat_key_ollama_needs_no_cloud_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = load_settings()
    settings.require_chat_key()
