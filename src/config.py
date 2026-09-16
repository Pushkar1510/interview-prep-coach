from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    groq_api_key: str | None
    llm_provider: str
    openai_chat_model: str
    groq_chat_model: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    mem0_data_path: str

    def require_chat_key(self) -> None:
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is required when LLM_PROVIDER=groq. "
                    "Add it to your .env file."
                )
        elif self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                    "Add it to your .env file."
                )
        # ollama: no cloud API key required

    def require_mem0_ready(self) -> None:
        """Mem0 uses local Ollama for LLM + embeddings (free path)."""
        if not self.ollama_base_url:
            raise ValueError(
                "OLLAMA_BASE_URL is required for Mem0. "
                "Example: http://localhost:11434"
            )


def load_settings() -> Settings:
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in {"openai", "groq", "ollama"}:
        provider = "ollama"

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip() or None,
        llm_provider=provider,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        groq_chat_model=os.getenv(
            "GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"
        ).strip(),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).strip(),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b").strip(),
        ollama_embed_model=os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text"
        ).strip(),
        mem0_data_path=os.getenv("MEM0_DATA_PATH", "./data/mem0").strip(),
    )
