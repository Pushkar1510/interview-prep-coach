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
    mem0_data_path: str

    def require_chat_key(self) -> None:
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is required when LLM_PROVIDER=groq. "
                    "Add it to your .env file."
                )
        elif not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Add it to your .env file."
            )

    def require_openai_for_mem0(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for Mem0 embeddings and memory "
                "extraction, even when chat uses Groq."
            )


def load_settings() -> Settings:
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "groq"}:
        provider = "openai"

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip() or None,
        llm_provider=provider,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        groq_chat_model=os.getenv(
            "GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"
        ).strip(),
        mem0_data_path=os.getenv("MEM0_DATA_PATH", "./data/mem0").strip(),
    )
