from __future__ import annotations

from typing import Any

from groq import Groq
from openai import OpenAI

from src.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.require_chat_key()

        if settings.llm_provider == "groq":
            self._client: Any = Groq(api_key=settings.groq_api_key)
            self._model = settings.groq_chat_model
            self._provider = "groq"
        else:
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_chat_model
            self._provider = "openai"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
