from __future__ import annotations

from pathlib import Path
from typing import Any

from mem0 import Memory

from src.config import Settings


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        settings.require_mem0_ready()
        self.settings = settings

        data_path = Path(settings.mem0_data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        # Local Ollama for Mem0 LLM + embeddings (no paid API).
        # New collection name avoids clashing with older OpenAI 1536-dim data.
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": settings.ollama_chat_model,
                    "temperature": 0.1,
                    "ollama_base_url": settings.ollama_base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": settings.ollama_embed_model,
                    "ollama_base_url": settings.ollama_base_url,
                    "embedding_dims": 768,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "interview_coach_ollama",
                    "path": str(data_path / "qdrant"),
                    "embedding_model_dims": 768,
                },
            },
            "history_db_path": str(data_path / "history.db"),
        }
        self._memory = Memory.from_config(config)

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._memory.add(messages, user_id=user_id, metadata=metadata or {})

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        result = self._memory.search(
            query=query,
            filters={"user_id": user_id},
            top_k=limit,
        )
        if isinstance(result, dict) and "results" in result:
            return list(result["results"])
        if isinstance(result, list):
            return result
        return []

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        result = self._memory.get_all(filters={"user_id": user_id})
        if isinstance(result, dict) and "results" in result:
            return list(result["results"])
        if isinstance(result, list):
            return result
        return []

    def delete(self, memory_id: str) -> None:
        self._memory.delete(memory_id=memory_id)

    def add_lesson(self, lesson: str, user_id: str, role: str) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": f"Interview lesson for {role}: {lesson}",
            }
        ]
        return self.add(
            messages,
            user_id=user_id,
            metadata={"category": "lesson", "role": role},
        )
