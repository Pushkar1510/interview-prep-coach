from __future__ import annotations

from pathlib import Path
from typing import Any

from mem0 import Memory

from src.config import Settings


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        settings.require_openai_for_mem0()
        self.settings = settings

        data_path = Path(settings.mem0_data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": settings.openai_api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": settings.openai_api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "interview_coach",
                    "path": str(data_path / "qdrant"),
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
        result = self._memory.search(query=query, user_id=user_id, limit=limit)
        if isinstance(result, dict) and "results" in result:
            return list(result["results"])
        if isinstance(result, list):
            return result
        return []

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        result = self._memory.get_all(user_id=user_id)
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
