from __future__ import annotations

from typing import Any

from src.llm_client import LLMClient
from src.memory_service import MemoryService
from src.prompts import (
    build_coach_system_prompt,
    build_feedback_prompt,
    build_interview_question_prompt,
    format_memory_block,
    parse_feedback_response,
)


class CoachService:
    def __init__(self, llm: LLMClient, memory: MemoryService) -> None:
        self.llm = llm
        self.memory = memory

    def chat(
        self,
        user_id: str,
        role: str,
        company: str,
        history: list[dict[str, str]],
        user_message: str,
        *,
        persist: bool = True,
    ) -> str:
        memories = self.memory.search(query=user_message, user_id=user_id, limit=5)
        memory_block = format_memory_block(memories)
        system_prompt = build_coach_system_prompt(role, company, memory_block)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        reply = self.llm.chat(messages)

        if persist:
            self.persist_chat_turn(
                user_id=user_id,
                role=role,
                user_message=user_message,
                reply=reply,
            )
        return reply

    def persist_chat_turn(
        self,
        user_id: str,
        role: str,
        user_message: str,
        reply: str,
    ) -> None:
        self.memory.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ],
            user_id=user_id,
            metadata={"category": "profile", "role": role},
        )

    def generate_question(self, user_id: str, role: str, company: str) -> str:
        memories = self.memory.search(
            query=f"interview weaknesses lessons for {role}",
            user_id=user_id,
            limit=5,
        )
        memory_block = format_memory_block(memories)
        prompt = build_interview_question_prompt(role, company, memory_block)
        return self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Ask your interview question now."},
            ],
            temperature=0.7,
        )

    def evaluate_answer(
        self,
        user_id: str,
        role: str,
        question: str,
        answer: str,
    ) -> dict[str, Any]:
        prompt = build_feedback_prompt(role, question, answer)
        raw = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Evaluate the answer."},
            ],
            temperature=0.2,
        )
        feedback = parse_feedback_response(raw)
        lesson = feedback.get("lesson") or ""
        if lesson:
            self.memory.add_lesson(lesson=lesson, user_id=user_id, role=role)
        return feedback
