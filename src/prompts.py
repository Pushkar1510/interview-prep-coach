from __future__ import annotations

import json
import re
from typing import Any


def format_memory_block(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No prior memories."

    lines: list[str] = []
    for index, item in enumerate(memories, start=1):
        text = str(item.get("memory") or item.get("text") or item.get("data") or "").strip()
        if not text and isinstance(item.get("data"), dict):
            text = str(item["data"].get("memory", "")).strip()
        if text:
            lines.append(f"{index}. {text}")

    return "\n".join(lines) if lines else "No prior memories."


def build_coach_system_prompt(role: str, company: str, memory_block: str) -> str:
    company_line = company.strip() if company.strip() else "unspecified company"
    return f"""You are an expert interview coach helping a candidate prepare for a {role} role at {company_line}.

Use the candidate's durable memories to personalize advice. Reference relevant memories when helpful.
Be concise, practical, and encouraging. Ask follow-up questions when useful.

Known memories about this candidate:
{memory_block}
"""


def build_interview_question_prompt(role: str, company: str, memory_block: str) -> str:
    company_line = company.strip() if company.strip() else "a typical company"
    return f"""You are interviewing a candidate for a {role} role at {company_line}.

Past lessons / weaknesses to probe (if any):
{memory_block}

Generate exactly ONE interview question. Prefer probing known weaknesses when present.
Return only the question text, with no preamble.
"""


def build_feedback_prompt(role: str, question: str, answer: str) -> str:
    return f"""You are evaluating a mock interview answer for a {role} candidate.

Question: {question}

Candidate answer: {answer}

Respond with ONLY valid JSON (no markdown fences) using this schema:
{{
  "score": <integer 1-5>,
  "strengths": "<short string>",
  "improvements": "<short string>",
  "lesson": "<one durable lesson sentence to remember for future sessions>"
}}
"""


def parse_feedback_response(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        data = json.loads(cleaned)
        return {
            "score": int(data.get("score", 0)),
            "strengths": str(data.get("strengths", "")).strip(),
            "improvements": str(data.get("improvements", "")).strip(),
            "lesson": str(data.get("lesson", "")).strip(),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "score": 0,
            "strengths": "",
            "improvements": cleaned[:500],
            "lesson": "Review this answer manually; model returned unstructured feedback.",
        }
