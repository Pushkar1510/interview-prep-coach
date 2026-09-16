from src.prompts import (
    build_coach_system_prompt,
    format_memory_block,
    parse_feedback_response,
)


def test_format_memory_block_empty():
    assert format_memory_block([]) == "No prior memories."


def test_format_memory_block_lists_items():
    memories = [
        {"memory": "Struggles with system design tradeoffs"},
        {"text": "Prefers behavioral practice"},
    ]
    block = format_memory_block(memories)
    assert "1. Struggles with system design tradeoffs" in block
    assert "2. Prefers behavioral practice" in block


def test_build_coach_system_prompt_includes_role_and_memories():
    prompt = build_coach_system_prompt(
        role="Backend Engineer",
        company="Stripe",
        memory_block="1. Struggles with system design",
    )
    assert "Backend Engineer" in prompt
    assert "Stripe" in prompt
    assert "Struggles with system design" in prompt


def test_parse_feedback_response_json():
    raw = """
    {
      "score": 3,
      "strengths": "Clear structure",
      "improvements": "Add metrics",
      "lesson": "Always quantify impact"
    }
    """
    parsed = parse_feedback_response(raw)
    assert parsed["score"] == 3
    assert parsed["lesson"] == "Always quantify impact"


def test_parse_feedback_response_fallback_on_garbage():
    parsed = parse_feedback_response("not json at all")
    assert parsed["score"] == 0
    assert "not json" in parsed["improvements"].lower() or parsed["lesson"]
