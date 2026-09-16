from __future__ import annotations

import streamlit as st

from src.coach_service import CoachService
from src.config import load_settings
from src.llm_client import LLMClient
from src.memory_service import MemoryService


st.set_page_config(page_title="Interview Prep Coach", page_icon="🎯", layout="wide")


@st.cache_resource
def get_services(provider: str):
    settings = load_settings()
    # Allow UI override of provider for the session
    object.__setattr__(
        settings,
        "llm_provider",
        provider,
    ) if False else None

    # Rebuild settings-like behavior without mutating frozen dataclass:
    from dataclasses import replace

    settings = replace(settings, llm_provider=provider)
    settings.require_openai_for_mem0()
    settings.require_chat_key()

    llm = LLMClient(settings)
    memory = MemoryService(settings)
    coach = CoachService(llm, memory)
    return coach, memory, settings


def main() -> None:
    st.title("Interview Prep Coach")
    st.caption("Self-learning interview practice with Mem0 long-term memory")

    with st.sidebar:
        st.header("Session")
        user_id = st.text_input("User ID", value="demo-user")
        role = st.text_input("Target role", value="Backend Engineer")
        company = st.text_input("Company (optional)", value="")
        provider = st.selectbox("Chat LLM provider", options=["openai", "groq"], index=0)

        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.session_state.mock_question = ""
            st.session_state.mock_feedback = None
            st.rerun()

        st.markdown("---")
        st.markdown(
            "Mem0 needs **OPENAI_API_KEY**. "
            "Groq chat needs **GROQ_API_KEY** when selected."
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mock_question" not in st.session_state:
        st.session_state.mock_question = ""
    if "mock_feedback" not in st.session_state:
        st.session_state.mock_feedback = None

    try:
        coach, memory, _settings = get_services(provider)
    except Exception as exc:  # show setup errors cleanly
        st.error(str(exc))
        st.stop()

    tab_chat, tab_mock, tab_memories = st.tabs(
        ["Coach chat", "Mock interview", "Memories"]
    )

    with tab_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask your coach…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking with memory…"):
                        history_for_model = [
                            m
                            for m in st.session_state.chat_history[:-1]
                            if m["role"] in {"user", "assistant"}
                        ]
                        reply = coach.chat(
                            user_id=user_id,
                            role=role,
                            company=company,
                            history=history_for_model,
                            user_message=prompt,
                        )
                        st.markdown(reply)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as exc:
                st.error(f"Chat failed: {exc}")

    with tab_mock:
        st.subheader("Mock interview")
        if st.button("Generate question"):
            try:
                with st.spinner("Generating question…"):
                    st.session_state.mock_question = coach.generate_question(
                        user_id=user_id, role=role, company=company
                    )
                    st.session_state.mock_feedback = None
            except Exception as exc:
                st.error(f"Could not generate question: {exc}")

        if st.session_state.mock_question:
            st.info(st.session_state.mock_question)
            answer = st.text_area("Your answer", height=160, key="mock_answer")
            if st.button("Submit answer for feedback"):
                if not answer.strip():
                    st.warning("Write an answer first.")
                else:
                    try:
                        with st.spinner("Evaluating…"):
                            feedback = coach.evaluate_answer(
                                user_id=user_id,
                                role=role,
                                question=st.session_state.mock_question,
                                answer=answer.strip(),
                            )
                            st.session_state.mock_feedback = feedback
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")

        if st.session_state.mock_feedback:
            fb = st.session_state.mock_feedback
            st.success(f"Score: {fb.get('score', 0)}/5")
            st.markdown(f"**Strengths:** {fb.get('strengths', '')}")
            st.markdown(f"**Improvements:** {fb.get('improvements', '')}")
            st.markdown(f"**Lesson saved:** {fb.get('lesson', '')}")

    with tab_memories:
        st.subheader("Stored memories")
        if st.button("Refresh memories"):
            st.rerun()
        try:
            items = memory.get_all(user_id=user_id)
        except Exception as exc:
            st.error(f"Could not load memories: {exc}")
            items = []

        if not items:
            st.write("No memories yet. Chat or finish a mock interview first.")
        else:
            for item in items:
                mem_id = str(item.get("id", ""))
                text = str(
                    item.get("memory")
                    or item.get("text")
                    or item.get("data", {}).get("memory", "")
                    if isinstance(item.get("data"), dict)
                    else item.get("data", "")
                )
                meta = item.get("metadata") or {}
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"- {text}")
                    if meta:
                        st.caption(str(meta))
                with cols[1]:
                    if mem_id and st.button("Delete", key=f"del-{mem_id}"):
                        try:
                            memory.delete(mem_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Delete failed: {exc}")


if __name__ == "__main__":
    main()
