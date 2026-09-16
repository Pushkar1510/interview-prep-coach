# Interview Prep Coach

Self-learning interview practice coach built with **Streamlit**, **Mem0**, and **Ollama** (free/local by default). Optional OpenAI/Groq chat providers. Remembers your target role, weaknesses, and mock-interview lessons across sessions.

## Architecture

```text
Streamlit UI → CoachService → Chat LLM (Ollama | Groq | OpenAI)
                 ↓
            MemoryService (Mem0 + Ollama embeddings + on-disk Qdrant)
```

## Tech stack

- Python, Streamlit
- Mem0 (long-term memory)
- Ollama (default chat + Mem0 LLM/embeddings — free/local)
- OpenAI / Groq (optional chat)
- pytest

## Setup (free path)

1. Install [Ollama](https://ollama.com) and pull models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

2. App setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Ensure Ollama is running (`ollama serve` if needed). Default `.env` uses `LLM_PROVIDER=ollama` — no paid API keys required.

## 30-second demo

1. Set User ID + target role (e.g. Backend Engineer).
2. Keep provider on **ollama**.
3. In Coach chat: "I struggle with system design tradeoffs."
4. Open Mock interview → Generate question → answer → submit.
5. Open Memories → confirm profile/lesson entries.

## Resume bullet

Built an Interview Prep Coach with Streamlit and Mem0 that persists candidate profile and interview lessons via local Ollama embeddings, retrieves relevant memories before each reply, and supports Ollama/Groq/OpenAI for chat.

## Tests

```bash
pytest -v
```
