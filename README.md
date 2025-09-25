# Local Llama Expert Agent (Ollama + Tools + RAG)

A lightweight, local “expert” agent that runs on Windows using **Ollama** + **Llama 3.1**.  
It follows a reasoning loop (plan → use tools → answer), supports function/tool calls, and can search your **local documents** via a simple RAG pipeline.


---

## Features

- **Reasoning agent** with private planning (no chain-of-thought leakage)
- **Tools**:
  - `calculator` – safe arithmetic
  - `web_search` – DuckDuckGo via `ddgs`
  - `retrieve` – local documents search using ChromaDB + sentence-transformers
- **Baked defaults model** via an Ollama **Modelfile** (`llama3.1:8b-expert`)
- **Windows-friendly** setup and commands

---

## Requirements

- Windows 11 (or 10)  
- Python 3.10+  
- [Ollama](https://ollama.com/) with a Llama 3.1 model (we build a custom one from `llama3.1:8b-instruct-q5_K_M`)

---

## Quickstart

```powershell
# in project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

.
├─ expert_agent.py        # main agent with tool-calling + reflection
├─ rag.py                 # ingest & retrieve (ChromaDB + sentence-transformers)
├─ Modelfile              # baked defaults for Ollama model
├─ requirements.txt
├─ docs/                  # your local PDFs/TXT/MD (not tracked by default)
└─ vectorstore/           # Chroma persistence (created after ingest)




