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

## Create your custom Modelfile (Step A):

FROM llama3.1:8b-instruct-q5_K_M
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
SYSTEM You are an expert assistant. Think privately, use tools when needed, never reveal chain-of-thought.


## Build and list:

ollama create llama3.1:8b-expert -f .\Modelfile
ollama list


## Update expert_agent.py:

MODEL = "llama3.1:8b-expert"


## Run the agent:

python .\expert_agent.py


## Example prompts:

What is (2.5e3 + 17)^2 ? Use the calculator.
Write a short email asking for a project update.

## Step B — Add local documents (RAG)

Install deps (already included in requirements.txt):

pip install chromadb sentence-transformers pypdf


## Put files in the docs/ folder (.pdf, .txt, .md).

## Ingest:

python .\rag.py ingest --folder docs


## Ask with retrieval (from the running agent):

Use the retrieve tool to answer: What are the main goals of the Mattress Recovery Center? Include a short quote with [source].

## Project Structure
.
├─ expert_agent.py        # main agent with tool-calling + reflection
├─ rag.py                 # ingest & retrieve (ChromaDB + sentence-transformers)
├─ Modelfile              # baked defaults for Ollama model
├─ requirements.txt
├─ docs/                  # your local PDFs/TXT/MD (not tracked by default)
└─ vectorstore/           # Chroma persistence (created after ingest)
