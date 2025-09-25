from __future__ import annotations
import json, traceback
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel
from rag import ingest_docs, retrieve_chunks
import ollama

# Prefer the renamed package 'ddgs', fall back to old name if needed
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS

# ------------------ Messages ------------------
class Msg(BaseModel):
    role: Literal["system","user","assistant","tool"]
    content: str
    name: Optional[str] = None

SYSTEM_PROMPT = (
    "You are an expert assistant. Plan privately, use tools when helpful, "
    "and NEVER reveal chain-of-thought. Be concise and correct. "
    "When using the retrieve tool, quote short snippets and include [source] after facts."
)

# ------------------ Tools ------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a safe arithmetic expression (+,-,*,/,**,%,//, parentheses).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web; returns JSON list of results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Retrieve top matching snippets from local docs (ChromaDB). Returns JSON with source, chunk_index, text, score.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"]
            }
        }
    }
]

def _safe_eval_math(expr: str) -> str:
    import ast, operator as op
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.FloorDiv: op.floordiv, ast.Mod: op.mod
    }
    def _eval(node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed: return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed: return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsafe expression")
    return str(_eval(ast.parse(expr, mode='eval').body))

def _tool_search_web(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return json.dumps(hits, ensure_ascii=False)

def execute_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "calculator":
            return json.dumps({"result": _safe_eval_math(args["expression"])})
        if name == "web_search":
            return _tool_search_web(args["query"], int(args.get("max_results", 5)))
        if name == "retrieve":
            return json.dumps(retrieve_chunks(args["query"], int(args.get("k", 5))), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})
    return json.dumps({"error": f"Unknown tool {name}"})

# ------------------ LLM adapter (Ollama) ------------------
MODEL = "llama3.1:8b-expert"

def chat(messages: List[Dict[str, Any]], tools=None):
    """Thin wrapper around ollama.chat that passes tools and returns the response dict."""
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "options": {"num_ctx": 8192, "temperature": 0.2}
    }
    if tools:
        kwargs["tools"] = tools
    return ollama.chat(**kwargs)

# ------------------ Agent core ------------------
class Agent:
    def __init__(self, reflection: bool = True, max_iters: int = 6):
        self.reflection = reflection
        self.max_iters = max_iters
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        # Tool loop
        for _ in range(self.max_iters):
            resp = chat(self.history, tools=TOOLS)
            msg = resp["message"]
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Execute each tool and append results
                for call in tool_calls:
                    fn = call["function"]["name"]
                    raw_args = call["function"].get("arguments", {})
                    if isinstance(raw_args, str):
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = {}
                    result = execute_tool(fn, args)
                    self.history.append({"role": "tool", "name": fn, "content": result})
                # continue loop so the model can use tool outputs
                continue
            else:
                # Final answer
                content = msg.get("content", "")
                self.history.append({"role": "assistant", "content": content})
                break

        # --- Reflection (safe, optional) ---
        if self.reflection:
            draft = self.history[-1]["content"]
            reviewer = chat([
                {"role": "system", "content": "You are a meticulous editor."},
                {"role": "user", "content": (
                    "Review the DRAFT for clarity/correctness.\n"
                    "Return JSON only:\n"
                    '{"verdict":"ok"}  OR  {"verdict":"revise","answer":"<improved final answer>"}\n'
                    "DRAFT:\n" + draft
                )}
            ])
            c = reviewer["message"]["content"].strip()

            # Try to parse JSON; if parse fails, keep the original draft
            try:
                data = json.loads(c)
                if isinstance(data, dict) and data.get("verdict") == "revise" and data.get("answer"):
                    self.history[-1] = {"role": "assistant", "content": data["answer"]}
                # if verdict == "ok" -> keep draft
            except Exception:
                # Heuristic fallback: if it starts with "OK", keep draft; otherwise treat as revised answer
                if not c.upper().startswith("OK"):
                    self.history[-1] = {"role": "assistant", "content": c}

        return self.history[-1]["content"]

# ------------------ CLI ------------------
if __name__ == "__main__":
    agent = Agent(reflection=False, max_iters=6)  # turn reflection on later if you like
    print(f"Agent ready on {MODEL}. Press Ctrl+C to exit.")
    try:
        while True:
            q = input("\nYou: ")
            a = agent.ask(q)
            print("\nAgent:", a)
    except KeyboardInterrupt:
        print("\nBye!")
