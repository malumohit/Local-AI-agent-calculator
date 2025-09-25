# rag.py
from __future__ import annotations
import os, re, uuid, pathlib, json
from typing import List, Dict, Any

# --- Chroma client (works for 0.4/0.5) ---
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(path="./vectorstore")
except Exception:
    client = chromadb.Client(Settings(persist_directory="./vectorstore"))

COLL_NAME = "docs"
def _collection():
    try:
        return client.get_or_create_collection(COLL_NAME, metadata={"hnsw:space":"cosine"})
    except TypeError:
        return client.get_or_create_collection(name=COLL_NAME, metadata={"hnsw:space":"cosine"})

# --- Embeddings (CPU, small & fast) ---
from sentence_transformers import SentenceTransformer
_EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBED_NAME, device="cpu")
    return _model

def _embed(texts: List[str]) -> List[List[float]]:
    return _embedder().encode(texts, normalize_embeddings=True).tolist()

# --- Loaders & chunking ---
def load_text(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk(text: str, max_words: int = 350) -> List[str]:
    if not text: return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur, cnt = [], [], 0
    for p in paras:
        w = p.split()
        if cnt + len(w) > max_words and cur:
            chunks.append(" ".join(cur)); cur, cnt = [], 0
        cur.extend(w); cnt += len(w)
    if cur: chunks.append(" ".join(cur))
    return chunks

# --- Ingest & retrieve ---
def ingest_docs(folder: str = "docs") -> Dict[str, Any]:
    coll = _collection()
    folder_path = pathlib.Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in folder_path.rglob("*") if p.suffix.lower() in {".txt",".md",".pdf"}]
    added = 0
    for f in files:
        text = load_text(f)
        chs = chunk(text)
        if not chs: continue
        ids = [f"{f.name}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chs))]
        metas = [{"source": str(f), "chunk": i} for i in range(len(chs))]
        coll.add(ids=ids, documents=chs, metadatas=metas, embeddings=_embed(chs))
        added += len(chs)
    try:
        client.persist()
    except Exception:
        pass
    return {"files": len(files), "chunks": added}

def retrieve_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    coll = _collection()
    qv = _embed([query])[0]
    res = coll.query(query_embeddings=[qv], n_results=int(k))
    out = []
    for i in range(len(res.get("ids",[[]])[0])):
        out.append({
            "source": res["metadatas"][0][i].get("source",""),
            "chunk_index": res["metadatas"][0][i].get("chunk", 0),
            "text": res["documents"][0][i],
            "score": 1 - res.get("distances", [[0]])[0][i]
        })
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest","query"])
    ap.add_argument("--folder", default="docs")
    ap.add_argument("--q", default="")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "ingest":
        print(json.dumps(ingest_docs(args.folder), indent=2))
    else:
        print(json.dumps(retrieve_chunks(args.q, args.k), ensure_ascii=False, indent=2))
