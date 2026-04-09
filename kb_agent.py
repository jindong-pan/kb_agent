"""
Personal Knowledge Base Agent
Uses: nomic-embed-text (embeddings) + qwen2.5:0.5b (LLM) via Ollama + ChromaDB

Folder convention:
  knowledge/
  ├── fleeting/    ← daily scratch, NOT ingested
  ├── literature/  ← source summaries, NOT ingested
  └── permanent/   ← atomic notes, ingested by default
"""

import sys
import glob
import requests
import json
import chromadb
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "qwen2.5:0.5b"
CHROMA_DIR      = "./chroma_db"
KNOWLEDGE_DIR   = "./knowledge"
PERMANENT_DIR   = f"{KNOWLEDGE_DIR}/permanent"   # ingested by default
SKIP_DIRS       = {"fleeting", "literature"}      # always skipped unless --all
CHUNK_SIZE      = 400                             # characters per chunk
CHUNK_OVERLAP   = 80
TOP_K           = 4                               # chunks retrieved per query


# ── Ollama helpers ─────────────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    """Get embedding vector from nomic-embed-text via Ollama."""
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def chat(prompt: str) -> str:
    """Send a prompt to qwen2.5:0.5b and stream back the response."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
        stream=True,
    )
    resp.raise_for_status()
    result = []
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            result.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
    return "".join(result)


# ── Text splitting ─────────────────────────────────────────────────────────────
def split_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text":   chunk_text,
                "source": source,
                "index":  i,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        i += 1
    return chunks


# ── ChromaDB setup ─────────────────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        "knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────
def collect_files(ingest_all: bool = False) -> list[str]:
    """
    Return files to ingest.
    - Default : only knowledge/permanent/
    - --all   : everything under knowledge/, skipping fleeting/ and literature/
    """
    if ingest_all:
        root = Path(KNOWLEDGE_DIR)
        files = []
        for ext in ("*.txt", "*.md"):
            for f in root.rglob(ext):
                # skip any path whose parts include a SKIP_DIR name
                if not any(part in SKIP_DIRS for part in f.parts):
                    files.append(str(f))
        return files
    else:
        return (
            glob.glob(f"{PERMANENT_DIR}/**/*.txt", recursive=True) +
            glob.glob(f"{PERMANENT_DIR}/**/*.md",  recursive=True)
        )


def ingest(ingest_all: bool = False):
    """Read notes and store chunks in ChromaDB."""
    files = collect_files(ingest_all)
    source_label = KNOWLEDGE_DIR if ingest_all else PERMANENT_DIR

    if not files:
        if ingest_all:
            print(f"  No .txt or .md files found under '{KNOWLEDGE_DIR}/'")
        else:
            print(f"  No files found in '{PERMANENT_DIR}/'")
            print(f"  Create permanent notes there, or run:")
            print(f"    python kb_agent.py ingest --all   # ingest all except fleeting/literature")
        return

    skipped = SKIP_DIRS if ingest_all else set()
    if skipped:
        print(f"  Skipping folders: {', '.join(sorted(skipped))}")
    print(f"  Found {len(files)} file(s) in '{source_label}'\n")

    col   = get_collection()
    total = 0

    for fpath in sorted(files):
        text   = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        chunks = split_text(text, fpath)
        print(f"  {'→':>2} {fpath}  ({len(chunks)} chunks)")

        col.upsert(
            ids        = [f"{fpath}::{c['index']}" for c in chunks],
            embeddings = [embed(c["text"])          for c in chunks],
            documents  = [c["text"]                 for c in chunks],
            metadatas  = [{"source": c["source"], "index": c["index"],
                           "folder": Path(fpath).parent.name}
                          for c in chunks],
        )
        total += len(chunks)

    print(f"\n  ✓ {total} chunks stored in ChromaDB at '{CHROMA_DIR}/'")
    print(f"  Total in DB: {col.count()}")


# ── Query ──────────────────────────────────────────────────────────────────────
def query(question: str, show_dist: bool = False):
    """Embed the question, retrieve top-k chunks, ask the LLM."""
    col = get_collection()
    count = col.count()
    if count == 0:
        print("  Knowledge base is empty. Run:  python kb_agent.py ingest")
        return

    q_embed = embed(question)
    results = col.query(query_embeddings=[q_embed], n_results=min(TOP_K, count),
                        include=["documents", "metadatas", "distances"])

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}]\n{d}" for d, m in zip(docs, metadatas)
    )

    prompt = f"""You are a helpful personal knowledge assistant.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have information about that in your knowledge base."

Context:
{context}

Question: {question}

Answer:"""

    print("\n🤖 Answer:\n")
    answer = chat(prompt)
    print(answer)

    if show_dist:
        print(f"\ndist={min(distances):.4f}")

    print("\n📄 Sources used:")
    seen = set()
    for m, dist in zip(metadatas, distances):
        src = m["source"]
        if src not in seen:
            line = f"   • {src}"
            if show_dist:
                line += f"  (dist={dist:.4f})"
            print(line)
            seen.add(src)


# ── Interactive chat loop ──────────────────────────────────────────────────────
def interactive(show_dist: bool = False):
    print("🧠 Personal Knowledge Base — interactive mode")
    print("   Type your question, or 'quit' to exit.\n")
    col = get_collection()
    print(f"   Chunks in DB: {col.count()}\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        query(q, show_dist=show_dist)
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    cmd      = sys.argv[1] if len(sys.argv) > 1 else "chat"
    flags    = set(sys.argv[2:])
    ingest_all = "--all" in flags
    show_dist  = "-dist" in flags

    if cmd == "ingest":
        label = "all (except fleeting/literature)" if ingest_all else "permanent/"
        print(f"📥 Ingesting [{label}] …\n")
        ingest(ingest_all)

    elif cmd == "query" and len(sys.argv) > 2:
        q = " ".join(a for a in sys.argv[2:] if not a.startswith("--") and a != "-dist")
        print(f"🔍 Query: {q}\n")
        query(q, show_dist=show_dist)

    elif cmd == "chat":
        interactive(show_dist=show_dist)

    elif cmd == "status":
        col = get_collection()
        perm  = len(collect_files(False))
        total = len(collect_files(True))
        print(f"📊 Knowledge base status")
        print(f"   Chunks in DB      : {col.count()}")
        print(f"   permanent/ files  : {perm}")
        print(f"   Total files found : {total}  (excl. fleeting/literature)")
        print(f"   ChromaDB path     : {CHROMA_DIR}")
        print(f"   Embed model       : {EMBED_MODEL}")
        print(f"   LLM model         : {LLM_MODEL}")

    else:
        print("Usage:")
        print("  python kb_agent.py ingest             # ingest knowledge/permanent/ only")
        print("  python kb_agent.py ingest --all       # ingest everything except fleeting/ & literature/")
        print("  python kb_agent.py chat               # interactive Q&A")
        print("  python kb_agent.py chat -dist         # interactive Q&A with similarity distances")
        print("  python kb_agent.py query <text>       # one-shot query")
        print("  python kb_agent.py query <text> -dist # one-shot query with similarity distances")
        print("  python kb_agent.py status             # show DB + file stats")


if __name__ == "__main__":
    main()
