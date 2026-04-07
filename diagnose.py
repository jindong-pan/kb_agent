"""
diagnose_kb.py — See exactly what's happening with your vectors

Run:  python diagnose_kb.py
"""

import requests
import chromadb
from pathlib import Path

CHROMA_DIR  = "./chroma_db"
OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def search(col, query_text, top_k=3):
    results = col.query(
        query_embeddings=[embed(query_text)],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))


def bar(d, width=25):
    filled = int(min(d, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


col = chromadb.PersistentClient(path=CHROMA_DIR) \
              .get_collection("knowledge_base")

print(f"\n{'═'*65}")
print(f"  DIAGNOSTIC REPORT")
print(f"  Total chunks in DB : {col.count()}")
print(f"{'═'*65}")

# ── 1. Show ALL chunks stored ─────────────────────────────────────────────────
print("\n── All chunks stored in DB ──────────────────────────────────────\n")
all_data = col.get(include=["documents", "metadatas"])
for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
    stem = Path(meta["source"]).stem
    print(f"  [{meta['index']}] {stem}")
    print(f"      {repr(doc[:80])} …")
    print()

# ── 2. Raw distances for every sanity test ────────────────────────────────────
print("\n── Raw distances for each test query ────────────────────────────\n")

TESTS = [
    ("What is a Zettelkasten?",                     "what-is-zettelkasten"),
    ("Who invented the Zettelkasten method?",       "niklas-luhmann"),
    ("What does atomic mean for a note?",           "atomic-note-principle"),
    ("How should I handle fleeting notes daily?",   "fleeting-notes-workflow"),
    ("Why should I write in my own words?",         "writing-in-your-own-words"),
    ("How does linking notes help understanding?",  "linking-notes"),
    ("What is the difference between Zettelkasten and folders?", "zettelkasten-vs-folders"),
    ("How does Zettelkasten help with vector databases?", "zettelkasten-for-vector-db"),
    ("How do I build a daily note habit?",          "daily-note-taking-habit"),
    ("What are the three note types?",              "three-note-types"),
]

all_distances = []

for question, expected in TESTS:
    hits = search(col, question, top_k=3)
    top_stem = Path(hits[0][1]["source"]).stem
    top_dist = hits[0][2]
    all_distances.append(top_dist)

    match = "✓" if top_stem == expected else "✗"
    print(f"  {match} Q: {question}")
    print(f"    Expected : {expected}")
    for i, (doc, meta, dist) in enumerate(hits):
        stem = Path(meta["source"]).stem
        tag  = " ← expected" if stem == expected else ""
        print(f"    #{i+1}  {dist:.4f}  {bar(dist)}  {stem}{tag}")
    print()

# ── 3. Distance summary ───────────────────────────────────────────────────────
print("\n── Distance summary ─────────────────────────────────────────────\n")
avg  = sum(all_distances) / len(all_distances)
mn   = min(all_distances)
mx   = max(all_distances)
print(f"  Top-1 distances across all queries:")
print(f"    Min  : {mn:.4f}")
print(f"    Max  : {mx:.4f}")
print(f"    Avg  : {avg:.4f}")
print()
print(f"  Interpretation:")
if avg < 0.35:
    print("  → Distances look healthy. The GOOD_MATCH threshold in test_kb.py")
    print("    may just be too strict. Try raising it to 0.55.")
elif avg < 0.55:
    print("  → Distances are moderate. Chunks may be slightly large or notes")
    print("    share too much generic language. Try CHUNK_SIZE=300.")
else:
    print("  → Distances are high across the board. This usually means:")
    print("    a) nomic-embed-text uses cosine distance (0–2 range), not 0–1")
    print("    b) ChromaDB is returning L2 distance instead of cosine")
    print("    → Check collection distance metric (see below).")

# ── 4. Check what distance metric ChromaDB is using ──────────────────────────
print("\n── ChromaDB collection metadata ─────────────────────────────────\n")
try:
    meta = col.metadata
    print(f"  Collection metadata: {meta}")
    hnsw = meta.get("hnsw:space", "not set")
    print(f"  Distance metric    : {hnsw}")
    if hnsw in ("l2", "not set"):
        print()
        print("  ⚠️  ChromaDB defaults to L2 (squared Euclidean) distance.")
        print("  L2 distances are NOT in the 0–1 range — they can be 0–4+.")
        print("  For text embeddings, cosine distance is more appropriate.")
        print()
        print("  Fix: delete chroma_db/ and recreate the collection with:")
        print('    chromadb.get_or_create_collection(')
        print('        "knowledge_base",')
        print('        metadata={"hnsw:space": "cosine"}')
        print('    )')
    elif hnsw == "cosine":
        print("  ✓ Using cosine distance (0–1 range). Metric is correct.")
except Exception as e:
    print(f"  Could not read metadata: {e}")

print(f"\n{'═'*65}\n")
