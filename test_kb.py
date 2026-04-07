import requests, chromadb
from pathlib import Path

CHROMA_DIR  = "./chroma_db"
OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GOOD_MATCH   = 0.55
WEAK_MATCH   = 0.75
OUT_OF_SCOPE = 0.55

def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def search(col, query_text, top_k=3):
    results = col.query(query_embeddings=[embed(query_text)], n_results=min(top_k, col.count()),
                        include=["documents","metadatas","distances"])
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))

def bar(d, width=30):
    filled = int(min(d, 1.0) * width)
    return "█" * filled + "░" * (width - filled)

SANITY_TESTS = [
    ("What is a Zettelkasten?",                              ["what-is-zettelkasten"]),
    ("Who was Niklas Luhmann and what did he create?",       ["niklas-luhmann","what-is-zettelkasten"]),
    ("What does atomic mean for a note?",                    ["atomic-note-principle"]),
    ("How should I handle fleeting notes daily?",            ["fleeting-notes-workflow","daily-note-taking-habit"]),
    ("Why should I write in my own words?",                  ["writing-in-your-own-words"]),
    ("How should permanent notes link to each other?",       ["linking-notes"]),
    ("What is the difference between Zettelkasten and folders?", ["zettelkasten-vs-folders"]),
    ("How does Zettelkasten help with vector databases?",    ["zettelkasten-for-vector-db"]),
    ("How do I build a daily note habit?",                   ["daily-note-taking-habit"]),
    ("What are the three note types?",                       ["three-note-types"]),
]

SIMILARITY_QUERIES = [
    "emergent connections between ideas",
    "Luhmann slip box index cards",
    "atomic idea one concept per file",
    "processing fleeting notes into permanent",
]

OUT_OF_SCOPE_QUERIES = [
    "what is the capital of France?",
    "how to cook pasta carbonara",
    "best JavaScript frameworks 2024",
]

def run():
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection("knowledge_base")
    total_chunks = col.count()
    print(f"\n{'─'*60}")
    print(f"  Knowledge Base Test Report")
    print(f"  Chunks in DB : {total_chunks}  |  Embed: {EMBED_MODEL}")
    print(f"{'─'*60}")
    if total_chunks == 0:
        print("  Empty DB. Run: python kb_agent.py ingest"); return

    passed = warned = failed = 0

    print("\n── Test 1: Sanity checks ────────────────────────────────────────\n")
    for question, acceptable in SANITY_TESTS:
        hits = search(col, question, 3)
        top_stem = Path(hits[0][1]["source"]).stem
        top_dist = hits[0][2]
        top3     = [Path(m["source"]).stem for _, m, _ in hits]
        hit1 = top_stem in acceptable
        hit3 = any(s in acceptable for s in top3)
        if hit1 and top_dist < GOOD_MATCH:   status = "✓ PASS";  passed += 1
        elif hit1 and top_dist < WEAK_MATCH: status = "~ WARN  (correct, weak confidence)"; warned += 1
        elif hit3:                            status = "~ WARN  (expected in top-3, not top-1)"; warned += 1
        else:                                 status = f"✗ FAIL  (expected: {acceptable})"; failed += 1
        print(f"  {status}")
        print(f"  Q : {question}")
        print(f"  → {top_stem}  dist={top_dist:.3f}  {bar(top_dist)}\n")

    print("── Test 2: Similarity inspection ────────────────────────────────\n")
    for q in SIMILARITY_QUERIES:
        hits = search(col, q, 3)
        print(f"  Query: '{q}'")
        for _, meta, dist in hits:
            stem = Path(meta["source"]).stem
            lbl  = "✓" if dist < GOOD_MATCH else ("~" if dist < WEAK_MATCH else "✗")
            print(f"    {lbl} {dist:.3f}  {bar(dist,20)}  {stem}")
        print()

    print("── Test 3: Out-of-scope (informational — small DBs always return something) ──\n")
    for q in OUT_OF_SCOPE_QUERIES:
        _, _, d = search(col, q, 1)[0]
        lbl = "✓ uncertain" if d >= OUT_OF_SCOPE else "~ low dist (normal for small DB)"
        print(f"  {lbl}  dist={d:.3f}  {bar(d)}")
        print(f"  Q : {q}\n")

    total = passed + warned + failed
    print(f"{'─'*60}")
    print(f"  Results:  ✓ {passed} passed  ~ {warned} warned  ✗ {failed} failed  / {total} sanity tests\n")
    if   failed == 0 and warned == 0:    print("  🟢 Excellent — retrieval is precise.")
    elif failed == 0 and warned <= 2:    print("  🟢 Good — warnings reflect normal semantic overlap.\n     Your knowledge base is working correctly.")
    elif failed == 0:                    print("  🟡 Acceptable — consider rewording overlapping notes.")
    else:                                print("  🔴 Review needed — check failed queries above.")
    print(f"{'─'*60}\n")

if __name__ == "__main__":
    run()
