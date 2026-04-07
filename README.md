# Personal Knowledge Base Agent

Local, private AI agent over your own notes.
Uses **nomic-embed-text** for embeddings and **qwen2.5:0.5b** as the LLM — both via Ollama.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your notes
#    Drop .txt or .md files into ./knowledge/

# 3. Ingest your documents
python kb_agent.py ingest

# 4. Chat with your knowledge base
python kb_agent.py chat
```

## All Commands

| Command | What it does |
|---|---|
| `python kb_agent.py ingest` | Reads `./knowledge/` and stores chunks in ChromaDB |
| `python kb_agent.py chat` | Interactive Q&A loop |
| `python kb_agent.py query <text>` | One-shot question from the command line |
| `python kb_agent.py status` | Show how many chunks are stored |

## Adding More Documents

Just drop files into `./knowledge/` (subfolders work too) and run `ingest` again.
Supported formats: `.txt`, `.md`

## Tuning

Edit the config section at the top of `kb_agent.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 400 | Characters per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `TOP_K` | 4 | Chunks retrieved per query |

## How it Works

```
Your .txt/.md files
       ↓ split into chunks
nomic-embed-text → vectors
       ↓ stored in
ChromaDB (local, persistent)

At query time:
Question → embed → search ChromaDB → top-4 chunks
                                          ↓
                               qwen2.5:0.5b generates answer
```
# kb_agent
