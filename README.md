# AI Discord Research Agent

Autonomous Discord bot for web research, site crawling, and local knowledge-base Q&A.

## What it does

- `/research` runs iterative search -> scrape -> summarize -> replan loops
- `/crawl_site` ingests one site with domain lock and PDF support
- `/ask` queries ChromaDB with local-first RAG, gap tracking, and optional web escalation
- `/chain_research` decomposes a large topic into multiple research branches
- `/finish` requests a graceful stop

## Core stack

| Layer | Tooling |
| --- | --- |
| Bot interface | `discord.py` |
| Search | SearXNG |
| Scraping | `httpx`, `beautifulsoup4`, PDF parsers |
| LLM API | OpenAI-compatible local server (`llama-server`, Ollama, vLLM, LM Studio) |
| Storage | ChromaDB + markdown knowledge base on disk |

## Quick start

1. Create virtualenv and install dependencies:
   ```bash
   python -m venv .
   source bin/activate
   CFLAGS="-Wno-incompatible-pointer-types" pip install -r requirements.txt
   ```
2. Create `.env`:
   ```env
   DISCORD_TOKEN=your_discord_bot_token_here
   LLM_API_BASE=http://localhost:8080/v1
   LLM_API_KEY=sk-no-key-required
   LLM_MODEL_NAME=local-model
   LLM_MAX_TOKENS=2048
   LLM_CONTEXT_WINDOW=131072
   LLM_TIMEOUT=120
   SEARXNG_URL=http://localhost:8081
   CHROMA_DB_PATH=./chroma_data
   ```
3. Start your model server. Detailed guide: [`docs/llama_server.md`](docs/llama_server.md)
4. Start bot:
   ```bash
   python main.py
   ```
5. In Discord, run `!sync` once to register slash commands.

## Main commands

```text
/research subject:"solid state batteries" iterations:5 depth:3
/crawl_site url:"https://example.com/docs" topic:"example_docs" max_pages:50
/ask topic:"solid state batteries" question:"What are the main electrolyte materials?" mode:Thorough style:Investigative
/chain_research prompt:"Map Paraguay radio law end to end" topic:"radio_laws"
/finish
```

## Project status

Current codebase includes:

- crash-resume for `/research` and `/chain_research`
- local-first `/ask` gap routing with deferred gap queue
- metadata backfill tool for older Chroma collections
- dual-ingestion storage (`summary` + `raw`)

Known boundary:

- `/ask` does **not** auto-resume after process crash; it resumes through `resume_from` markdown reports

## Maintenance tools

```bash
PYTHONPATH=. ./bin/python tools/clean_db.py --topic radio_research --execute
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic radio_research --execute
```

## Documentation map

- [`docs/index.md`](docs/index.md) — documentation hub
- [`docs/architecture.md`](docs/architecture.md) — system structure and data flow
- [`docs/operations.md`](docs/operations.md) — setup, run, maintenance, recovery
- [`docs/current_status.md`](docs/current_status.md) — current capabilities, recent changes, known limits
- [`docs/llama_server.md`](docs/llama_server.md) — model server tuning
