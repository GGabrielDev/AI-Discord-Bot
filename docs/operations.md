# Operations

## Prerequisites

- Python 3.12+
- Discord bot token
- SearXNG instance
- OpenAI-compatible local LLM server
- writable storage for:
  - `chroma_data/`
  - `knowledge_base/`
  - `checkpoints/`

## Initial setup

1. Create virtualenv:
   ```bash
   python -m venv .
   source bin/activate
   ```
2. Install dependencies:
   ```bash
   CFLAGS="-Wno-incompatible-pointer-types" pip install -r requirements.txt
   ```
3. Create `.env`:
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
4. Start LLM server
5. Start bot:
   ```bash
   python main.py
   ```

## Discord commands

| Command | Purpose |
| --- | --- |
| `/research` | autonomous multi-iteration research loop |
| `/crawl_site` | domain-focused crawl into one topic |
| `/ask` | query stored knowledge with local-first RAG |
| `/chain_research` | decompose one large problem into multiple research loops |
| `/finish` | soft-stop active long-running loops |

## Recovery

### Research and chain resume

If process closes during `/research` or `/chain_research`, restart bot and invoke same command again. Checkpoint state is designed to resume progress.

### `/ask` resume

`/ask` does not auto-resume after crash. Use the intermediate or previous markdown report with the `resume_from` attachment field.

## Maintenance

### Remove duplicate source entries

```bash
PYTHONPATH=. ./bin/python tools/clean_db.py --topic radio_research --execute
```

### Backfill retrieval metadata for older collections

Dry run:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic radio_research
```

Apply:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic radio_research --execute
```

All collections:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --execute
```

## Data reset

Full reset:

```bash
rm -rf chroma_data/
rm -rf knowledge_base/
rm -rf checkpoints/
```

Checkpoint-only reset:

```bash
rm -rf checkpoints/
```

## Deployment note

This repository may be edited on one machine and executed on another. Validate runtime behavior on the target device that has:

- the real Chroma data
- the real model server
- the real Discord token
- the real network and scraper environment
