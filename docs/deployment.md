# Deployment Guide

This project is often edited on one machine and run on another. This guide is for the **target machine** that actually hosts the bot, model server, Chroma data, and network access.

## Required services

| Service | Purpose | Expected setting |
| --- | --- | --- |
| Discord bot token | bot authentication | `DISCORD_TOKEN` |
| Local LLM server | summarization and RAG synthesis | `LLM_API_BASE` |
| SearXNG | private web search | `SEARXNG_URL` |
| Chroma persistent storage | vector database | `CHROMA_DB_PATH` |

## Startup order

1. Ensure the target machine has the repository and virtualenv ready
2. Start the LLM server
3. Start SearXNG
4. Verify `.env` values
5. Start the bot with `python main.py`
6. In Discord, run `!sync` if commands were not registered yet

## Suggested layout

```text
AI-Discord-Bot/
  .env
  chroma_data/
  checkpoints/
  knowledge_base/
  bin/
```

Keep all three data directories on persistent storage.

## `/ask` tuning knobs

These values are optional environment variables for tuning the local-first gap router without code edits:

```env
ASK_MAX_GAPS_PER_CYCLE=3
ASK_LOCAL_RESOLUTION_THRESHOLD=0.72
ASK_WEB_TRIGGER_THRESHOLD=0.38
ASK_PARTIAL_CONTEXT_THRESHOLD=0.28
ASK_LOCAL_RETRY_LIMIT=2
ASK_WEB_BACKOFF_LOOPS=2
ASK_FRESHNESS_MAX_AGE_DAYS=180
ASK_FRESHNESS_PENALTY=0.18
```

## Target-machine validation checklist

Run these on the target machine after deploying new code:

### Static-safe

```bash
./bin/python -m compileall main.py query.py agent config llm storage tools tests
./bin/python -m unittest discover -s tests -q
```

### Runtime checks

1. Confirm LLM server responds
2. Confirm SearXNG responds
3. Start `python main.py`
4. In Discord:
   - run a small `/research`
   - run a small `/crawl_site`
   - run `/ask` on an existing topic
   - interrupt a long `/ask`, restart bot, rerun same request, confirm resume

## Recovery checklist

### If `/research` or `/chain_research` was interrupted

- restart bot
- rerun same command
- confirm checkpoint resume message appears

### If `/ask` was interrupted

- rerun same topic/question/settings to auto-resume from checkpoint
- or attach markdown draft with `resume_from` for manual resume

### If metadata backfill is needed

Dry run first:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic your_topic
```

Apply:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic your_topic --execute
```

Force full rewrite only if you intentionally want to refresh already-derived fields:

```bash
PYTHONPATH=. ./bin/python tools/backfill_metadata.py --topic your_topic --execute --force
```
