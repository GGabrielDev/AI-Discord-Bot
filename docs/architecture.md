# Architecture

## High-level layout

```text
Discord
  ├── /research
  │     ├── planner -> search -> scraper -> summarizer
  │     ├── ChromaDB storage
  │     ├── markdown article archive
  │     └── checkpoint persistence
  ├── /crawl_site
  │     ├── domain-locked crawler
  │     ├── HTML/PDF ingestion
  │     └── checkpoint persistence
  ├── /ask
  │     ├── query expansion
  │     ├── vector retrieval
  │     ├── SIP local evidence probe
  │     ├── confidence-gated gap routing
  │     └── markdown report synthesis
  └── /chain_research
        ├── prompt decomposition
        ├── repeated /research runs
        └── shared target topic
```

## Main modules

| Path | Responsibility |
| --- | --- |
| `main.py` | Discord bot entrypoint and slash commands |
| `agent/loop.py` | autonomous research loop for `/research` |
| `agent/crawler.py` | focused crawler for `/crawl_site` |
| `agent/planner.py` | search query generation and loop replanning |
| `agent/summarizer.py` | page summarization and chunking |
| `agent/checkpoint.py` | crash-resume state for research and chain flows |
| `query.py` | `/ask` RAG pipeline, gap queue, local-first routing |
| `storage/vectordb.py` | ChromaDB access layer |
| `tools/clean_db.py` | duplicate cleanup utility |
| `tools/backfill_metadata.py` | metadata enrichment for existing collections |

## Storage model

The project stores data in two forms:

1. **ChromaDB**
   - semantic vector search
   - per-chunk metadata
   - `summary` chunks from LLM synthesis
   - `raw` chunks from compressed source text
2. **Markdown files on disk**
   - article archive
   - final reports
   - human-readable audit trail

## `/research` flow

1. Generate search queries for the subject
2. Query SearXNG
3. Scrape HTML or PDF sources
4. Deduplicate by URL and content hash
5. Summarize content through local LLM
6. Store `summary` and `raw` chunks in ChromaDB
7. Save markdown article to disk
8. Re-evaluate knowledge gaps and generate next iteration queries
9. Persist checkpoints after meaningful progress

## `/ask` flow

1. Expand the user question into semantic variants
2. Retrieve summary-first evidence from ChromaDB
3. Pull raw technical evidence from highest-signal sources
4. Synthesize markdown answer draft
5. Extract `Knowledge Gaps`
6. For each queued gap:
   - run SIP local evidence probe
   - score local evidence quality
   - preserve partial local findings if useful
   - escalate to web only when local signal is weak, stale, or repeatedly unresolved
7. Refine draft recursively until loop budget stops or gaps are cleared

## Checkpoint model

Checkpoint coverage currently exists for:

- `/research`
- `/crawl_site`
- `/chain_research`

`/ask` does not currently auto-resume after a crash. It resumes through uploaded markdown drafts with `resume_from`.
