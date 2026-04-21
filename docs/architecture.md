# Architecture

## High-level layout

```text
Discord
  ├── /research
  │     ├── planner -> search -> profile prefilter -> scraper -> summarizer
  │     ├── ChromaDB storage
  │     ├── markdown article archive
  │     ├── runtime telemetry
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
  │     └── English markdown report synthesis
  ├── /translate
  │     ├── uploaded markdown intake
  │     ├── markdown-preserving translation
  │     └── translated report archival
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
| `query.py` | `/ask` English-report pipeline, `/translate` helpers, gap queue, local-first routing |
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

## Runtime optimizations

- shared AsyncOpenAI and HTTP clients are reused across hot paths
- runtime profiles drive search prefilter thresholds, HTML/PDF ingest caps, and raw-retention budgets
- PDF handling is lightweight-first, with optional Marker escalation for text-poor/scanned files
- transient in-process caches reduce repeated planner, search, and query-helper work inside a bot session
- runtime telemetry tracks cache/search/LLM/source activity and prints a concise summary at the end of a run

## `/research` flow

1. Generate search queries for the subject
2. Query SearXNG
3. Apply profile-aware search prefiltering before expensive fetches
4. Scrape HTML or PDF sources
5. Use lightweight-first PDF triage when the source is a PDF
6. Deduplicate by URL and content hash
7. Summarize content through local LLM
8. Store `summary` and adaptively retained `raw` chunks in ChromaDB
9. Save markdown article to disk
10. Re-evaluate knowledge gaps and generate next iteration queries
11. Persist checkpoints after meaningful progress
12. Emit runtime telemetry summary if enabled

## `/ask` flow

1. Expand the user question into semantic variants
2. Retrieve summary-first evidence from ChromaDB
3. Pull raw technical evidence from highest-signal sources
4. Synthesize English markdown answer draft
5. Extract `Knowledge Gaps`
6. For each queued gap:
   - run SIP local evidence probe
   - score local evidence quality
   - preserve partial local findings if useful
   - escalate to web only when local signal is weak, stale, or repeatedly unresolved
7. Refine draft recursively until loop budget stops or gaps are cleared

## `/translate` flow

1. Accept uploaded `.md` report from Discord
2. Infer archive topic from the source report filename
3. Translate the markdown while preserving structure
4. Archive the translated copy under the target language
5. Return the translated markdown file to the operator

## Checkpoint model

Checkpoint coverage currently exists for:

- `/research`
- `/crawl_site`
- `/chain_research`
- `/ask` for matching topic/question/settings reruns

`/ask` can also resume from uploaded markdown drafts with `resume_from`.
