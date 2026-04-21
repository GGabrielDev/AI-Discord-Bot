# Current Status

## Summary

Project is in a usable but still evolving state. Core research, crawl, and query flows exist and are integrated. Recent work improved checkpoint recovery and local-first gap handling.

## Implemented

### Stable core flows

- Discord slash-command bot entrypoint
- autonomous `/research` loop
- focused `/crawl_site`
- `/chain_research` orchestration
- `/ask` markdown report generation
- Chroma-backed vector retrieval
- markdown article/report persistence

### Data handling

- dual-ingestion storage:
  - `summary` chunks
  - `raw` chunks
- URL-level duplicate avoidance
- content-hash duplicate avoidance
- source metadata on chunk writes

### Resilience

- checkpoint recovery for `/research`
- checkpoint recovery for `/crawl_site`
- checkpoint recovery for `/chain_research`
- checkpoint recovery for matching `/ask` reruns
- forced-close checkpoint recovery from `.tmp` files
- soft-stop flag with `/finish`

### `/ask` improvements

- capped gap processing per cycle
- deferred gap queue
- repeat-aware prioritization
- persistent gap-routing memory across separate runs
- local-first confidence-gated routing
- preservation of partial offline evidence
- web escalation only when local signal remains weak or repeated

### Runtime efficiency

- shared AsyncOpenAI client reused across hot paths
- shared HTTP clients for SearXNG and scraper traffic
- hardware-aware runtime profiles:
  - `low-memory`
  - `balanced`
  - `max-recall`
- cheap search-result prefilter before scrape/summarize
- tighter low-memory HTML/PDF ingest budgets

### Maintenance

- duplicate cleanup tool
- metadata backfill tool for existing Chroma collections

## Known limitations

### `/ask`

- gap routing is heuristic-based and should be validated on target data
- automatic resume expects the same topic/question/settings key
- `resume_from` remains useful when you want to resume a draft manually or move work between machines

### Validation

- runtime behavior depends on target machine services:
  - Chroma installation
  - local LLM server
  - SearXNG
  - Discord token and network access
- local code edits can be statically checked, but real end-to-end validation must happen on deployment machine
- deployment and runtime validation steps are documented in `docs/deployment.md`

## Recommended next work

1. Add PDF preflight so large/low-text PDFs can be skipped or downgraded before expensive parsing
2. Add transient caches for search responses, planner outputs, and source probes
3. Add storage pruning and cold-data policies for long-running collections
