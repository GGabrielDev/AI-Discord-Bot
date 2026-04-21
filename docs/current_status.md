# Current Status

## Summary

Project is in a usable but still evolving state. Core research, crawl, and query flows exist and are integrated. The recent low-end optimization wave is now landed alongside checkpoint recovery and local-first gap handling.

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
  - adaptive `raw` chunks for high-value sources under profile-aware caps
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
- profile-aware search-result prefilter before scrape/summarize
- lightweight-first PDF triage with optional Marker escalation for text-poor PDFs
- transient in-process caches for search responses, planner work, query expansion, and gap extraction
- adaptive raw retention budgets tied to the active runtime profile
- runtime telemetry summaries for cache hits, search keep/reject counts, route decisions, and top sources

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

1. Add storage pruning and cold-data policies for long-running collections
2. Promote runtime telemetry beyond console summaries when long-run observability is needed
3. Validate profile and gap-routing thresholds on more target-machine datasets
