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
- forced-close checkpoint recovery from `.tmp` files
- soft-stop flag with `/finish`

### `/ask` improvements

- capped gap processing per cycle
- deferred gap queue
- repeat-aware prioritization
- local-first confidence-gated routing
- preservation of partial offline evidence
- web escalation only when local signal remains weak or repeated

### Maintenance

- duplicate cleanup tool
- metadata backfill tool for existing Chroma collections

## Known limitations

### `/ask`

- no automatic crash-resume checkpoint system
- resume depends on uploaded markdown draft via `resume_from`
- gap routing is heuristic-based and should be validated on target data

### Validation

- runtime behavior depends on target machine services:
  - Chroma installation
  - local LLM server
  - SearXNG
  - Discord token and network access
- local code edits can be statically checked, but real end-to-end validation must happen on deployment machine

## Recommended next work

1. Add persistent `/ask` checkpointing
2. Persist gap-state history across separate command invocations
3. Add automated tests around:
   - checkpoint load/save
   - gap queue ordering
   - metadata backfill behavior
4. Add deployment docs for target machine service layout
