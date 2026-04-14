# Autonomous Research Agent

An offline-first, LLM-driven research assistant designed to run unattended. It takes a seed topic, generates search queries, scrapes web content, and compiles the findings into a local vector database for later retrieval (RAG).

## Core Stack
* **LLM Backend:** `llama.cpp` / `llama-server` (Local)
* **Agent Logic:** Python (via `venv`)
* **Search:** `ddgs` (DuckDuckGo, with future migration to SearXNG)
* **Storage:** `chromadb` (Local Vector Database)

## Directory Architecture

```text
.
├── .env                  # API keys, DB paths, Search URLs (Ignored in Git)
├── main.py               # Entry point and CLI/TTY interface
├── requirements.txt      # Python dependencies
├── config/
│   └── settings.py       # Validates and loads .env variables
├── llm/
│   └── client.py         # AsyncOpenAI wrapper for the local llama-server
├── tools/
│   ├── search.py         # Handles web searching and URL extraction
│   └── scraper.py        # Visits URLs and extracts clean text (stripping HTML)
├── storage/
│   └── vectordb.py       # Handles ChromaDB (chunking text and saving embeddings)
└── agent/
    ├── planner.py        # Generates search queries based on the main topic
    ├── summarizer.py     # Reads scraped text and writes summaries
    └── loop.py           # The infinite autonomous orchestration loop
```
## Implementation Plan

### Phase 1: Foundation & Storage
1. Define the environment variables in `config/settings.py`.
2. Set up `storage/vectordb.py` using ChromaDB to handle document chunking and vector storage.

### Phase 2: The Harvester (Tools)
1. Build `tools/search.py` using `ddgs` to fetch URLs.
2. Build `tools/scraper.py` using robust HTTP clients to extract readable text from raw HTML, ignoring ads and boilerplate.

### Phase 3: LLM Integration
1. Configure `llm/client.py` to route all AI requests to the local `llama-server` instance.
2. Build `agent/planner.py` to prompt the AI to generate targeted search queries.

### Phase 4: The Autonomous Loop
1. Wire all components together in `agent/loop.py`.
2. Implement the evaluation logic: The agent reviews what is currently in ChromaDB, decides what knowledge is missing, and triggers the next iteration.
