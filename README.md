# Autonomous Research Agent

An offline-first, LLM-driven research assistant designed to run unattended. It takes a seed topic, generates search queries, scrapes web content, and compiles the findings into a local vector database for later retrieval (RAG).

## Core Stack
* **LLM Backend:** `llama.cpp` / `llama-server` (Local) — optimized for `unsloth/gemma-4-E4B-it-GGUF:UD-Q8_K_XL`
* **Agent Logic:** Python (via `venv`)
* **Search:** `SearXNG` (local meta-search engine)
* **Storage:** `chromadb` (Local Vector Database)

## How It Works

The agent runs an **iterative deepening loop**:

1. **Plan** — The LLM generates targeted search queries for the topic
2. **Harvest** — SearXNG finds URLs, the scraper extracts clean text
3. **Analyze** — The LLM reads each scraped page and produces a structured summary extracting key facts, data points, and relationships
4. **Store** — Analyzed summaries are chunked and saved to ChromaDB with source metadata
5. **Evaluate** — The LLM reviews what's been collected, identifies knowledge gaps, and generates NEW queries targeting the missing information
6. **Repeat** — Steps 2-5 repeat for the configured number of iterations, with queries evolving each cycle

This means the agent **gets smarter with each iteration** — it doesn't just repeat the same searches.

## Directory Architecture

```text
.
├── .env                  # API keys, DB paths, Search URLs (Ignored in Git)
├── main.py               # Entry point: Discord bot with slash commands
├── AI_Discord_Bot.py     # Legacy entry point: simple !prompt command
├── requirements.txt      # Python dependencies
├── config/
│   └── settings.py       # Validates and loads .env variables (LLM, search, storage)
├── llm/
│   └── client.py         # Hardened LLM client: retries, timeouts, task temperatures
├── tools/
│   ├── search.py         # SearXNG integration for web searching
│   └── scraper.py        # HTTP scraper: visits URLs, extracts clean text
├── storage/
│   └── vectordb.py       # ChromaDB: chunking, embeddings, search, stats
├── agent/
│   ├── planner.py        # Generates search queries + evaluates knowledge gaps
│   ├── summarizer.py     # LLM-powered text analysis and fact extraction
│   └── loop.py           # The autonomous research orchestration loop
└── query.py              # RAG query interface for /ask command
```

## Capabilities
1. **Iterative Deep Research** — Multi-pass search with LLM-driven query evolution
2. **Knowledge Gap Analysis** — After each iteration, the LLM evaluates collected data and identifies what's missing
3. **LLM-Powered Summarization** — Every scraped page is read and analyzed by the LLM before storage
4. **Semantic Chunking** — Text is split at paragraph/sentence boundaries with overlap, not arbitrary word counts
5. **Content Deduplication** — URL-level and content-hash-level duplicate detection
6. **Smart RAG Retrieval** — Prioritizes analyzed summaries over raw text, cross-references sources
7. **Source Provenance** — Every chunk tracks its origin URL, type, and timestamp
8. **Robust Web Scraping** — Streams content to enforce strict 150MB limits and verifies `Content-Type` immediately, preventing the agent from stalling on ISO files or slow connections.
9. **Native PDF Parsing** — Extracts extremely accurate Markdown from research papers using `marker`.
10. **Markdown Wikipedia** — Automatically generates a readable, locally browsable `knowledge_base/` linking all summarized articles by topic.
11. **High-Recall Multi-Query RAG** — The `/ask` command supports *Fast, Balanced, and Thorough* modes, generating semantic variations of queries to recall up to 60+ chunks perfectly. Answers are formatted via a strict schema and attached as downloadable `.md` files to bypass Discord limits.
12. **Collection Management** — Autocomplete for existing research topics in Discord

## Setup & Installation

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd AI-Discord-Bot
```

**2. Initialize the virtual environment:**
```bash
python -m venv .
```

**3. Install dependencies:**
```bash
./bin/pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a `.env` file in the root directory:
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

**5. Start llama-server:**
```bash
llama-server \
  -m <path-to-model.gguf> \
  --jinja -ngl 99 -c 32768 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080
```

> See [docs/llama_server.md](docs/llama_server.md) for a full configuration guide including RotorQuant KV cache compression, context sizing, and troubleshooting.

**6. Run the bot:**
```bash
python main.py
```

**Note on Version Control:** The `bin/`, `lib/`, `chroma_data/`, and `__pycache__/` directories are excluded via `.gitignore`.
