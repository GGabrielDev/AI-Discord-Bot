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
3. **Store** — Text is chunked and saved to ChromaDB with source metadata
4. **Evaluate** — The LLM reviews what's been collected, identifies knowledge gaps, and generates NEW queries targeting the missing information
5. **Repeat** — Steps 2-4 repeat for the configured number of iterations, with queries evolving each cycle

This means the agent **gets smarter with each iteration** — it doesn't just repeat the same searches.

## Directory Architecture

```text
.
├── .env                  # API keys, DB paths, Search URLs (Ignored in Git)
├── main.py               # Entry point: Discord bot with slash commands
├── AI_Discord_Bot.py     # Legacy entry point: simple !prompt command
├── requirements.txt      # Python dependencies
├── config/
│   └── settings.py       # Validates and loads .env variables
├── llm/
│   └── client.py         # AsyncOpenAI wrapper for the local llama-server
├── tools/
│   ├── search.py         # SearXNG integration for web searching
│   └── scraper.py        # HTTP scraper: visits URLs, extracts clean text
├── storage/
│   └── vectordb.py       # ChromaDB: chunking, embeddings, search, stats
├── agent/
│   ├── planner.py        # Generates search queries + evaluates knowledge gaps
│   ├── summarizer.py     # (Coming soon) LLM-powered text analysis
│   └── loop.py           # The autonomous research orchestration loop
└── query.py              # RAG query interface for /ask command
```

## Capabilities
1. **Iterative Deep Research** — Multi-pass search with LLM-driven query evolution
2. **Knowledge Gap Analysis** — After each iteration, the LLM evaluates collected data and identifies what's missing
3. **Content Deduplication** — URL-level and content-hash-level duplicate detection
4. **Web Scraping** — Robust HTML-to-text extraction with ad/boilerplate stripping
5. **RAG Query** — Ask questions against your local research database
6. **Discord Integration** — `/research` and `/ask` slash commands with live progress updates
7. **Collection Management** — Autocomplete for existing research topics in Discord

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
SEARXNG_URL=http://localhost:8081
CHROMA_DB_PATH=./chroma_data
```

**5. Start llama-server:**
```bash
llama-server -m <path-to-model.gguf> --jinja -ngl 99 --host 0.0.0.0 --port 8080
```

**6. Run the bot:**
```bash
python main.py
```

**Note on Version Control:** The `bin/`, `lib/`, `chroma_data/`, and `__pycache__/` directories are excluded via `.gitignore`.
