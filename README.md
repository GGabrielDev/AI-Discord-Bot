# 🧠 AI Discord Research Agent

> An autonomous research agent that lives inside your Discord server. Give it a topic — it searches the web, scrapes pages, parses PDFs, summarizes everything through a local LLM, and builds a structured knowledge base you can query with natural language.

## Architecture Overview

```
Discord (User Interface)
  ├── /research  →  Autonomous Research Loop
  │     ├── SearXNG (Private Search)
  │     ├── Async Scraper (HTML + PDF)
  │     │     └── Marker PDF Engine (Deep Learning OCR)
  │     ├── LLM Summarizer (Gemma 4 via llama-server)
  │     ├── ChromaDB (Vector Embeddings)
  │     ├── Wiki Builder (Markdown Knowledge Base)
  │     ├── Checkpoint System (Crash Resilience)
  │     ├── Live Dashboard Reporting
  │     └── /chain_research → Macro-Prompt Decomposition
  │
  └── /ask  →  Multi-Query RAG Pipeline
        ├── Semantic Query Expansion
        ├── Vector Similarity Search
        ├── Context Budget Manager
        └── Structured Markdown Reports
```

## Features

### 🔬 Autonomous Research (`/research`)
- **Iterative Deep Dive** — Runs multiple cycles of search → scrape → summarize → evaluate → replan
- **Gap Analysis** — After each iteration, the LLM reviews what it's learned and identifies missing information
- **PDF Intelligence** — Deep learning vision models (Surya OCR) parse complex PDFs including tables, equations, and multi-column layouts with high accuracy
- **Content Deduplication** — MD5 content hashing prevents the same information from being stored twice
- **Crash Resilience** — JSON checkpoint system saves loop state after every URL, enabling seamless resume after blackouts or crashes
- **Live Discord Dashboard** — Deep technical logs (PDF stats, memory chunks, parsing metrics) are streamed natively into Discord via dynamic message editing, completely mimicking the backend console without spamming channels
- **Soft Stop Interrupts** — Use `/finish` at any time to gracefully wind down loops and wrap up active research sessions without corrupting data or discarding partial work.

### 🔗 Chain Research (`/chain_research`)
- **Macro Decomposition** — Feed the bot a massive, unbounded prompt and the LLM will exhaustively slice it into non-redundant, highly focused sub-topics
- **Sequential Agent Looping** — Automatically deploys the autonomous `/research` loop iterativley across every sub-topic seamlessly
- **Macro Resilience** — Chain-level checkpoints securely save progress between sub-topics; a server crash during topic 4 means the agent instantly resumes at topic 4 upon reboot
- **Knowledge Pooling** — Vector data from all chain branches are centralized into a single unified `save_to` database

### 🧠 Knowledge Query & Agentic RAG (`/ask`)
- **Multi-Query Retrieval** — Generates semantic variations of your question to maximize recall across the vector database
- **Four Agentic Reasoning Modes:**
  - **Fast** — Single query, 10 chunks, 0 auto-loops (~5 seconds)
  - **Balanced** — 3 queries, 30 chunks, Max 1 auto-research loop (~15s+)
  - **Thorough** — 5 queries, 60 chunks, Max 3 auto-research loops (~40s+)
  - **Omniscient** — Uncapped gap-seeking. Dynamically spawns autonomous web agent loops to fill its own knowledge gaps indefinitely until the question is perfectly answered or `/finish` is triggered.
- **Iterative Draft Refining** — By feeding the LLM its previous incomplete drafts alongside brand new gap-filling chunks, the agent inherently bypasses the context window limits, generating ever-growing multi-page documents!
- **Native Translation** — Force reports into any language (Spanish, French, etc.) while automatically preserving mathematical symbology and proper nouns
- **Context Budget Protection** — Automatically truncates assembled context to stay within the model's context window
- **Structured Reports** — Outputs standardized Markdown with Executive Summary, Comprehensive Analysis, Citations, and Knowledge Gaps sections
- **Discord Integration** — Large reports are packaged as downloadable `.md` file attachments

### 📚 Knowledge Base
- **Dual Storage** — ChromaDB for semantic vector search + human-readable Markdown files on disk
- **Auto-Generated Index** — Master `index.md` with links to all research articles, organized by topic
- **Full Provenance** — Every chunk tracks its source URL, timestamp, and processing metadata

### ⚡ Checkpoint System (Crash Resilience)
When a research session is interrupted (power outage, network failure, OOM crash), the bot automatically resumes from exactly where it left off:
- Loop position (iteration number, query index) is saved after every significant action
- All visited URLs and content hashes are preserved
- The LLM's replanned queries survive the restart
- Users see a clear Discord notification: *"⚡ Resuming interrupted research..."*
- On successful completion, the checkpoint is automatically cleaned up
- **Chain Checkpoints** — For `/chain_research`, the bot securely hashes the `prompt` to remember which sub-topics have been fully researched


### 🛡️ Hardware-Aware Optimizations
Designed for constrained hardware (tested on AMD BC-250 with 14.75GB unified memory):
- **Two-Tier PDF Extraction** — Lightweight PyMuPDF (~5MB RAM) by default; deep learning Marker OCR (~1.3GB RAM) available as opt-in for complex layouts
- **CPU-Only Processing** — When Marker is enabled, forces PyTorch to CPU to prevent OOM conflicts with llama-server's GPU memory
- **Context Budget Enforcement** — All LLM calls enforce word-count ceilings to prevent context window overflow
- **Fully Async Pipeline** — Scraper, search, and PDF parsing all run without blocking the Discord event loop
- **Token Usage Tracking** — Every LLM call logs prompt/completion token counts with running session totals

---

## Installation

### Prerequisites
- Python 3.12+
- A running [SearXNG](https://docs.searxng.org/) instance for private web search
- A running [llama-server](https://github.com/ggerganov/llama.cpp) with an OpenAI-compatible API (see [docs/llama_server.md](docs/llama_server.md))

### Setup

**1. Clone the repository:**
```bash
git clone https://github.com/GGabrielDev/AI-Discord-Bot.git
cd AI-Discord-Bot
```

**2. Create a Python virtual environment:**
```bash
python -m venv .
source bin/activate
```

**3. Install dependencies:**
```bash
CFLAGS="-Wno-incompatible-pointer-types" pip install -r requirements.txt
```
> **Note:** The `CFLAGS` prefix is required on systems with GCC 15+ (e.g., CachyOS, Arch) to bypass strict pointer-type warnings in Pillow's C extensions. The requirements file also forces CPU-only PyTorch to save ~5GB of unnecessary CUDA binaries.

**Optional — Enable deep learning PDF extraction (Marker):**
```bash
pip install marker-pdf[full]
```
> **⚠️ Warning:** Marker requires ~1.3GB of RAM for model weights plus additional memory for inference. On memory-constrained systems running llama-server concurrently, this **will** cause OOM kills. Only enable if you have sufficient free RAM (8GB+ recommended beyond what llama-server uses). To activate, add `ENABLE_MARKER_PDF=1` to your `.env` file.

**4. Configure environment variables:**

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

> See [docs/llama_server.md](docs/llama_server.md) for a full configuration guide including KV cache compression, context sizing, and troubleshooting.

**6. Run the bot:**
```bash
python main.py
```

**7. Sync slash commands (first run only):**

Send `!sync` in your Discord server to register the `/research` and `/ask` slash commands.

---

## Usage

### Starting a Research Session
```
/research subject:"solid state batteries" iterations:5 depth:3
```
The bot will autonomously:
1. Generate targeted search queries via the LLM
2. Search the web through SearXNG
3. Scrape and parse each result (HTML pages and PDFs)
4. Summarize findings through the LLM
5. Store analyzed chunks in ChromaDB
6. Write human-readable Markdown to `knowledge_base/`
7. Evaluate knowledge gaps and generate new queries for the next iteration
8. Save checkpoint state after every processed URL

### Querying Your Knowledge Base
```
/ask topic:"solid state batteries" question:"What are the main electrolyte materials?" mode:Thorough
```
The bot will:
1. Generate semantic variations of your question
2. Run parallel vector searches across ChromaDB
3. Deduplicate and prioritize results
4. Synthesize a comprehensive Markdown report
5. Deliver it as a downloadable `.md` file attachment

---

## Maintenance

### Resetting the AI Brain
To fully wipe the bot's memory and start fresh, delete all three data directories:
```bash
rm -rf chroma_data/
rm -rf knowledge_base/
rm -rf checkpoints/
```
The bot will automatically generate fresh, blank directories the next time you run a command.

### Clearing a Stuck Checkpoint
If a checkpoint file prevents a fresh start on a specific topic:
```bash
rm -rf checkpoints/
```
This only removes session state — your ChromaDB data and knowledge base files remain intact.

---

## Project Structure

```
AI-Discord-Bot/
├── main.py                  # Discord bot entry point, slash commands
├── query.py                 # Multi-query RAG pipeline (/ask)
├── requirements.txt         # Python dependencies
├── agent/
│   ├── loop.py              # Autonomous research loop (core brain)
│   ├── planner.py           # Search query generation & gap analysis
│   ├── summarizer.py        # LLM-powered content summarization
│   ├── wiki_builder.py      # Markdown knowledge base generator
│   └── checkpoint.py        # Crash-resilient state persistence
├── llm/
│   └── client.py            # Hardened LLM client with retries & token tracking
├── tools/
│   ├── scraper.py           # Async HTML/PDF streaming scraper
│   ├── search.py            # Async SearXNG search integration
│   └── pdf_parser.py        # Marker PDF-to-Markdown engine (cached models)
├── storage/
│   └── vectordb.py          # ChromaDB wrapper with rich metadata
├── config/
│   └── settings.py          # Environment variable loader
├── docs/
│   └── llama_server.md      # llama-server configuration guide
│
├── chroma_data/             # (generated) Vector embeddings
├── knowledge_base/          # (generated) Human-readable Markdown wiki
└── checkpoints/             # (generated) Research session state files
```

**Note on Version Control:** The `bin/`, `lib/`, `chroma_data/`, `knowledge_base/`, `checkpoints/`, and `__pycache__/` directories are excluded via `.gitignore`.
