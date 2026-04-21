import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# --- Discord Settings ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("CRITICAL: DISCORD_TOKEN is missing from your .env file.")

# --- LLM Settings ---
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key-required")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-model")
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "131072"))  # 128K for Gemma 4 E4B
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", min(16384, int(LLM_CONTEXT_WINDOW * 0.125))))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))  # seconds

# --- Context Budgeting ---
# Derived from the hardware context window. 
# We estimate ~0.75 words per token (safe for technical text) 
# and reserve 20% for system prompts/output generation headroom.
SAFE_WORD_BUDGET = int(LLM_CONTEXT_WINDOW * 0.75 * 0.8)


# --- Search Settings ---
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "20"))
HTTP_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "10"))
HTTP_KEEPALIVE_EXPIRY = int(os.getenv("HTTP_KEEPALIVE_EXPIRY", "30"))

# --- Runtime Profiles ---
RESOURCE_PROFILE = os.getenv("RESOURCE_PROFILE", "low-memory").strip().lower()
_RESOURCE_PROFILES = {
    "low-memory": {
        "search_candidate_multiplier": 2,
        "max_sources_per_query": 2,
        "search_prefilter_min_score": 1.5,
        "summarizer_words_per_call_ratio": 0.45,
        "max_html_mb": 6,
        "max_pdf_mb": 40,
    },
    "balanced": {
        "search_candidate_multiplier": 3,
        "max_sources_per_query": 3,
        "search_prefilter_min_score": 1.0,
        "summarizer_words_per_call_ratio": 0.6,
        "max_html_mb": 10,
        "max_pdf_mb": 80,
    },
    "max-recall": {
        "search_candidate_multiplier": 4,
        "max_sources_per_query": 5,
        "search_prefilter_min_score": 0.5,
        "summarizer_words_per_call_ratio": 0.7,
        "max_html_mb": 12,
        "max_pdf_mb": 150,
    },
}

if RESOURCE_PROFILE not in _RESOURCE_PROFILES:
    RESOURCE_PROFILE = "low-memory"

_ACTIVE_PROFILE = _RESOURCE_PROFILES[RESOURCE_PROFILE]
SEARCH_CANDIDATE_MULTIPLIER = _ACTIVE_PROFILE["search_candidate_multiplier"]
MAX_SOURCES_PER_QUERY = _ACTIVE_PROFILE["max_sources_per_query"]
SEARCH_PREFILTER_MIN_SCORE = _ACTIVE_PROFILE["search_prefilter_min_score"]
SUMMARIZER_WORDS_PER_CALL_RATIO = _ACTIVE_PROFILE["summarizer_words_per_call_ratio"]
SCRAPER_MAX_HTML_SIZE = _ACTIVE_PROFILE["max_html_mb"] * 1024 * 1024
SCRAPER_MAX_PDF_SIZE = _ACTIVE_PROFILE["max_pdf_mb"] * 1024 * 1024

# --- Storage Settings ---
# Defaults to a folder named 'chroma_data' in your project root
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
