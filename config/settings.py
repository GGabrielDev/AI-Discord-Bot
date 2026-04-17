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
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", min(16384, int(LLM_CONTEXT_WINDOW * 0.125))))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "131072"))  # 128K for Gemma 4 E4B
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))  # seconds

# --- Context Budgeting ---
# Derived from the hardware context window. 
# We estimate ~0.75 words per token (safe for technical text) 
# and reserve 20% for system prompts/output generation headroom.
SAFE_WORD_BUDGET = int(LLM_CONTEXT_WINDOW * 0.75 * 0.8)


# --- Search Settings ---
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")

# --- Storage Settings ---
# Defaults to a folder named 'chroma_data' in your project root
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
