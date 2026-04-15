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

# --- Search Settings ---
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")

# --- Storage Settings ---
# Defaults to a folder named 'chroma_data' in your project root
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
