"""Checkpoint persistence for the autonomous research loop.

This module provides crash-resilient state management for long-running research
sessions. When the bot is interrupted mid-research (blackout, internet outage,
OOM crash, accidental Ctrl+C), the checkpoint system allows seamless resumption
from exactly where the loop left off.

Architecture:
    - State is serialized to a JSON file in the `checkpoints/` directory.
    - Writes are atomic (write to .tmp, then os.rename) to prevent corruption
      if the process dies mid-write.
    - ChromaDB and knowledge_base/ already persist the OUTPUT data to disk.
      This module persists the SESSION STATE (loop position, visited URLs, etc.).
"""

import json
import os
from datetime import datetime

CHECKPOINT_DIR = "checkpoints"


def _checkpoint_path(topic: str) -> str:
    """Returns the filesystem path for a topic's checkpoint file."""
    # Reuse the same sanitization logic as vectordb
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', topic).strip('_')
    if len(safe_name) < 3:
        safe_name = f"topic_{safe_name}" if safe_name else "default"
    return os.path.join(CHECKPOINT_DIR, f"{safe_name}.json")


def save_checkpoint(
    subject: str,
    collection_name: str,
    max_iterations: int,
    depth: int,
    current_iteration: int,
    current_query_index: int,
    current_queries: list[str],
    seen_urls: set[str],
    seen_hashes: set[str],
    status: str = "in_progress"
):
    """Atomically saves the current research loop state to disk.
    
    Uses a write-to-temp-then-rename strategy to prevent JSON corruption
    if the process is killed mid-write (e.g., during a power outage).
    
    Args:
        subject: The original research topic
        collection_name: ChromaDB collection name
        max_iterations: Total planned iterations
        depth: URLs per query
        current_iteration: Which iteration we're currently in (1-indexed)
        current_query_index: Which query within the iteration we're processing
        current_queries: The current set of search queries
        seen_urls: URLs already scraped (will be skipped on resume)
        seen_hashes: Content hashes for deduplication
        status: Either 'in_progress' or 'complete'
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    state = {
        "subject": subject,
        "collection_name": collection_name,
        "max_iterations": max_iterations,
        "depth": depth,
        "current_iteration": current_iteration,
        "current_query_index": current_query_index,
        "current_queries": current_queries,
        "seen_urls": list(seen_urls),
        "seen_hashes": list(seen_hashes),
        "status": status,
        "last_checkpoint": datetime.now().isoformat()
    }
    
    filepath = _checkpoint_path(subject)
    tmp_path = filepath + ".tmp"
    
    # Atomic write: write to .tmp first, then rename.
    # os.rename() is atomic on POSIX systems (Linux), so even if the process
    # is killed between the write and rename, the old checkpoint survives intact.
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    
    os.rename(tmp_path, filepath)
    print(f"[Checkpoint] 💾 State saved (iteration {current_iteration}/{max_iterations}, {len(seen_urls)} URLs processed)")


def load_checkpoint(subject: str) -> dict | None:
    """Loads a checkpoint for the given topic, if one exists.
    
    Returns:
        The checkpoint state dictionary, or None if no checkpoint exists.
    """
    filepath = _checkpoint_path(subject)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Convert lists back to sets for the loop
        state["seen_urls"] = set(state.get("seen_urls", []))
        state["seen_hashes"] = set(state.get("seen_hashes", []))
        
        print(f"[Checkpoint] 📂 Found saved session: iteration {state['current_iteration']}/{state['max_iterations']}, "
              f"{len(state['seen_urls'])} URLs already processed")
        return state
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Checkpoint] ⚠️ Corrupted checkpoint file detected ({e}). Starting fresh.")
        delete_checkpoint(subject)
        return None


def delete_checkpoint(subject: str):
    """Removes the checkpoint file for a topic after successful completion."""
    filepath = _checkpoint_path(subject)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[Checkpoint] 🗑️ Session checkpoint cleaned up (research complete).")
    
    # Also clean up any orphaned .tmp files
    tmp_path = filepath + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
