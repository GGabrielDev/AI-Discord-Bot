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
SOFT_STOP_FLAG = os.path.join(CHECKPOINT_DIR, "SOFT_STOP.flag")

def request_soft_stop():
    """Drops a flag file to request a graceful exit of currently running loops."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(SOFT_STOP_FLAG, "w", encoding="utf-8") as f:
        f.write("STOP_REQUESTED")
    print("[Checkpoint] 🛑 Soft stop requested by user.")

def check_soft_stop() -> bool:
    """Checks if a soft stop was requested. If so, consumes the flag and returns True."""
    if os.path.exists(SOFT_STOP_FLAG):
        os.remove(SOFT_STOP_FLAG)
        print("[Checkpoint] 🛑 Soft stop flag detected and consumed. Wrapping up iteration...")
        return True
    return False


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


# --- MACRO CHAIN CHECKPOINTS ---

def _chain_checkpoint_path(prompt: str) -> str:
    """Returns the filesystem path for a chain checkpoint file based on the initial prompt."""
    import hashlib
    # Hash the prompt because it can be massive, use as filename
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    return os.path.join(CHECKPOINT_DIR, f"chain_{prompt_hash}.json")


def save_chain_checkpoint(
    prompt: str,
    original_save_to: str,
    iterations: int,
    depth: int,
    sub_topics: list[str],
    current_topic_index: int,
    status: str = "in_progress"
):
    """Atomically saves the overarching chain loop state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    state = {
        "prompt": prompt,
        "save_to": original_save_to,
        "iterations": iterations,
        "depth": depth,
        "sub_topics": sub_topics,
        "current_topic_index": current_topic_index,
        "status": status,
        "last_checkpoint": datetime.now().isoformat()
    }
    
    filepath = _chain_checkpoint_path(prompt)
    tmp_path = filepath + ".tmp"
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    
    os.rename(tmp_path, filepath)
    print(f"[Chain Checkpoint] 💾 Master chain state saved (Topic {current_topic_index + 1}/{len(sub_topics)})")


def load_chain_checkpoint(prompt: str) -> dict | None:
    """Loads a macro chain checkpoint by the hash of the original prompt."""
    filepath = _chain_checkpoint_path(prompt)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        print(f"[Chain Checkpoint] 📂 Found saved chain: topic {state['current_topic_index'] + 1}/{len(state['sub_topics'])}")
        return state
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Chain Checkpoint] ⚠️ Corrupted chain checkpoint file detected ({e}). Starting fresh.")
        delete_chain_checkpoint(prompt)
        return None


def delete_chain_checkpoint(prompt: str):
    """Removes the chain checkpoint file after successful completion."""
    filepath = _chain_checkpoint_path(prompt)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[Chain Checkpoint] 🗑️ Master chain checkpoint cleaned up (all topics complete).")
    
    tmp_path = filepath + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
