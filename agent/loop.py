"""Autonomous Research Loop — The Core Brain of the Agent.

This module orchestrates the full research pipeline:
1. Generate search queries via the LLM
2. Search the web via SearXNG
3. Scrape and parse each result (HTML and PDF)
4. Summarize content via the LLM
5. Store chunks in ChromaDB + write human-readable Markdown to disk
6. Evaluate knowledge gaps and replan for the next iteration

Crash Resilience:
    The loop saves its state to a JSON checkpoint after every significant action
    (URL scraped, iteration replanned). If the bot crashes mid-research, the next
    invocation automatically resumes from exactly where it left off.
"""

import asyncio
import hashlib
import time
from llm.client import LocalLLM
from agent.planner import generate_search_queries, evaluate_and_replan
from agent.summarizer import summarize_page, compress_raw_text, chunk_text
from config.settings import LLM_MODEL_NAME, SAFE_WORD_BUDGET
from agent.wiki_builder import store_article
from agent.checkpoint import save_checkpoint, load_checkpoint, delete_checkpoint, check_soft_stop
from tools.search import get_search_results
from tools.scraper import scrape_text_from_url
from storage.vectordb import VectorDB

import re as re_module


def content_hash(text: str) -> str:
    """Generate a short hash of content to detect duplicates."""
    return hashlib.md5(text[:1000].encode()).hexdigest()


async def run_autonomous_loop(subject, topic, max_iterations=3, depth=3, log_func=None):
    """Executes the autonomous research pipeline with checkpoint-based crash resilience.
    
    On startup, checks for an existing checkpoint file. If one exists, the loop
    resumes from exactly where it left off — same iteration, same query index,
    with all previously visited URLs and content hashes pre-loaded.
    
    On clean completion, the checkpoint file is deleted.
    """
    # This helper sends text to Discord if a log_func is provided, 
    # otherwise it just prints to the terminal.
    async def report(msg, is_sub_step=False):
        print(msg)
        if log_func:
            await log_func(msg, is_sub_step)

    # --- Checkpoint Recovery ---
    checkpoint = load_checkpoint(subject)
    if checkpoint and checkpoint.get("status") == "in_progress":
        topic = checkpoint.get("topic", topic)
        max_iterations = checkpoint.get("max_iterations", max_iterations)
        depth = checkpoint.get("depth", depth)

    db = VectorDB(collection_name=topic)
    llm = LocalLLM()
    
    if checkpoint and checkpoint.get("status") == "in_progress":
        # RESUME from saved state
        seen_urls = checkpoint["seen_urls"]
        seen_hashes = checkpoint["seen_hashes"]
        start_iteration = checkpoint["current_iteration"]
        start_query_index = checkpoint["current_query_index"]
        current_queries = checkpoint["current_queries"]
        
        await report(
            f"⚡ **RESUMING INTERRUPTED SESSION** — "
            f"Picking up at iteration {start_iteration}/{max_iterations}, "
            f"{len(seen_urls)} sources already processed. "
            f"*(Previous session was interrupted. All prior work has been preserved.)*"
        )
    else:
        # FRESH start
        seen_urls = set()
        seen_hashes = set()
        start_iteration = 1
        start_query_index = 0
        current_queries = await generate_search_queries(subject, num_queries=3)
        
        await report(f"📡 **RECONNAISSANCE STARTING:** `{subject}`")
        
        # Save initial checkpoint
        save_checkpoint(
            subject=subject, topic=topic,
            max_iterations=max_iterations, depth=depth,
            current_iteration=1, current_query_index=0,
            current_queries=current_queries,
            seen_urls=seen_urls, seen_hashes=seen_hashes
        )
    
    for iteration in range(start_iteration, max_iterations + 1):
        # Soft stop interrupt check
        if check_soft_stop():
            await report("🛑 **Soft Stop Acknowledged:** Halting loops, finalizing data...")
            break
            
        await report(f"\n🔄 **ITERATION {iteration}/{max_iterations}**")
        
        # Determine where to start within this iteration's queries
        query_start = start_query_index if iteration == start_iteration else 0
        
        for q_idx, query in enumerate(current_queries):
            # Skip queries we already completed in a previous session
            if q_idx < query_start:
                continue
                
            await report(f"🔎 *Searching:* `{query}`")
            results = await get_search_results(query, max_results=depth)
            
            for res in results:
                url = res["url"]
                # 1. Session-level safeguard (Prevent re-processing in same recursive loop)
                if url in seen_urls: continue
                
                # 2. Persistent Cross-Session Memory Check
                if await db.has_source(url):
                    await report(f"📦 **Cache Hit:** Surgical probe of known source <{url}>", is_sub_step=False)
                    # Pull chunks for this specific source
                    cached_chunks = await db.get_chunks_by_source(url)
                    
                    # Surgical Evaluation: Can we solve the current query with what we already have for this URL?
                    system_prompt = (
                        "You are an internal data evaluator. You are given a RESEARCH QUERY and chunks from a specific website we have previously analyzed.\n\n"
                        "Task: Determine if the answer to the query exists in these specific chunks.\n"
                        "- If YES, provide the technical answer and set 'found' to true.\n"
                        "- If NO, set 'found' to false.\n\n"
                        "Return ONLY JSON: {\"found\": bool, \"answer\": \"string\"}"
                    )
                    # Context limit for surgical probe: Dynamic based on budget
                    # Surgical probes are targeted, so we allow up to 40% of the budget for the context
                    max_probe_words = int(SAFE_WORD_BUDGET * 0.4)
                    current_words = 0
                    probe_chunks = []
                    for d, m in cached_chunks:
                        chunk_words = len(d.split())
                        if current_words + chunk_words > max_probe_words:
                            break
                        probe_chunks.append(d)
                        current_words += chunk_words
                    
                    probe_context = "\n\n".join(probe_chunks)
                    user_prompt = f"QUERY: {query}\n\nCACHED DATA FROM {url}:\n{probe_context}"
                    
                    probe_result = await llm.generate_json(system_prompt, user_prompt)
                    
                    if probe_result.get("found"):
                        await report(f"✅ **Surgical SIP Success:** Answer extracted from cache.", is_sub_step=True)
                        seen_urls.add(url)
                        # We still save the checkpoint so we don't re-probe this URL if we crash
                        save_checkpoint(
                            subject=subject, topic=topic,
                            max_iterations=max_iterations, depth=depth,
                            current_iteration=iteration, current_query_index=q_idx,
                            current_queries=current_queries,
                            seen_urls=seen_urls, seen_hashes=seen_hashes
                        )
                        continue
                    else:
                        await report(f"🔍 *Cache insufficient for current query. Proceeding to fresh scrape...*", is_sub_step=True)
                
                await report(f"🌐 **Processing:** <{url}>", is_sub_step=False)
                
                # Pass a closure down into scraper and summarizer that edits the dashboard message
                async def step_log(msg):
                    await report(msg, is_sub_step=True)
                
                text = await scrape_text_from_url(url, log_func=step_log)
                
                if len(text) > 300:
                    # Content-level deduplication
                    text_hash = content_hash(text)
                    if text_hash in seen_hashes:
                        await step_log(f"⏭️ *Duplicate content detected, skipping.*")
                        seen_urls.add(url)
                        continue
                    
                    # === SUMMARIZE: The LLM actually reads the content ===
                    await step_log(f"🧠 *Summarizing content...*")
                    summary = await summarize_page(text, subject, url, log_func=step_log)
                    
                    # Store the LLM summary as primary chunks
                    summary_chunks = chunk_text(summary)
                    await db.add_chunks(summary_chunks, url, chunk_type="summary")
                    
                    # Store the raw text to preserve specific granular details (Dual-Ingestion)
                    compressed_raw = compress_raw_text(text)
                    raw_chunks = chunk_text(compressed_raw)
                    await db.add_chunks(raw_chunks, url, chunk_type="raw")
                    
                    # === WIKI BUILDER: Save human-readable markdown to disk ===
                    store_article(topic, url, summary)
                    
                    seen_urls.add(url)
                    seen_hashes.add(text_hash)
                    await step_log(f"✅ *Success: Stored {len(summary_chunks)} analyzed chunks in memory.*")
                    
                    # === CHECKPOINT: Save state after each successfully processed URL ===
                    save_checkpoint(
                        subject=subject, topic=topic,
                        max_iterations=max_iterations, depth=depth,
                        current_iteration=iteration, current_query_index=q_idx,
                        current_queries=current_queries,
                        seen_urls=seen_urls, seen_hashes=seen_hashes
                    )
                
                await asyncio.sleep(2) # Non-blocking sleep
        
        # === THE EVALUATION STEP ===
        if iteration < max_iterations:
            await report("🧠 *AI is evaluating current knowledge and identifying gaps...*")
            
            # 1. Sample what we've collected so far
            # We target ~25% of the SAFE_WORD_BUDGET for the evaluation overview (avg chunk ~400 words)
            dynamic_n_samples = max(10, int(SAFE_WORD_BUDGET * 0.25 / 400))
            sample = await db.get_sample(n_samples=dynamic_n_samples)
            stats = await db.get_collection_stats()

            
            await report(f"📊 *Progress: {stats['total_chunks']} chunks from {stats['unique_sources']} sources*")
            
            # 2. Ask the LLM to analyze gaps and generate new queries
            new_queries, gap_analysis = await evaluate_and_replan(
                subject=subject,
                existing_knowledge=sample,
                stats=stats,
                num_queries=3
            )
            
            current_queries = new_queries
            await report(f"💡 *Gap analysis:* {gap_analysis}")
            await report(f"🎯 *New targets identified:* {current_queries}")
            
            # === CHECKPOINT: Save state after replanning with new queries ===
            save_checkpoint(
                subject=subject, topic=topic,
                max_iterations=max_iterations, depth=depth,
                current_iteration=iteration + 1, current_query_index=0,
                current_queries=current_queries,
                seen_urls=seen_urls, seen_hashes=seen_hashes
            )

    # Final stats
    final_stats = await db.get_collection_stats()
    await report(f"\n📊 **Final: {final_stats['total_chunks']} chunks from {final_stats['unique_sources']} sources**")
    
    # === CLEANUP: Research completed successfully, remove checkpoint ===
    delete_checkpoint(subject)
    
    return len(seen_urls)
