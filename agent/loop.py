import asyncio
import hashlib
import time
from llm.client import LocalLLM
from agent.planner import generate_search_queries, evaluate_and_replan
from agent.summarizer import summarize_page
from tools.search import get_search_results
from tools.scraper import scrape_text_from_url
from storage.vectordb import VectorDB

import re as re_module

def chunk_text(text: str, target_size: int = 400, overlap_sentences: int = 2) -> list[str]:
    """Semantic chunker: splits on paragraph boundaries with sentence-level overlap.
    
    1. Split on paragraph boundaries (double newline)
    2. Merge small paragraphs with adjacent ones
    3. Split oversized paragraphs at sentence boundaries
    4. Add overlap — each chunk includes the last N sentences of the previous chunk
    
    Args:
        text: The text to chunk
        target_size: Target chunk size in words
        overlap_sentences: Number of trailing sentences to carry into the next chunk
    """
    # Split into paragraphs
    paragraphs = re_module.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Split oversized paragraphs at sentence boundaries
    sections = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= target_size:
            sections.append(para)
        else:
            # Split at sentence boundaries
            sentences = re_module.split(r'(?<=[.!?])\s+', para)
            current = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > target_size and current:
                    sections.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += sent_len
            if current:
                sections.append(" ".join(current))
    
    # Merge small sections with neighbors (< 80 words)
    merged = []
    buffer = ""
    for section in sections:
        if buffer:
            combined = buffer + "\n\n" + section
            if len(combined.split()) <= target_size:
                buffer = combined
                continue
            else:
                merged.append(buffer)
                buffer = section
        else:
            buffer = section
        
        if len(buffer.split()) >= 80:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    
    if not merged:
        # Fallback: word-based split if nothing else works
        words = text.split()
        return [" ".join(words[i:i + target_size]) for i in range(0, len(words), target_size)]
    
    # Add overlap: carry trailing sentences from previous chunk into next
    if overlap_sentences > 0 and len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_sentences = re_module.split(r'(?<=[.!?])\s+', merged[i - 1])
            overlap = prev_sentences[-overlap_sentences:] if len(prev_sentences) >= overlap_sentences else prev_sentences
            overlapped.append(" ".join(overlap) + "\n\n" + merged[i])
        return overlapped
    
    return merged

def content_hash(text: str) -> str:
    """Generate a short hash of content to detect duplicates."""
    return hashlib.md5(text[:1000].encode()).hexdigest()

async def run_autonomous_loop(subject, collection_name, max_iterations=3, depth=3, log_func=None):
    # This helper sends text to Discord if a log_func is provided, 
    # otherwise it just prints to the terminal.
    async def report(msg):
        print(msg)
        if log_func:
            await log_func(msg)

    await report(f"📡 **RECONNAISSANCE STARTING:** `{subject}`")
    db = VectorDB(collection_name=collection_name)
    llm = LocalLLM()
    seen_urls = set()
    seen_hashes = set()  # Content-level deduplication
    
    current_queries = await generate_search_queries(subject, num_queries=3)
    
    for iteration in range(1, max_iterations + 1):
        await report(f"\n🔄 **ITERATION {iteration}/{max_iterations}**")
        
        for query in current_queries:
            await report(f"🔎 *Searching:* `{query}`")
            results = get_search_results(query, max_results=depth)
            
            for res in results:
                url = res["url"]
                if url in seen_urls: continue
                
                await report(f"🌐 *Scraping:* <{url}>")
                text = await scrape_text_from_url(url)
                
                if len(text) > 300:
                    # Content-level deduplication
                    text_hash = content_hash(text)
                    if text_hash in seen_hashes:
                        await report(f"⏭️ *Duplicate content detected, skipping.*")
                        seen_urls.add(url)
                        continue
                    
                    # === SUMMARIZE: The LLM actually reads the content ===
                    await report(f"🧠 *Summarizing content...*")
                    summary = await summarize_page(text, subject, url)
                    
                    # Store the LLM summary as primary chunks
                    summary_chunks = chunk_text(summary)
                    db.add_chunks(summary_chunks, url)
                    
                    seen_urls.add(url)
                    seen_hashes.add(text_hash)
                    await report(f"📥 *Stored {len(summary_chunks)} analyzed chunks in memory.*")
                
                await asyncio.sleep(2) # Non-blocking sleep
        
        # === THE EVALUATION STEP (was previously a stub) ===
        if iteration < max_iterations:
            await report("🧠 *AI is evaluating current knowledge and identifying gaps...*")
            
            # 1. Sample what we've collected so far
            sample = db.get_sample(n_samples=15)
            stats = db.get_collection_stats()
            
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

    # Final stats
    final_stats = db.get_collection_stats()
    await report(f"\n📊 **Final: {final_stats['total_chunks']} chunks from {final_stats['unique_sources']} sources**")
    
    return len(seen_urls)
