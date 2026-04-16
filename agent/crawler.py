import asyncio
import hashlib
from collections import deque
from urllib.parse import urlparse
from llm.client import LocalLLM
from agent.summarizer import summarize_page, compress_raw_text, chunk_text
from agent.wiki_builder import store_article
from agent.checkpoint import save_checkpoint, load_checkpoint, delete_checkpoint, check_soft_stop
from tools.scraper import scrape_with_links
from storage.vectordb import VectorDB

def content_hash(text: str) -> str:
    """Generate a short hash of content to detect duplicates."""
    return hashlib.md5(text[:1000].encode()).hexdigest()

def get_domain(url: str) -> str:
    """Extracts the base domain (e.g. conatel.gov.py)."""
    parsed = urlparse(url)
    return str(parsed.netloc)

def is_pdf(url: str) -> bool:
    """Checks if a URL points specifically to a PDF file."""
    return url.lower().split("?")[0].endswith(".pdf")

async def run_focused_crawler(base_url, topic, max_pages=30, max_depth=3, log_func=None):
    """Recursively crawls a single domain while strictly avoiding external sites (except for PDFs)."""
    
    async def report(msg, is_sub_step=False):
        if log_func:
            await log_func(msg, is_sub_step)
        else:
            print(msg)

    db = VectorDB(collection_name=topic)
    llm = LocalLLM()
    origin_domain = get_domain(base_url)
    
    # --- Checkpoint Recovery (Reusing the system from loop.py) ---
    # We use a unique subject name for the crawler
    subject_key = f"crawl_{topic}_{hashlib.md5(base_url.encode()).hexdigest()[:8]}"
    checkpoint = load_checkpoint(subject_key)
    
    if checkpoint and checkpoint.get("status") == "in_progress":
        seen_urls = checkpoint["seen_urls"]
        seen_hashes = checkpoint["seen_hashes"]
        # In crawler, current_queries is our queue
        queue = deque(checkpoint["current_queries"])
        pages_processed = checkpoint["current_iteration"]
        await report(f"⚡ **RESUMING CRAWL:** {len(seen_urls)} pages already stored. Queue size: {len(queue)}")
    else:
        seen_urls = set()
        seen_hashes = set()
        queue = deque([(base_url, 0)]) # (url, current_depth)
        pages_processed = 0
        await report(f"🚀 **CRUISE INITIATED:** Starting focused crawl of `{origin_domain}`")

    while queue and pages_processed < max_pages:
        # Soft stop interrupt check
        if check_soft_stop():
            await report("🛑 **Soft Stop Acknowledged:** Halting crawler, saving state...")
            break

        url, depth = queue.popleft()
        
        if url in seen_urls:
            continue
            
        await report(f"🌐 **Ingesting:** <{url}> (Depth: {depth})", is_sub_step=False)
        
        async def step_log(msg):
            await report(msg, is_sub_step=True)

        # Scrape content and find links
        text, links = await scrape_with_links(url, log_func=step_log)
        seen_urls.add(url)

        if len(text) > 300:
            text_hash = content_hash(text)
            if text_hash not in seen_hashes:
                # 1. Summarize
                await step_log("🧠 *Deep AI analysis in progress...*")
                summary = await summarize_page(text, f"Website content from {origin_domain}", url, log_func=step_log)
                
                # 2. Store (Dual-Ingestion)
                summary_chunks = chunk_text(summary)
                db.add_chunks(summary_chunks, url, chunk_type="summary")
                
                compressed_raw = compress_raw_text(text)
                raw_chunks = chunk_text(compressed_raw)
                db.add_chunks(raw_chunks, url, chunk_type="raw")
                
                # 3. Wiki Builder
                store_article(topic, url, summary)
                
                seen_hashes.add(text_hash)
                pages_processed += 1
                await step_log(f"✅ *Page Ingested ({pages_processed}/{max_pages})*")
            else:
                await step_log("⏭️ *Duplicate content, skipping.*")

        # Link Discovery & Filtering
        if depth < max_depth:
            found_count = 0
            for link in links:
                if link in seen_urls: continue
                
                # Logic: Same domain OR points to a PDF
                link_domain = get_domain(link)
                if link_domain == origin_domain or is_pdf(link):
                    queue.append((link, depth + 1))
                    found_count += 1
            
            if found_count > 0:
                await step_log(f"🔍 *Discovered {found_count} valid internal paths.*")

        # Checkpoint
        save_checkpoint(
            subject=subject_key, topic=topic,
            max_iterations=max_pages, depth=max_depth,
            current_iteration=pages_processed,
            current_query_index=0, # Not used in crawler
            current_queries=list(queue),
            seen_urls=seen_urls, seen_hashes=seen_hashes
        )

        await asyncio.sleep(2) # Politeness delay

    if pages_processed >= max_pages or not queue:
        await report(f"\n📊 **Crawl Complete!** Ingested {pages_processed} pages into `{topic}`.")
        delete_checkpoint(subject_key)
    
    return pages_processed
