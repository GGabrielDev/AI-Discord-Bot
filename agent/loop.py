import asyncio
import time
from llm.client import LocalLLM
from agent.planner import generate_search_queries
from tools.search import get_search_results
from tools.scraper import scrape_text_from_url
from storage.vectordb import VectorDB

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

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
                text = scrape_text_from_url(url)
                
                if len(text) > 300:
                    chunks = chunk_text(text)
                    db.add_chunks(chunks, url)
                    seen_urls.add(url)
                    await report(f"📥 *Stored {len(chunks)} chunks in memory.*")
                
                await asyncio.sleep(2) # Non-blocking sleep
        
        if iteration < max_iterations:
            await report("🧠 *AI is evaluating current knowledge...*")
            # ... (rest of the evaluation logic) ...
            await report(f"🎯 *New targets identified:* {current_queries}")

    return len(seen_urls)
