import asyncio
import time
from llm.client import LocalLLM
from agent.planner import generate_search_queries
from tools.search import get_search_results
from tools.scraper import scrape_text_from_url
from storage.vectordb import VectorDB

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Breaks massive scraped text into readable chunks."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

async def evaluate_knowledge(topic: str, db: VectorDB, llm: LocalLLM) -> list[str]:
    """Decides what the AI is still missing after the previous round."""
    recent_docs = db.search(topic, n_results=5)
    context = "\n\n".join(recent_docs) if recent_docs else "No data collected yet."
    
    system_prompt = (
        "You are a research director. Based on what we have learned, generate 2 specific search queries to "
        "fill in the gaps. Output ONLY valid JSON: {\"queries\": [\"query 1\", \"query 2\"]}"
    )
    user_prompt = f"Topic: {topic}\nExisting Knowledge: {context}"
    
    result = await llm.generate_json(system_prompt, user_prompt)
    return result.get("queries", [])

async def run_autonomous_loop(topic: str, max_iterations: int = 5):
    print(f"\n🚀 STARTING OVERNIGHT RESEARCH: '{topic}'")
    db = VectorDB(collection_name=topic)
    llm = LocalLLM()
    seen_urls = set()
    
    # Initial Queries
    current_queries = await generate_search_queries(topic, num_queries=3)
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 🔄 ITERATION {iteration}/{max_iterations} ---")
        
        for query in current_queries:
            results = get_search_results(query, max_results=3)
            
            for res in results:
                url = res["url"]
                if url in seen_urls:
                    continue
                
                text = scrape_text_from_url(url)
                if len(text) > 300:
                    chunks = chunk_text(text)
                    db.add_chunks(chunks, url)
                    seen_urls.add(url)
                
                # Polite delay for web etiquette
                time.sleep(3)
        
        if iteration < max_iterations:
            current_queries = await evaluate_knowledge(topic, db, llm)
            print(f"[Loop] New leads discovered: {current_queries}")

    print(f"\n✅ RESEARCH COMPLETE. {len(seen_urls)} sources saved to {topic}.")

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    # For an overnight run, you might set iterations to 10 or 20
    asyncio.run(run_autonomous_loop(topic, max_iterations=3))
