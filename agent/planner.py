import asyncio
from llm.client import LocalLLM

async def generate_search_queries(topic: str, num_queries: int = 3) -> list[str]:
    """Asks the LLM to generate specific search queries for a topic."""
    llm = LocalLLM()
    
    system_prompt = (
        "You are an expert research planner. "
        "Given a broad topic, generate highly specific search engine queries to gather comprehensive information. "
        "Return ONLY a valid JSON object with a single key 'queries' containing a list of strings. "
        "Example: {\"queries\": [\"query 1\", \"query 2\"]}"
    )
    
    user_prompt = f"Topic: {topic}\nGenerate {num_queries} specific search queries. Output strictly valid JSON without markdown formatting."
    
    print(f"[Planner] Asking LLM to plan research for: '{topic}'...")
    result = await llm.generate_json(system_prompt, user_prompt)
    
    if result and "queries" in result:
        queries = result["queries"]
        print(f"[Planner] Generated {len(queries)} queries successfully.")
        return queries
    else:
        print("[Planner] Failed to parse queries. Using fallback.")
        return [topic] # Fallback to just searching the topic itself

# Quick manual test
if __name__ == "__main__":
    topic = "The history of TTL logic chips and their modern applications"
    
    # Run the async function
    queries = asyncio.run(generate_search_queries(topic))
    
    print("\n--- Generated Research Plan ---")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
