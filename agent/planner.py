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


async def evaluate_and_replan(subject: str, existing_knowledge: list[str], stats: dict, num_queries: int = 3) -> tuple[list[str], str]:
    """Reviews what the agent has learned so far and generates NEW targeted queries.
    
    Args:
        subject: The original research topic
        existing_knowledge: Sample chunks of what's already been collected
        stats: Collection stats (chunk count, source count, etc.)
        num_queries: How many new queries to generate
        
    Returns:
        Tuple of (new_queries, gap_analysis_summary)
    """
    llm = LocalLLM()
    
    knowledge_summary = "\n".join([f"- {chunk[:300]}" for chunk in existing_knowledge[:15]])
    
    system_prompt = (
        "You are an expert research evaluator. You are reviewing the progress of an autonomous research agent. "
        "Your job is to:\n"
        "1. Analyze what has been learned so far\n"
        "2. Identify critical GAPS in the knowledge — what's missing?\n"
        "3. Generate NEW search queries that target the missing information\n\n"
        "Return ONLY a valid JSON object with two keys:\n"
        "- 'gap_analysis': A brief 1-2 sentence summary of what's missing\n"
        "- 'queries': A list of new, specific search queries to fill the gaps\n\n"
        "Example: {\"gap_analysis\": \"Missing technical details...\", \"queries\": [\"query 1\", \"query 2\"]}"
    )
    
    user_prompt = (
        f"RESEARCH TOPIC: {subject}\n\n"
        f"CURRENT PROGRESS:\n"
        f"- {stats.get('total_chunks', 0)} text chunks collected from {stats.get('unique_sources', 0)} sources\n\n"
        f"SAMPLE OF COLLECTED KNOWLEDGE:\n{knowledge_summary}\n\n"
        f"Based on this, what critical information is STILL MISSING about '{subject}'? "
        f"Generate {num_queries} NEW search queries that target the gaps. "
        f"Do NOT repeat queries that would find the same information already collected. "
        f"Output strictly valid JSON without markdown formatting."
    )
    
    print(f"[Planner] Evaluating knowledge gaps for: '{subject}'...")
    result = await llm.generate_json(system_prompt, user_prompt)
    
    if result and "queries" in result:
        queries = result["queries"]
        gap_analysis = result.get("gap_analysis", "Gap analysis unavailable.")
        print(f"[Planner] Identified gaps and generated {len(queries)} new queries.")
        return queries, gap_analysis
    else:
        print("[Planner] Failed to parse re-plan. Generating fresh queries as fallback.")
        fallback = await generate_search_queries(subject, num_queries)
        return fallback, "Re-planning failed; using fresh queries."


async def decompose_chain_prompt(prompt: str) -> list[str]:
    """Takes a massive initial prompt, extracts all specific sub-topics/subjects, and deduplicates them.
    
    Returns a list of strings representing highly specific research subjects.
    No artificial limits are placed on the number of extracted topics.
    """
    llm = LocalLLM()
    
    system_prompt = (
        "You are an elite research director. The user has provided a massive, overarching research goal. "
        "Your task is to exhaustively break this master prompt down into distinct, highly specific sub-topics.\n\n"
        "Rules:\n"
        "1. Extract AS MANY distinct sub-topics as necessary to fully cover the user's request. Do not artificially limit the number.\n"
        "2. Ensure zero redundancy. Each sub-topic must target a uniquely different aspect of the prompt.\n"
        "3. Format each sub-topic as a concise noun-phrase or question suitable as an autonomous research subject.\n\n"
        "Return ONLY a valid JSON object with a single key 'sub_topics' containing a list of strings.\n"
        "Example: {\"sub_topics\": [\"Topic 1\", \"Topic 2\", \"Topic 3\"]}"
    )
    
    user_prompt = f"MASTER PROMPT:\n{prompt}\n\nDecompose this prompt into an exhaustive list of non-redundant sub-topics. Output strictly valid JSON."
    
    print(f"[Planner] Decomposing chain prompt: '{prompt[:50]}...'")
    result = await llm.generate_json(system_prompt, user_prompt)
    
    if result and "sub_topics" in result:
        sub_topics = result["sub_topics"]
        print(f"[Planner] Successfully decomposed chain into {len(sub_topics)} sub-topics.")
        return sub_topics
    else:
        print("[Planner] Failed to parse decomposed chain. Using single prompt as fallback.")
        return [prompt]

# Quick manual test
if __name__ == "__main__":
    topic = "The history of TTL logic chips and their modern applications"
    
    # Run the async function
    queries = asyncio.run(generate_search_queries(topic))
    
    print("\n--- Generated Research Plan ---")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
