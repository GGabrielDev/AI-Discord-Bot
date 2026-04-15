import argparse
import asyncio
from storage.vectordb import VectorDB
from llm.client import LocalLLM
from config.settings import LLM_CONTEXT_WINDOW

# Safety ceiling for context fed to the LLM during /ask synthesis.
# Estimated at ~0.75 words per token, with 80% headroom for system prompt + generation.
MAX_CONTEXT_WORDS = int(LLM_CONTEXT_WINDOW * 0.75 * 0.8)

async def _expand_query(llm: LocalLLM, question: str, num_variations: int) -> list[str]:
    """Uses the LLM to generate semantic variations of the original question."""
    if num_variations <= 0:
        return [question]
        
    system_prompt = (
        "You are an AI research assistant. Your task is to take a user's question and generate "
        "alternate semantic search queries that capture the same intent but use different terminology "
        "or synonyms to ensure maximum search recall in a vector database.\n"
        "Rules:\n"
        f"- Output exactly {num_variations} variations.\n"
        "- Respond with ONLY the variations, separated by newlines.\n"
        "- Do not include numbering, bullet points, or any conversational text."
    )
    
    variations_text = await llm.generate_text(system_prompt, f"Original specific question: {question}", temperature=0.7)
    
    # Clean up output
    variations = [v.strip("-*1234567890. ") for v in variations_text.split("\n") if v.strip()]
    
    # Cap to requested amount and append original
    return [question] + variations[:num_variations]

async def answer_question(topic: str, question: str, mode: str = "Balanced", log_func=None):
    """Answers a question pulling from the vector DB, adapting exhaustiveness based on mode."""
    db = VectorDB(collection_name=topic)
    llm = LocalLLM()
    
    async def log(msg: str):
        if log_func:
            await log_func(msg)
        else:
            print(msg)

    # 1. Parameter Matrix Mapping
    if mode == "Fast":
        num_variations = 0
        chunks_per_query = 10
    elif mode == "Thorough":
        num_variations = 4
        chunks_per_query = 20
    else: # Balanced
        num_variations = 2
        chunks_per_query = 15

    await log(f"⏳ **Phase 1: Generating Semantic Variations...** (Target: {num_variations + 1} distinct queries)")
    search_queries = await _expand_query(llm, question, num_variations)
    
    await log(f"⏳ **Phase 2: Rummaging through database...** (Running {len(search_queries)} parallel vector searches)")
    
    # 2. Multi-Query Retrieval
    all_results = []
    for q in search_queries:
        res = db.search_with_metadata(q, n_results=chunks_per_query)
        if res:
            all_results.extend(res)
            
    if not all_results:
        msg = "⚠️ No relevant information found in the database. Have you run the `/research` command for this topic yet?"
        await log(msg)
        return msg

    # 3. Deduplicate
    unique_chunks = {}
    sources_seen = set()
    
    for doc, meta in all_results:
        source = meta.get("source", "unknown")
        # Use a simple hash of the first 100 characters to deduplicate
        chunk_hash = hash(doc[:100])
        if chunk_hash not in unique_chunks:
            unique_chunks[chunk_hash] = (doc, source, meta.get("chunk_type", "unknown"))
            sources_seen.add(source)
            
    await log(f"⏳ **Phase 3: Deep AI analysis in progress...** (Ingesting {len(unique_chunks)} unique knowledge chunks from {len(sources_seen)} sources. This may take up to 40 seconds depending on mode.)")

    # Order context: Summaries highest priority
    context_parts = []
    for chunk_hash, (doc, source, chunk_type) in unique_chunks.items():
        if chunk_type == "summary":
            context_parts.insert(0, f"--- Analyzed Summary (from: {source}) ---\n{doc}")
        else:
            context_parts.append(f"--- Raw Source Data (from: {source}) ---\n{doc}")
            
    context_text = "\n\n".join(context_parts)
    
    # Guard: truncate if the assembled context exceeds the model's budget
    context_words = context_text.split()
    if len(context_words) > MAX_CONTEXT_WORDS:
        await log(f"⚠️ Context exceeds budget ({len(context_words):,} words). Truncating to {MAX_CONTEXT_WORDS:,} words to protect model stability.")
        context_text = " ".join(context_words[:MAX_CONTEXT_WORDS]) + "\n\n[... context truncated for model budget]"
    
    # 4. Standardized Markdown Synthesis (Schema Enforcer)
    system_prompt = (
        "You are an elite, highly technical research analyst generating intelligence reports from raw database extracts.\n"
        "Your sole task is to answer the user's question exhaustively using ONLY the provided context.\n\n"
        "You MUST structure your response into the following EXACT Markdown sections without deviation:\n\n"
        "## Executive Summary\n"
        "(A direct, concise overarching answer to the question.)\n\n"
        "## Comprehensive Analysis\n"
        "(A thorough, highly detailed deep-dive referencing the granular data points in the context.)\n\n"
        "## Citations\n"
        "(Explicitly list which sources support the claims.)\n\n"
        "## Knowledge Gaps\n"
        "(Explicitly list what limits were found in the context or if parts of the question could not be fully answered.)\n\n"
        "Rules:\n"
        "- Never hallucinate or synthesize information outside the context.\n"
        "- If sources contradict, explicitly detail the contradiction.\n"
        "- Keep the formatting perfectly clean Markdown."
    )
    
    source_list = "\n".join([f"  - {url}" for url in sources_seen])
    user_prompt = (
        f"CONTEXT FROM RESEARCH:\n"
        f"{context_text}\n\n"
        f"SOURCES CONSULTED:\n{source_list}\n\n"
        f"USER QUESTION: {question}"
    )

    answer = await llm.generate_text(system_prompt, user_prompt, temperature=0.3) # 0.3 for high analytical precision
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Query your local research database via CLI.")
    parser.add_argument("question", type=str, help="The question you want to ask your research data.")
    parser.add_argument("--topic", type=str, required=True, help="The collection/topic name to query.")
    parser.add_argument("--mode", type=str, default="Balanced", choices=["Fast", "Balanced", "Thorough"])
    args = parser.parse_args()

    # Create dummy async runner mapped back to prints
    async def run():
        answer = await answer_question(args.topic, args.question, mode=args.mode)
        print("\n\n" + answer)
    asyncio.run(run())

if __name__ == "__main__":
    main()
