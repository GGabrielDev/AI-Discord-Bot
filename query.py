import argparse
import asyncio
from storage.vectordb import VectorDB
from llm.client import LocalLLM

async def answer_question(topic: str, question: str, num_results: int = 10, show_sources: bool = False):
    db = VectorDB(collection_name=topic)
    llm = LocalLLM()

    print(f"\n🔍 Searching database for: '{question}'...")
    
    # 1. Retrieve the most relevant chunks (prefer summaries over raw)
    context_results = db.search_with_metadata(question, n_results=num_results)
    
    if not context_results:
        msg = "⚠️ No relevant information found in the database. Have you run the researcher for this topic yet?"
        print(msg)
        return msg

    # Separate summaries and raw chunks, prioritize summaries
    summaries = []
    raw_chunks = []
    sources_seen = set()
    
    for doc, meta in context_results:
        source = meta.get("source", "unknown")
        chunk_type = meta.get("chunk_type", "unknown")
        
        if chunk_type == "summary":
            summaries.append((doc, source))
        else:
            raw_chunks.append((doc, source))
        sources_seen.add(source)
    
    # Build context: summaries first, then raw as supplement
    context_parts = []
    for doc, source in summaries:
        context_parts.append(f"--- Analyzed Summary (from: {source}) ---\n{doc}")
    for doc, source in raw_chunks:
        context_parts.append(f"--- Raw Source Data (from: {source}) ---\n{doc}")
    
    context_text = "\n\n".join(context_parts)

    if show_sources:
        print("\n=== RAW CONTEXT PULLED FROM DATABASE ===")
        print(context_text)
        print("========================================\n")

    # 2. Build the enhanced RAG Prompt
    system_prompt = (
        "You are a specialized research assistant analyzing data from multiple sources. "
        "Answer the user's question using ONLY the provided context.\n\n"
        "Rules:\n"
        "- Cross-reference information across different sources when possible\n"
        "- If sources contradict each other, explicitly note the contradiction\n"
        "- Cite which source(s) support each claim when relevant\n"
        "- If the answer isn't fully covered by the context, clearly state what is known "
        "and what information is missing\n"
        "- Be thorough and technical — the user wants depth, not surface-level answers\n"
        "- Structure your response with clear sections if the answer is complex"
    )
    
    source_list = "\n".join([f"  - {url}" for url in sources_seen])
    user_prompt = (
        f"CONTEXT FROM RESEARCH ({len(context_parts)} chunks from {len(sources_seen)} sources):\n"
        f"{context_text}\n\n"
        f"SOURCES CONSULTED:\n{source_list}\n\n"
        f"USER QUESTION: {question}"
    )

    # 3. Generate Answer
    print("🧠 AI is analyzing research data...")
    answer = await llm.generate_text(system_prompt, user_prompt, temperature=0.5)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("====================\n")
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Query your local research database.")
    
    # Flags
    parser.add_argument("question", type=str, help="The question you want to ask your research data.")
    parser.add_argument("--topic", type=str, required=True, help="The collection/topic name to query.")
    parser.add_argument("--results", type=int, default=10, help="Number of chunks to retrieve (default: 10).")
    parser.add_argument("--sources", action="store_true", help="Show the raw text chunks used to generate the answer.")

    args = parser.parse_args()

    asyncio.run(answer_question(args.topic, args.question, args.results, args.sources))

if __name__ == "__main__":
    main()
