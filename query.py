import argparse
import asyncio
from storage.vectordb import VectorDB
from llm.client import LocalLLM

async def answer_question(topic: str, question: str, num_results: int, show_sources: bool):
    db = VectorDB(collection_name=topic)
    llm = LocalLLM()

    print(f"\n🔍 Searching database for: '{question}'...")
    
    # 1. Retrieve the most relevant chunks
    context_chunks = db.search(question, n_results=num_results)
    
    if not context_chunks:
        print("⚠️ No relevant information found in the database. Have you run the researcher for this topic yet?")
        return

    context_text = "\n\n".join([f"--- Source Chunk ---\n{c}" for c in context_chunks])

    if show_sources:
        print("\n=== RAW CONTEXT PULLED FROM DATABASE ===")
        print(context_text)
        print("========================================\n")

    # 2. Build the RAG Prompt
    system_prompt = (
        "You are a specialized research assistant. Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, state that you don't have enough information based on the research. "
        "Keep your answer technical and concise."
    )
    
    user_prompt = f"CONTEXT FROM RESEARCH:\n{context_text}\n\nUSER QUESTION: {question}"

    # 3. Generate Answer
    print("🧠 AI is analyzing research data...")
    answer = await llm.generate_text(system_prompt, user_prompt)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("====================\n")

def main():
    parser = argparse.ArgumentParser(description="Query your local research database.")
    
    # Flags
    parser.add_argument("question", type=str, help="The question you want to ask your research data.")
    parser.add_argument("--topic", type=str, required=True, help="The collection/topic name to query.")
    parser.add_argument("--results", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    parser.add_argument("--sources", action="store_true", help="Show the raw text chunks used to generate the answer.")

    args = parser.parse_args()

    asyncio.run(answer_question(args.topic, args.question, args.results, args.sources))

if __name__ == "__main__":
    main()
