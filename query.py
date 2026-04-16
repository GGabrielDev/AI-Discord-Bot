import argparse
import asyncio
import re
from storage.vectordb import VectorDB
from llm.client import LocalLLM
from config.settings import LLM_CONTEXT_WINDOW
from agent.checkpoint import check_soft_stop

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

async def extract_gap_queries(llm: LocalLLM, answer_markdown: str) -> list[str]:
    """Isolates the Knowledge Gaps section and politely extracts them into actionable search queries."""
    # Find everything under '## Knowledge Gaps'
    match = re.search(r"## Knowledge Gaps\n(.*?)(?=\n##|\Z)", answer_markdown, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
        
    gaps_text = match.group(1).strip()
    # Check if empty or explicitly states "no gaps"
    if not gaps_text or "none identified" in gaps_text.lower() or "no significant gaps" in gaps_text.lower():
        return []
        
    system_prompt = (
        "You are an AI research extraction tool. Read the provided 'Knowledge Gaps' text. "
        "Formulate highly specific web search queries to find the missing information. "
        "Keep queries concise. Return ONLY a valid JSON object with key 'queries' containing a list of strings. "
        "If the text indicates there are no gaps, return an empty list."
    )
    user_prompt = f"Knowledge Gaps Text:\n{gaps_text}\n\nExtract search queries to solve these gaps."
    
    result = await llm.generate_json(system_prompt, user_prompt)
    if result and "queries" in result:
        return result["queries"]
    return []

async def deep_internal_probe(db: VectorDB, llm: LocalLLM, gap_query: str, mode: str, log_func=None) -> tuple[str, list[str], bool]:
    """Exhaustively searches the local database using multiple semantic variations to resolve a gap before hitting the web.
    
    Returns:
        tuple (found_text, source_urls, is_fully_resolved)
    """
    if mode == "Fast":
        return "", [], False
        
    num_probes = 5 if mode in ["Thorough", "Omniscient"] else 3
    chunks_per_probe = 20 if mode in ["Thorough", "Omniscient"] else 12
    
    async def log(m):
        if log_func: await log_func(m, True)
        
    await log(f"🧪 **SIP Probe initiated:** Decomposing `{gap_query}` into semantic vectors...")
    
    # 1. Generate deep probes
    probe_queries = await _expand_query(llm, gap_query, num_probes - 1)
    
    all_results = []
    for q in probe_queries:
        res = db.search_with_metadata(q, n_results=chunks_per_probe)
        if res: all_results.extend(res)
        
    if not all_results:
        return "", [], False
        
    # 2. Deduplicate and focus on target info
    unique_knowledge = {}
    sources = set()
    for doc, meta in all_results:
        chunk_hash = hash(doc[:150])
        if chunk_hash not in unique_knowledge:
            unique_knowledge[chunk_hash] = doc
            sources.add(meta.get("source", "unknown"))
            
    context = "\n\n".join(list(unique_knowledge.values())[:30]) # Limit to ~30 best chunks for probe analysis
    
    # 3. LLM Evaluator: Can we solve this now?
    system_prompt = (
        "You are a strict knowledge verification engine. You are given a specific KNOWLEDGE GAP and a list of chunks from an internal database. "
        "The data includes both specialized AI summaries and 'Raw Source Data' which contains verbatim text from documents.\n\n"
        "Your task: Determine if the answer to the gap is definitively contained in these chunks.\n\n"
        "Rules:\n"
        "- Scavenge the 'Raw Source Data' for granular evidence: article numbers, specific letters (a, b, c), technical limits, and exact dates.\n"
        "- If the answer IS there, provide the full technical answer and set 'resolved' to true.\n"
        "- If the answer is partially there but key details are missing, return what you have and set 'resolved' to false.\n"
        "- If search found nothing relevant, set 'resolved' to false.\n\n"
        "Return ONLY a valid JSON object:\n"
        "{\"resolved\": bool, \"answer\": \"string or empty\", \"confidence\": 0.0-1.0}"
    )
    user_prompt = f"GAP TO RESOLVE: {gap_query}\n\nINTERNAL DATA:\n{context}"
    
    await log(f"🧠 *Evaluating {len(unique_knowledge)} internal matches for a native solution...*")
    result = await llm.generate_json(system_prompt, user_prompt)
    
    if result and result.get("resolved") and result.get("confidence", 0) > 0.8:
        return result["answer"], list(sources), True
        
    return result.get("answer", "") if result else "", list(sources), False

async def answer_question(topic: str, question: str, mode: str = "Balanced", log_func=None, draft_callback=None, language: str = "English", _current_auto_loop: int = 0, _draft: str = None, _extra_context: str = None) -> str:
    """Answers a question pulling from the vector DB, with true Agentic RAG auto-looping for explicitly identified gaps."""
    # Import here to avoid circular imports if query.py is loaded first
    from agent.loop import run_autonomous_loop
    
    db = VectorDB(collection_name=topic)
    llm = LocalLLM()
    
    # Mode Constraints
    mode_limits = {"Fast": 0, "Balanced": 1, "Thorough": 3, "Omniscient": 999}
    max_auto_loops = mode_limits.get(mode, 1)
    
    # 1. Fetch chunks or use just a tight cluster if Refining Draft
    num_queries = 1 if mode == "Fast" else (5 if mode in ["Thorough", "Omniscient"] else 3)
    chunk_budget = 10 if mode == "Fast" else (60 if mode in ["Thorough", "Omniscient"] else 30)
    
    async def log(msg: str, is_sub_step: bool = False):
        print(msg)
        if log_func:
            await log_func(msg, is_sub_step)
        else:
            print(f"[Query] {msg}")

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
        
    if _extra_context:
        context_text = f"{_extra_context}\n\n{context_text}"
        
    source_list = "\n".join([f"  - {url}" for url in sources_seen])
    
    # 4. Standardized Markdown Synthesis OR Draft Refining
    if _draft:
        system_prompt = (
            "You are an elite, highly technical research analyst refining an intelligence report.\n"
            "Below is your PREVIOUS incomplete draft. You are also given brand NEW CONTEXT from targeted research designed to fill the draft's Knowledge Gaps.\n\n"
            "Your task is to merge the new context into the report organically. Expand the 'Comprehensive Analysis', integrate citations, "
            "and meticulously remove the Knowledge Gaps that the new tracking data has resolved.\n\n"
            "Maintain the EXACT formatting schema:\n"
            "## Executive Summary\n## Comprehensive Analysis\n## Citations\n## Knowledge Gaps\n\n"
            "Rules:\n"
            "- If all gaps are filled, explicitly write 'None identified.' under ## Knowledge Gaps.\n"
            "- Never hallucinate."
        )
        user_prompt = (
            f"=== NEW CONTEXT ===\n{context_text}\n\n"
            f"=== NEW SOURCES ===\n{source_list}\n\n"
            f"=== PREVIOUS DRAFT TO UPDATE ===\n{_draft}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Update the draft using ONLY the new context."
        )
    else:
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
        
        if language.lower() != "english":
            system_prompt += (
                f"\n\nCRITICAL INSTRUCTION: You MUST write your entire final response natively in {language.capitalize()}. "
                f"However, you must act intelligently: preserve proper nouns, mathematical symbology, "
                f"and highly specific technical industry terms in their original form without forcing a translation."
            )
            
        user_prompt = (
            f"CONTEXT FROM RESEARCH:\n"
            f"{context_text}\n\n"
            f"SOURCES CONSULTED:\n{source_list}\n\n"
            f"USER QUESTION: {question}"
        )

    answer = await llm.generate_text(system_prompt, user_prompt, temperature=0.3, max_tokens=8192, timeout_override=600)
    
    # 5. End if Fast mode or max auto-loops hit
    if _current_auto_loop >= max_auto_loops:
        return answer
        
    # Check Soft Stop
    if check_soft_stop():
        async def loc_log(m):
            print(m)
            if log_func: await log_func(m)
        await loc_log("🛑 **Soft Stop Acknowledged:** Halting Agentic gaps auto-loop early. Returning final draft.")
        return answer
        
    # 6. Extract Gaps and Agentic RAG Re-research
    gap_queries = await extract_gap_queries(llm, answer)
    
    if gap_queries:
        # Deliver intermediate draft file so the user can see the gaps before it stalls researching
        if draft_callback:
            await draft_callback(answer, _current_auto_loop)
            
        async def loc_log(m, is_sub_step=False):
            print(m)
            if log_func: await log_func(m, is_sub_step)
            
        await loc_log(f"⚠️ **Knowledge Gaps detected.** Auto-initiating targeted research loop (Iteration {_current_auto_loop + 1}/{max_auto_loops if max_auto_loops < 999 else '∞'})...")
        await loc_log(f"🎯 *Formulating Gap Chain:* {', '.join(gap_queries)}", is_sub_step=True)
        
        for idx, gap_query in enumerate(gap_queries):
            # Soft stop inter-query check
            if check_soft_stop():
                await loc_log("🛑 **Soft Stop Acknowledged:** Gracefully abandoning remaining gap queries.")
                break
                
            await loc_log(f"🚀 *Gap Tracker {idx+1}/{len(gap_queries)}* -> `{gap_query}`")

            # --- TASK: DEEP SEMANTIC INTERNAL PROBING (SIP) ---
            # We try to solve the gap using what we HAVE before hitting the web
            sip_answer, sip_sources, is_resolved = await deep_internal_probe(db, llm, gap_query, mode, log_func=log_func)
            
            if is_resolved:
                await loc_log(f"✅ **SIP MATCH:** High-confidence answer found internally! Skipping web search for this gap.", is_sub_step=True)
                # We inject this "Virtual Research" into the context for the next draft refinement
                # We format it to look like it came from research so the refiner treats it properly
                sip_context = f"--- Internal Deep Probe Resolution (Self-Search) ---\n{sip_answer}"
                answer = await answer_question(
                    topic=topic,
                    question=question,
                    mode="Fast", # Just synthesis
                    log_func=log_func,
                    draft_callback=draft_callback,
                    language=language,
                    _current_auto_loop=_current_auto_loop + 1,
                    _draft=answer, # Use current draft
                    _extra_context=sip_context
                )
                continue # Gap solved!
            
            # If not resolved or partially resolved, we hit the web (Graceful Fallback)
            if sip_answer:
                await loc_log(f"🔸 *Partial internal match found, but doubt remains. Initiating web fallback...*", is_sub_step=True)

            try:
                await run_autonomous_loop(
                    subject=gap_query,
                    topic=topic,
                    max_iterations=1,
                    depth=5,
                    log_func=log_func
                )
            except Exception as e:
                await loc_log(f"🚨 Sub-tracker failed on `{gap_query}`: {e}")
                
        # We researched everything! Now structurally pull that new data and re-evaluate the draft
        await loc_log("♻️ **Draft Refinement phase:** Merging heavily targeted gap data into intelligence report...")
        return await answer_question(
            topic=topic,
            question=question,
            mode=mode,
            log_func=log_func,
            draft_callback=draft_callback,
            language=language,
            _current_auto_loop=_current_auto_loop + 1,
            _draft=answer
        )
    
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
