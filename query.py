import asyncio
import argparse
import copy
import hashlib
import re
from storage.vectordb import VectorDB
from llm.client import LocalLLM
from config.settings import SAFE_WORD_BUDGET, LLM_MAX_TOKENS
from agent.ask_state import (
    MAX_GAPS_PER_CYCLE,
    advance_gap_cycle,
    build_gap_context,
    build_gap_memory_snapshot,
    clamp,
    dequeue_gap_batch,
    ensure_gap_state,
    explain_gap_route,
    merge_gap_memory,
    quality_from_meta,
    queue_gap_queries,
    record_gap_probe,
    record_gap_web_outcome,
    restore_gap_batch,
    select_gap_route,
    set_gap_route,
    safe_float,
)
from agent.checkpoint import (
    check_soft_stop,
    delete_ask_checkpoint,
    load_ask_checkpoint,
    load_gap_memory,
    save_ask_checkpoint,
    save_gap_memory,
)
from agent.wiki_builder import store_final_report
from runtime_cache import TTLCache
from runtime_telemetry import add as telemetry_add, bump as telemetry_bump, telemetry_session

# Safety ceiling for context fed to the LLM during /ask synthesis.
# We utilize a large portion of the SAFE_WORD_BUDGET to allow the model 
# to 'stretch its legs' with massive context, while reserving room for the R1 <think> block.
MAX_CONTEXT_WORDS = int(SAFE_WORD_BUDGET * 0.85)
_query_expansion_cache = TTLCache(ttl_seconds=300, max_entries=128)
_gap_extraction_cache = TTLCache(ttl_seconds=180, max_entries=128)


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def fit_to_context_budget(system_prompt: str, user_prompt: str, max_words: int) -> tuple[str, str]:
    """Ensures total words across system and user prompts stay within max_words.
    Truncates the user_prompt from the TOP (oldest parts) if necessary,
    while PRESERVING all original newlines and formatting.
    """
    sys_words_est = len(system_prompt.split())
    user_words_est = len(user_prompt.split())
    total_est = sys_words_est + user_words_est
    
    if total_est <= max_words:
        return system_prompt, user_prompt
        
    # We need to truncate strings without using split().join() which destroys newlines.
    # We use a character-based proportional slice as a safe estimate.
    # Words * 6 is a safe character estimate for technical text.
    safe_char_limit = max_words * 6
    sys_chars = len(system_prompt)
    
    available_user_chars = safe_char_limit - sys_chars - 500 # safety margin
    if available_user_chars < 1000: available_user_chars = 1000 # guard rail
    
    if len(user_prompt) > available_user_chars:
        print(f"[Context Shield] Truncating user prompt to ~{available_user_chars} chars (Preserving Structure).")
        # Truncate from the top (oldest content/context) and keep the bottom (new queries/instructions)
        truncated_user = user_prompt[-available_user_chars:]
        return system_prompt, f"[... context truncated for budget ...]\n{truncated_user}"
    
    return system_prompt, user_prompt

async def _expand_query(llm: LocalLLM, question: str, num_variations: int) -> list[str]:
    """Uses the LLM to generate semantic variations of the original question."""
    if num_variations <= 0:
        return [question]

    cache_key = (question.strip(), num_variations)
    cached_variations = _query_expansion_cache.get(cache_key)
    if cached_variations is not None:
        telemetry_bump("cache.hits")
        telemetry_bump("cache.hits.query_expansion")
        print(f"[Query] Cache hit for semantic expansion: '{question}'")
        return cached_variations
    telemetry_bump("cache.misses")
    telemetry_bump("cache.misses.query_expansion")

    async def produce_variations() -> list[str]:
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
        variations = [v.strip("-*1234567890. ") for v in variations_text.split("\n") if v.strip()]
        return [question] + variations[:num_variations]

    return await _query_expansion_cache.get_or_set(cache_key, produce_variations)

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

    cache_key = _text_digest(gaps_text)
    cached_queries = _gap_extraction_cache.get(cache_key)
    if cached_queries is not None:
        telemetry_bump("cache.hits")
        telemetry_bump("cache.hits.gap_extraction")
        print("[Query] Cache hit for gap extraction.")
        return cached_queries
    telemetry_bump("cache.misses")
    telemetry_bump("cache.misses.gap_extraction")

    async def produce_gap_queries() -> list[str]:
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

    return await _gap_extraction_cache.get_or_set(cache_key, produce_gap_queries)

async def deep_internal_probe(db: VectorDB, llm: LocalLLM, gap_query: str, mode: str, log_func=None) -> dict:
    """Exhaustively searches the local database using multiple semantic variations to resolve a gap before hitting the web.
    
    Returns:
        Probe evidence and scoring used to choose local-vs-web routing.
    """
    if mode == "Fast":
        return {
            "answer": "",
            "sources": [],
            "resolved": False,
            "llm_confidence": 0.0,
            "local_score": 0.0,
            "source_count": 0,
            "raw_hits": 0,
            "summary_hits": 0,
            "avg_quality": 0.0,
            "avg_distance": 1.0,
            "total_hits": 0,
            "has_partial_answer": False
        }
        
    # SIP v2 Unified Greed Tiering
    if mode in ["Thorough", "Omniscient"]:
        num_probes = 7
        scout_depth = 30
        scavenge_depth = 15
    else: # Balanced
        num_probes = 4
        scout_depth = 20
        scavenge_depth = 10
    
    async def log(m):
        if log_func: await log_func(m, True)
        
    await log(f"🧪 **SIP Probe (v2) initiated:** Exhaustively mining local data before web fallback...")
    
    # 1. Internal Scout (Summaries)
    probe_queries = await _expand_query(llm, gap_query, num_probes - 1)
    
    scout_results = []
    for q in probe_queries:
        res = await db.search_with_metadata(q, n_results=scout_depth, where={"chunk_type": "summary"}, include_distances=True)
        if res: scout_results.extend(res)
        
    # Identification of top sources for Internal Scavenge
    source_stats = {}
    for _, meta, _ in scout_results:
        src = meta.get("source")
        if src: source_stats[src] = source_stats.get(src, 0) + 1
    
    top_sources = sorted(source_stats.keys(), key=lambda x: source_stats[x], reverse=True)[:5]
    
    # 2. Internal Scavenge (Raw Data from top sources)
    scavenge_results = []
    for src in top_sources:
        res = await db.search_with_metadata(
            gap_query,
            n_results=scavenge_depth,
            where={"$and": [{"source": src}, {"chunk_type": "raw"}]},
            include_distances=True
        )
        if res: scavenge_results.extend(res)

    # 3. Fallback: General internal search if scout found nothing
    if not scout_results and not scavenge_results:
        for q in probe_queries:
            res = await db.search_with_metadata(q, n_results=scout_depth, include_distances=True)
            if res: scavenge_results.extend(res)

    # Deduplicate and stay within Budget-Aware cap
    unique_knowledge = {}
    sources = set()
    raw_hits = 0
    summary_hits = 0
    quality_values = []
    distance_values = []
    
    # Estimate safe chunk count: Each chunk is ~400 words. 
    # We want to leave some headroom for the prompt/system instructions.
    max_chunks = max(10, int(SAFE_WORD_BUDGET / 450)) 
    
    for doc, meta, distance in (scout_results + scavenge_results):
        chunk_hash = hash(doc[:150])
        if chunk_hash not in unique_knowledge:
            unique_knowledge[chunk_hash] = (doc, meta)
            sources.add(meta.get("source", "unknown"))
            if meta.get("chunk_type") == "raw":
                raw_hits += 1
            elif meta.get("chunk_type") == "summary":
                summary_hits += 1
            quality_values.append(quality_from_meta(meta))
            if distance is not None:
                distance_values.append(distance)
        if len(unique_knowledge) >= max_chunks: break # Dynamic Cap
            
    context = "\n\n".join(doc for doc, _ in unique_knowledge.values())
    
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
    llm_confidence = safe_float(result.get("confidence"), 0.0) if result else 0.0
    avg_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
    avg_distance = sum(distance_values) / len(distance_values) if distance_values else 1.0
    source_count = len(sources)
    source_diversity = min(1.0, source_count / 4)
    raw_support = min(1.0, raw_hits / 6)
    summary_support = min(1.0, summary_hits / 6)
    local_score = clamp(
        (llm_confidence * 0.45) +
        (raw_support * 0.2) +
        (summary_support * 0.1) +
        (source_diversity * 0.15) +
        (avg_quality * 0.1)
    )

    answer_text = result.get("answer", "") if result else ""
    return {
        "answer": answer_text,
        "sources": list(sources),
        "resolved": bool(result and result.get("resolved")),
        "llm_confidence": llm_confidence,
        "local_score": local_score,
        "source_count": source_count,
        "raw_hits": raw_hits,
        "summary_hits": summary_hits,
        "avg_quality": avg_quality,
        "avg_distance": avg_distance,
        "total_hits": len(unique_knowledge),
        "has_partial_answer": bool(answer_text.strip())
    }


def _merge_extra_context(existing_context: str | None, new_context: str | None) -> str | None:
    existing_context = existing_context or ""
    new_context = new_context or ""
    if not new_context.strip():
        return existing_context or None
    if not existing_context.strip():
        return new_context
    if new_context in existing_context:
        return existing_context
    return f"{existing_context}\n\n{new_context}"


def _checkpoint_gap_state(gap_state: dict, remaining_batch: list[str]) -> dict:
    if not remaining_batch:
        return gap_state
    checkpoint_gap_state = copy.deepcopy(gap_state)
    restore_gap_batch(checkpoint_gap_state, remaining_batch)
    return checkpoint_gap_state


async def _process_gap_batch(
    *,
    topic: str,
    mode: str,
    no_web: bool,
    gap_batch: list[str],
    draft_text: str,
    extra_context: str | None,
    gap_state: dict,
    current_auto_loop: int,
    db: VectorDB,
    llm: LocalLLM,
    log_func,
    persist_ask_state,
):
    from agent.loop import run_autonomous_loop

    async def loc_log(m, is_sub_step=False):
        print(m)
        if log_func:
            await log_func(m, is_sub_step)

    accumulated_context = extra_context
    for idx, gap_query in enumerate(gap_batch):
        remaining_batch = gap_batch[idx:]
        if check_soft_stop():
            restore_gap_batch(gap_state, remaining_batch)
            persist_ask_state(draft_text, accumulated_context, gap_state)
            await loc_log("🛑 **Soft Stop Acknowledged:** Gracefully abandoning remaining gap queries.")
            return accumulated_context, True

        await loc_log(f"🚀 *Gap Tracker {idx+1}/{len(gap_batch)}* -> `{gap_query}`")
        probe = await deep_internal_probe(db, llm, gap_query, mode, log_func=log_func)
        gap_meta = record_gap_probe(gap_state, gap_query, probe, current_auto_loop)
        route = select_gap_route(gap_meta, probe, no_web)
        telemetry_bump(f"gap.route.{route}")

        sip_context = build_gap_context(gap_query, probe)
        if route == "resolved_local":
            set_gap_route(gap_meta, route, current_auto_loop)
            await loc_log(
                f"✅ **SIP MATCH:** {explain_gap_route(gap_meta, probe, route, no_web)}. Web skipped.",
                is_sub_step=True
            )
        elif route == "partial_local":
            set_gap_route(gap_meta, route, current_auto_loop, 1)
            await loc_log(
                f"🧩 **Partial local evidence retained:** {explain_gap_route(gap_meta, probe, route, no_web)}.",
                is_sub_step=True
            )
        elif route == "blocked_offline":
            set_gap_route(gap_meta, route, current_auto_loop, 1)
            await loc_log(f"⚠️ **Local-Only Mode:** {explain_gap_route(gap_meta, probe, route, no_web)}.", is_sub_step=True)
        elif route == "defer_local":
            set_gap_route(gap_meta, route, current_auto_loop, 1)
            await loc_log(
                f"⏸️ **Deferring local retry:** {explain_gap_route(gap_meta, probe, route, no_web)}.",
                is_sub_step=True
            )
        else:
            await loc_log(
                f"🌐 **Web escalation triggered:** {explain_gap_route(gap_meta, probe, route, no_web)}.",
                is_sub_step=True
            )
            web_sources_added = 0
            web_failed = False
            try:
                web_sources_added = await run_autonomous_loop(
                    subject=gap_query,
                    topic=topic,
                    max_iterations=1,
                    depth=5,
                    log_func=log_func
                )
            except Exception as e:
                web_failed = True
                await loc_log(f"🚨 Sub-tracker failed on `{gap_query}`: {e}")
            telemetry_add("gap.web_sources_added", web_sources_added)
            record_gap_web_outcome(gap_meta, current_auto_loop, web_sources_added, web_failed)

        accumulated_context = _merge_extra_context(accumulated_context, sip_context)
        persist_ask_state(
            draft_text,
            accumulated_context,
            _checkpoint_gap_state(gap_state, gap_batch[idx + 1:])
        )

    return accumulated_context, False

STYLE_CONFIG = {
    "Concise": {
        "description": "Standard high-level intelligence report. Direct, efficient, and technical.",
        "persona": (
            "You are an elite, highly technical research analyst generating intelligence reports from raw database extracts.\n"
            "Your sole task is to answer the user's question exhaustively using ONLY the provided context.\n\n"
            "You MUST structure your response into the following EXACT Markdown sections without deviation:\n\n"
            "## Executive Summary\n"
            "(A direct, concise overarching answer to the question.)\n\n"
            "## Comprehensive Analysis\n"
            "(A thorough deep-dive referencing the granular data points in the context.)\n\n"
            "## Citations\n"
            "(Explicitly list which sources support the claims.)\n\n"
            "## Knowledge Gaps\n"
            "(Explicitly list what limits were found in the context or if parts of the question could not be fully answered.)\n\n"
            "Rules:\n"
            "- Never hallucinate or synthesize information outside the context.\n"
            "- If sources contradict, explicitly detail the contradiction.\n"
            "- Keep the formatting perfectly clean Markdown."
        )
    },
    "Investigative": {
        "description": "Deep-dive forensic analysis. Explores technical nuances, legal contradictions, and edge cases.",
        "persona": (
            "You are a master digital forensic investigator and technical deep-dive specialist. "
            "Your objective is to produce a massive, exhaustive intelligence report that scours for the hidden 'Why' and 'How' behind every piece of data.\n\n"
            "Approach:\n"
            "1. Nuance Scavenging: Look for contradictions between sources and explain them in detail.\n"
            "2. Cross-Layer Analysis: Connect high-level summaries with low-level 'Raw' technical details (article letters, frequency limits, exact code sections).\n"
            "3. Edge-Case Mapping: Highlight legal or technical risks, exceptions, and unique edge cases found in the documentation.\n\n"
            "You MUST structure your response into the following EXACT Markdown sections:\n\n"
            "## Executive Summary\n"
            "(A high-level overview of the most critical findings.)\n\n"
            "## Investigative Deep-Dive\n"
            "(The largest section. Use nested bullet points and tables. Detail contradictions, technical specifics, and granular evidence found in the raw text.)\n\n"
            "## Contextual Implications\n"
            "(What does this information actually MEAN in a real-world scenario? Explain the risks or technical constraints uncovered.)\n\n"
            "## Citations\n"
            "(Explicitly list all sources with specific mentions of which article/page provided the data.)\n\n"
            "## Knowledge Gaps\n"
            "(Explicitly list what limits were found in the context. Be aggressive about identifying gaps for the autonomous loop to target.)\n\n"
            "Rules:\n"
            "- Use ONLY the provided context. Never hallucinate.\n"
            "- Be verbose, technical, and meticulous."
        )
    }
}

async def answer_question(topic: str, question: str, mode: str = "Balanced", style: str = "Concise", 
                      log_func=None, draft_callback=None, language: str = "English", 
                      no_web: bool = False,
                      _current_auto_loop: int = 0, _draft: str = None, _extra_context: str = None,
                      _gap_state: dict | None = None) -> str:
    """Answers a question pulling from the vector DB, with true Agentic RAG auto-looping for explicitly identified gaps."""
    with telemetry_session(
        f"ask:{question}",
        metadata={"topic": topic, "mode": mode, "style": style, "language": language, "local_only": no_web},
    ):
        # Global logging helper for this session
        async def log(msg: str, is_sub_step: bool = False):
            print(msg)
            if log_func:
                await log_func(msg, is_sub_step)
            else:
                print(f"[Query] {msg}")

        resumed_ask = False
        loaded_memory = load_gap_memory(topic)
        gap_state = merge_gap_memory(advance_gap_cycle(ensure_gap_state(_gap_state), _current_auto_loop), loaded_memory)

        if _current_auto_loop == 0 and _draft is None and _gap_state is None:
            ask_checkpoint = load_ask_checkpoint(topic, question, mode, style, language, no_web)
            if ask_checkpoint and ask_checkpoint.get("status") == "in_progress":
                telemetry_bump("ask.resumed")
                resumed_ask = True
                _draft = ask_checkpoint.get("draft")
                _extra_context = ask_checkpoint.get("extra_context") or _extra_context
                gap_state = merge_gap_memory(gap_state, ask_checkpoint.get("gap_state"))
                gap_state = advance_gap_cycle(gap_state, ask_checkpoint.get("current_auto_loop", 0))
                await log("⚡ **Resuming interrupted /ask session.** Restoring saved draft and gap state...")

        db = VectorDB(collection_name=topic)
        llm = LocalLLM()
    
        # Mode Constraints
        mode_limits = {"Fast": 0, "Balanced": 1, "Thorough": 3, "Omniscient": 999}
        max_auto_loops = mode_limits.get(mode, 1)
        if no_web: 
            max_auto_loops = 0

        def persist_gap_history():
            save_gap_memory(topic, build_gap_memory_snapshot(gap_state))

        def persist_ask_state(
            draft_text: str | None,
            extra_context: str | None = None,
            checkpoint_gap_state: dict | None = None
        ):
            save_ask_checkpoint(
                topic=topic,
                question=question,
                mode=mode,
                style=style,
                language=language,
                no_web=no_web,
                current_auto_loop=_current_auto_loop,
                draft=draft_text,
                gap_state=checkpoint_gap_state if checkpoint_gap_state is not None else gap_state,
                extra_context=extra_context
            )
            persist_gap_history()

        def clear_ask_state():
            delete_ask_checkpoint(topic, question, mode, style, language, no_web)
            persist_gap_history()

        def has_pending_gap_queue() -> bool:
            return bool(gap_state.get("order"))

        async def log_retained_gap_checkpoint(stage: str):
            if has_pending_gap_queue():
                await log(
                    f"📦 **{len(gap_state['order'])} deferred knowledge gaps remain queued after {stage}. Checkpoint retained for resume.**"
                )

        if _current_auto_loop == 0:
            persist_ask_state(_draft, _extra_context)

        # --- Resume Shortcut ---
        # If a draft is provided at the start, skip redundant broad searches 
        # and jump straight to gap extraction.
        if _draft and _current_auto_loop == 0:
            if resumed_ask:
                if gap_state["order"]:
                    await log("📦 **Resuming from saved /ask checkpoint.** Continuing pending gap queue...")
                    queued_gap_count = len(gap_state["order"])
                elif _extra_context:
                    await log("📦 **Resuming from saved /ask checkpoint.** Applying saved batch context before continuing...")
                    queued_gap_count = 0
                else:
                    queued_gap_count = 0
            else:
                await log("📦 **Resuming from provided report.** Parsing knowledge gaps...")
                gap_queries = await extract_gap_queries(llm, _draft)
                telemetry_add("gap.queries_total", len(gap_queries))
                gap_state = queue_gap_queries(gap_state, gap_queries)
                queued_gap_count = len(gap_state["order"])

            if queued_gap_count:
                persist_ask_state(_draft, _extra_context)
                gap_batch, deferred_gap_count = dequeue_gap_batch(gap_state)
                await log(
                    f"🎯 **Queued {queued_gap_count} resumed gaps. Processing {len(gap_batch)} now; {deferred_gap_count} deferred.**",
                    is_sub_step=True
                )
                batch_context, batch_interrupted = await _process_gap_batch(
                    topic=topic,
                    mode=mode,
                    no_web=no_web,
                    gap_batch=gap_batch,
                    draft_text=_draft,
                    extra_context=_extra_context,
                    gap_state=gap_state,
                    current_auto_loop=_current_auto_loop,
                    db=db,
                    llm=llm,
                    log_func=log_func,
                    persist_ask_state=persist_ask_state,
                )
                if not batch_interrupted:
                    answer = await answer_question(
                        topic=topic,
                        question=question,
                        mode=mode, 
                        style=style,
                        log_func=log_func,
                        draft_callback=draft_callback,
                        language=language,
                        no_web=no_web,
                        _current_auto_loop=_current_auto_loop + 1,
                        _draft=_draft,
                        _extra_context=batch_context,
                        _gap_state=gap_state
                    )
                    _draft = answer["english"] if isinstance(answer, dict) else answer
                    persist_ask_state(_draft, None)

                if _current_auto_loop == 0:
                    result = await finalize_dual_report(llm, _draft, topic, language, mode, batch_interrupted or check_soft_stop())
                    if not batch_interrupted and not has_pending_gap_queue():
                        clear_ask_state()
                    elif not batch_interrupted:
                        await log_retained_gap_checkpoint("this refinement cycle")
                    return result
                return _draft
            else:
                await log("💡 **Resumed report appears complete.** No new gaps identified.")
                # If no gaps, just proceed to a final synthesis to ensure persona formatting is applied
                # By letting it fall through, the code below will run Phase 1-4.
                # Actually, if we're resuming a finished report, maybe we just return it.
                # But let's let it run once to be safe.
    
        # 1. Parameter Matrix Mapping
        if mode == "Fast":
            num_variations = 0
            scout_chunks = 12
            scavenge_count = 0 # No scavenging for Fast
        elif mode == "Thorough":
            num_variations = 4
            scout_chunks = 40
            scavenge_count = 5 # Scavenge raw from top 5 sources
        elif mode == "Omniscient":
            num_variations = 6
            scout_chunks = 60
            scavenge_count = 8 # Scavenge top 8 sources
        else: # Balanced
            num_variations = 2
            scout_chunks = 25
            scavenge_count = 3 # Scavenge top 3 sources

        # --- Refinement Boost (1.5x) ---
        # If we are refining a draft, we know we just added data specifically for this topic.
        # We boost retrieval to ensure that data is captured.
        if _draft:
            scout_chunks = int(scout_chunks * 1.5)
            scavenge_count = int(scavenge_count * 1.5)
            # Cap at 200 chunks total to stay safe for context window
            if scout_chunks > 150: scout_chunks = 150
            if scavenge_count > 12: scavenge_count = 12

        await log(f"⏳ **Phase 1: Multi-Stage Scout initiated...** (Variations: {num_variations + 1})")
    
        if no_web:
            await log("🛡️ **Local-Only Mode:** Skipping web reconnaissance.")
            search_queries = [question]
        else:
            search_queries = await _expand_query(llm, question, num_variations)
    
        # 2. Stage 1: The Scout (Summaries Only)
        # We find the 'Hints' first to know which sources are actually worth scavenging.
        scout_results = []
        for q in search_queries:
            res = await db.search_with_metadata(q, n_results=scout_chunks, where={"chunk_type": "summary"})
            if res: scout_results.extend(res)
            
        if not scout_results and mode != "Fast":
            # Fallback to general search if no summaries exist (e.g. initial crawl with only raw data)
            await log("⚠️ No summaries found. Falling back to global raw scavenge.")
            scout_results = []
            for q in search_queries:
                res = await db.search_with_metadata(q, n_results=scout_chunks)
                if res: scout_results.extend(res)
            
        if not scout_results:
            msg = "⚠️ No relevant information found in the database. Have you run the `/research` or `/crawl_site` command for this topic yet?"
            await log(msg)
            if _current_auto_loop == 0:
                clear_ask_state()
            return msg

        # 3. Stage 2: The Scavenge (Targeted Raw Extraction)
        # Identify top sources based on scout hits
        source_scores = {}
        for _, meta in scout_results:
            src = meta.get("source")
            if src: source_scores[src] = source_scores.get(src, 0) + 1
    
        # Sort sources by hit frequency
        top_sources = sorted(source_scores.keys(), key=lambda x: source_scores[x], reverse=True)[:scavenge_count]
    
        await log(f"🔎 **Phase 2: Scavenging technical evidence from {len(top_sources)} high-relevance sources...**")
    
        scavenge_results = []
        if no_web:
            await log("🛡️ **Local-Only Mode:** Skipping web scavenging.")
        else:
            for src in top_sources:
                # For each top source, pull the most relevant raw chunks
                res = await db.search_with_metadata(question, n_results=15, where={
                    "$and": [
                        {"source": src},
                        {"chunk_type": "raw"}
                    ]
                })
                if res: scavenge_results.extend(res)

        
        # 4. Deduplicate and Assemble
        all_hits = scout_results + scavenge_results
        unique_chunks = {}
        sources_seen = set()
    
        for doc, meta in all_hits:
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
        # We prioritize Summaries over Raw Source Data during truncation
        context_words = context_text.split()
        if len(context_words) > MAX_CONTEXT_WORDS:

            await log(f"⚠️ Context exceeds expanded budget ({len(context_words):,} words). Applying tiered truncation...")
        
            # New Tiered Approach:
            # 1. Truncate 'Raw Source Data' blocks first
            # 2. Finally truncate everything to the limit if still over
            parts = context_text.split("\n\n")
            summaries = [p for p in parts if "--- Analyzed Summary" in p]
            raw_parts = [p for p in parts if "--- Raw Source Data" in p]
        
            # Keep as many summaries as possible, then fill with raw
            new_parts = summaries[:]
            current_len = sum(len(p.split()) for p in new_parts)
        
            for p in raw_parts:
                p_len = len(p.split())
                if current_len + p_len < MAX_CONTEXT_WORDS:
                    new_parts.append(p)
                    current_len += p_len
                else:
                    break
        
            context_text = "\n\n".join(new_parts)
            # Final hard cap (Character based to preserve newlines)
            if len(context_text.split()) > MAX_CONTEXT_WORDS:
                # We slice by characters (words * 6 is a safe technical avg)
                char_cap = MAX_CONTEXT_WORDS * 6
                context_text = context_text[:char_cap] + "\n\n[... hard structural truncation for budget ...]"

        if _extra_context:
            context_text = f"{_extra_context}\n\n{context_text}"
        
        source_list = "\n".join([f"  - {url}" for url in sources_seen])
    
        # 4. Standardized Markdown Synthesis OR Draft Refining
        persona_config = STYLE_CONFIG.get(style, STYLE_CONFIG["Concise"])
    
        if _draft:
            system_prompt = (
                "You are an elite research analyst refining an intelligence report.\n"
                f"STYLE PERSONA: {persona_config['persona']}\n\n"
                "Below is your PREVIOUS incomplete draft. You are also given brand NEW CONTEXT from targeted research designed to fill the draft's Knowledge Gaps.\n\n"
                "Your task is to merge the new context into the report organically. Expand the analysis, integrate citations, "
                "and meticulously remove the Knowledge Gaps that the new tracking data has resolved.\n\n"
                "Rules:\n"
                "- Maintain the formatting schema dictated by your persona.\n"
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
            system_prompt = persona_config["persona"]
        
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

        # Final Context Shield Check: Combine and ensure it fits the model
        system_prompt, user_prompt = fit_to_context_budget(system_prompt, user_prompt, MAX_CONTEXT_WORDS)

        # Use English for all internal synthesis to maintain fidelity
        internal_language = "English"
        if language.lower() != "english":
            # Only apply target language instructions if this is the FINAL iteration
            # (No gaps remaining OR soft stop OR loop limit)
            # We check this condition later to see if we need a final translation pass.
            pass

        # Use default LLM_TIMEOUT from .env (no hardcoded override)
        answer = await llm.generate_text(system_prompt, user_prompt, temperature=0.3, max_tokens=LLM_MAX_TOKENS)
        persist_ask_state(answer, _extra_context)
    
        # 5. End if Fast mode or max auto-loops hit
        is_interrupted = check_soft_stop()
        if _current_auto_loop >= max_auto_loops or is_interrupted:
            if is_interrupted:
                await log("🛑 **Soft Stop Acknowledged:** Halting Agentic gaps auto-loop early.")
        
            # --- Task 17: Dual Output Logic ---
            # If this is the TOP-LEVEL call (loop 0), return the dual-language dict.
            # If it's a recursive call, return the EN string for continued refinement.
            if _current_auto_loop == 0:
                result = await finalize_dual_report(llm, answer, topic, language, mode, is_interrupted)
                if has_pending_gap_queue():
                    await log_retained_gap_checkpoint("this refinement cycle")
                else:
                    clear_ask_state()
                return result
            return answer
        
        # 6. Extract Gaps and Agentic RAG Re-research
        gap_queries = await extract_gap_queries(llm, answer)
        telemetry_add("gap.queries_total", len(gap_queries))
        gap_state = queue_gap_queries(gap_state, gap_queries)
        persist_ask_state(answer, _extra_context)
    
        async def loc_log(m, is_sub_step=False):
            print(m)
            if log_func: await log_func(m, is_sub_step)

        queued_gap_count = len(gap_state["order"])
        preserve_checkpoint = False
        if queued_gap_count:
            # Deliver intermediate draft file so the user can see the gaps before it stalls researching
            if draft_callback:
                await draft_callback(answer, _current_auto_loop)

            gap_batch, deferred_gap_count = dequeue_gap_batch(gap_state)
            await loc_log(f"⚠️ **Knowledge Gaps detected.** Auto-initiating targeted research loop (Iteration {_current_auto_loop + 1}/{max_auto_loops if max_auto_loops < 999 else '∞'})...")
            if gap_queries:
                await loc_log(
                    f"🎯 *Queued {len(gap_queries)} new gaps. Processing {len(gap_batch)} now; {deferred_gap_count} deferred.*",
                    is_sub_step=True
                )
            else:
                await loc_log(
                    f"🎯 *No new gaps extracted. Resuming deferred queue: {len(gap_batch)} now; {deferred_gap_count} still deferred.*",
                    is_sub_step=True
                )
            batch_context, preserve_checkpoint = await _process_gap_batch(
                topic=topic,
                mode=mode,
                no_web=no_web,
                gap_batch=gap_batch,
                draft_text=answer,
                extra_context=_extra_context,
                gap_state=gap_state,
                current_auto_loop=_current_auto_loop,
                db=db,
                llm=llm,
                log_func=log_func,
                persist_ask_state=persist_ask_state,
            )
            if not preserve_checkpoint:
                answer = await answer_question(
                    topic=topic,
                    question=question,
                    mode=mode,
                    style=style,
                    log_func=log_func,
                    draft_callback=draft_callback,
                    language=language,
                    no_web=no_web,
                    _current_auto_loop=_current_auto_loop + 1,
                    _draft=answer,
                    _extra_context=batch_context,
                    _gap_state=gap_state
                )
                persist_ask_state(answer if not isinstance(answer, dict) else answer.get("english"), None)
    
        # Final Guard: Ensure loop 0 always returns the Dual Report Dict
        if _current_auto_loop == 0:
            if isinstance(answer, dict):
                if not preserve_checkpoint and not has_pending_gap_queue():
                    clear_ask_state()
                elif not preserve_checkpoint:
                    await log_retained_gap_checkpoint("this refinement cycle")
                return answer
            result = await finalize_dual_report(llm, answer, topic, language, mode, check_soft_stop())
            if not preserve_checkpoint and not has_pending_gap_queue():
                clear_ask_state()
            elif not preserve_checkpoint:
                await log_retained_gap_checkpoint("this refinement cycle")
            return result
    
        # Recursive layers just return the EN string
        if isinstance(answer, dict): return answer["english"]
        return answer

async def finalize_dual_report(llm: LocalLLM, english_draft: str, topic: str, target_language: str, mode: str, is_interrupted: bool) -> dict:
    """Performs the final translation pass and packages both EN and Translated versions."""
    if target_language.lower() == "english":
        return {"english": english_draft, "translated": None}

    # User's Interruption Logic:
    # - Omniscient: Always translate.
    # - Balanced/Thorough + Interrupted: User originally said return EN, then said "I want both".
    # We will provide both.
    
    print(f"[Query] Performing final translation pass to {target_language}...")
    system_prompt = (
        f"You are a professional translator and research analyst. "
        f"Translate the following HIGH-FIDELITY technical report into {target_language.capitalize()}.\n\n"
        "Guidelines:\n"
        "- Maintain all Markdown formatting, charts, and structure.\n"
        "- Preserve proper nouns, product names, and mathematical symbology precisely.\n"
        "- Use professional, formal technical terminology."
    )
    user_prompt = f"REPORT TO TRANSLATE:\n{english_draft}"
    
    translated = await llm.generate_text(system_prompt, user_prompt, temperature=0.1, max_tokens=LLM_MAX_TOKENS)
    
    # 💾 Auto-Archive Persistent Reports
    store_final_report(topic, english_draft, "EN")
    if translated:
        store_final_report(topic, translated, target_language)
        
    return {"english": english_draft, "translated": translated}

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
