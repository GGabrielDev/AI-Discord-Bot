from llm.client import LocalLLM
from config.settings import LLM_CONTEXT_WINDOW

# Max words to send per summarization call. 
# Gemma 4 E4B can handle large contexts, but we keep it reasonable 
# to get focused summaries and avoid overwhelming the model.
MAX_WORDS_PER_CALL = 3000

# Safety ceiling: max words we'll ever feed to the LLM in a single call.
# Estimated at ~0.75 words per token, with 80% of the context window as headroom.
MAX_INPUT_WORDS = int(LLM_CONTEXT_WINDOW * 0.75 * 0.8)


async def summarize_page(raw_text: str, subject: str, url: str) -> str:
    """Sends scraped text to the LLM to extract key facts and produce a structured summary.
    
    For long pages, the text is split into manageable sections and each is summarized
    individually, then the partial summaries are consolidated into one final summary.
    
    Args:
        raw_text: The full scraped text from a web page
        subject: The research topic (used to focus extraction)
        url: Source URL (for context in the prompt)
        
    Returns:
        A structured summary string ready for storage in ChromaDB
    """
    llm = LocalLLM()
    words = raw_text.split()
    
    system_prompt = (
        "You are a meticulous research analyst extracting information from a web page. "
        "Your job is to read the text carefully and extract ALL relevant information about the given topic.\n\n"
        "Rules:\n"
        "- Extract key facts, data points, statistics, dates, and named entities\n"
        "- Note relationships between concepts\n"
        "- Preserve specific numbers, percentages, and technical details — do NOT generalize them\n"
        "- If information seems tangential but could be relevant, include it with a note\n"
        "- Be thorough — missing information is worse than including too much\n"
        "- Output a structured summary with clear bullet points or sections\n"
        "- Do NOT make up information that isn't in the source text"
    )
    
    # Short enough to fit in one call
    if len(words) <= MAX_WORDS_PER_CALL:
        user_prompt = (
            f"RESEARCH TOPIC: {subject}\n"
            f"SOURCE URL: {url}\n\n"
            f"TEXT TO ANALYZE:\n{raw_text}\n\n"
            f"Extract all relevant information about '{subject}' from this text. "
            f"Produce a structured summary."
        )
        
        print(f"[Summarizer] Analyzing {len(words)} words from {url}...")
        summary = await llm.generate_text_with_budget(
            system_prompt, user_prompt, max_input_words=MAX_INPUT_WORDS, temperature=0.3
        )
        return summary if summary else raw_text[:2000]  # Fallback to truncated raw if LLM fails
    
    # Long page: split into sections, summarize each, then consolidate
    sections = []
    for i in range(0, len(words), MAX_WORDS_PER_CALL):
        section = " ".join(words[i:i + MAX_WORDS_PER_CALL])
        sections.append(section)
    
    print(f"[Summarizer] Long page ({len(words)} words) — splitting into {len(sections)} sections...")
    
    partial_summaries = []
    for idx, section in enumerate(sections):
        user_prompt = (
            f"RESEARCH TOPIC: {subject}\n"
            f"SOURCE URL: {url}\n"
            f"SECTION {idx + 1} of {len(sections)}\n\n"
            f"TEXT TO ANALYZE:\n{section}\n\n"
            f"Extract all relevant information about '{subject}' from this section."
        )
        
        print(f"[Summarizer] Processing section {idx + 1}/{len(sections)}...")
        partial = await llm.generate_text_with_budget(
            system_prompt, user_prompt, max_input_words=MAX_INPUT_WORDS, temperature=0.3
        )
        if partial:
            partial_summaries.append(partial)
    
    if not partial_summaries:
        print("[Summarizer] All section summaries failed. Returning truncated raw text.")
        return raw_text[:2000]
    
    # If only one section succeeded, return it directly
    if len(partial_summaries) == 1:
        return partial_summaries[0]
    
    # Consolidate partial summaries into one cohesive summary
    consolidation_prompt = (
        f"RESEARCH TOPIC: {subject}\n"
        f"SOURCE: {url}\n\n"
        f"Below are partial summaries extracted from different sections of the same web page. "
        f"Consolidate them into ONE cohesive, structured summary. "
        f"Remove duplicates but preserve ALL unique facts and details.\n\n"
    )
    
    for idx, ps in enumerate(partial_summaries):
        consolidation_prompt += f"--- Section {idx + 1} Summary ---\n{ps}\n\n"
    
    consolidation_prompt += "Produce the final consolidated summary:"
    
    print(f"[Summarizer] Consolidating {len(partial_summaries)} section summaries...")
    final_summary = await llm.generate_text_with_budget(
        "You are a research editor. Consolidate multiple partial summaries into one cohesive document. "
        "Preserve all unique facts and remove duplicates.",
        consolidation_prompt,
        max_input_words=MAX_INPUT_WORDS,
        temperature=0.3
    )
    
    return final_summary if final_summary else "\n\n".join(partial_summaries)


async def extract_key_facts(raw_text: str, subject: str) -> list[str]:
    """Extracts discrete facts from text as a list of strings.
    
    Each fact is self-contained and suitable for individual chunk storage.
    
    Args:
        raw_text: Source text to extract from
        subject: Research topic for focus
        
    Returns:
        List of fact strings
    """
    llm = LocalLLM()
    
    # Truncate if too long for a single call
    words = raw_text.split()
    if len(words) > MAX_WORDS_PER_CALL:
        raw_text = " ".join(words[:MAX_WORDS_PER_CALL])
    
    system_prompt = (
        "You are a fact extraction engine. Read the text and extract discrete, self-contained facts. "
        "Return ONLY a valid JSON object with a single key 'facts' containing a list of strings. "
        "Each fact should be a complete sentence that stands on its own. "
        "Example: {\"facts\": [\"The population of X is 1.5 million.\", \"Y was founded in 2003.\"]}"
    )
    
    user_prompt = (
        f"TOPIC: {subject}\n\n"
        f"TEXT:\n{raw_text}\n\n"
        f"Extract all facts relevant to '{subject}'. Output strictly valid JSON."
    )
    
    result = await llm.generate_json(system_prompt, user_prompt)
    
    if result and "facts" in result:
        return result["facts"]
    
    return []
