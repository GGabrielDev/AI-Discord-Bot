import re
from llm.client import LocalLLM
from config.settings import LLM_CONTEXT_WINDOW, SAFE_WORD_BUDGET

# Safety ceiling: max words we'll ever feed to the LLM in a single call.
MAX_INPUT_WORDS = SAFE_WORD_BUDGET

# Max words to send per summarization call. 
# Use 70% of the safe word budget to leave plenty of room for 
# the prompt instructions and the R1 model's <think> response.
MAX_WORDS_PER_CALL = int(SAFE_WORD_BUDGET * 0.7)


def chunk_text(text: str, target_size: int = 400, overlap_sentences: int = 2) -> list[str]:
    """Semantic chunker: splits on paragraph boundaries with sentence-level overlap.
    
    1. Split on paragraph boundaries (double newline)
    2. Merge small paragraphs with adjacent ones
    3. Split oversized paragraphs at sentence boundaries
    4. Add overlap — each chunk includes the last N sentences of the previous chunk
    
    Args:
        text: The text to chunk
        target_size: Target chunk size in words
        overlap_sentences: Number of trailing sentences to carry into the next chunk
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Split oversized paragraphs at sentence boundaries
    sections = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= target_size:
            sections.append(para)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > target_size and current:
                    sections.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += sent_len
            if current:
                sections.append(" ".join(current))
    
    # Merge small sections with neighbors (< 80 words)
    merged = []
    buffer = ""
    for section in sections:
        if buffer:
            combined = buffer + "\n\n" + section
            if len(combined.split()) <= target_size:
                buffer = combined
                continue
            else:
                merged.append(buffer)
                buffer = section
        else:
            buffer = section
        
        if len(buffer.split()) >= 80:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    
    if not merged:
        # Fallback: word-based split if nothing else works
        words = text.split()
        return [" ".join(words[i:i + target_size]) for i in range(0, len(words), target_size)]
    
    # Add overlap: carry trailing sentences from previous chunk into next
    if overlap_sentences > 0 and len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_sentences = re.split(r'(?<=[.!?])\s+', merged[i - 1])
            overlap = prev_sentences[-overlap_sentences:] if len(prev_sentences) >= overlap_sentences else prev_sentences
            overlapped.append(" ".join(overlap) + "\n\n" + merged[i])
        return overlapped
    
    return merged


def compress_raw_text(text: str) -> str:
    """Aggressively strips useless formatting and whitespace noise to compress raw text directly."""
    if not text:
        return ""
        
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace weird tab structures with single spaces
    text = re.sub(r'\t+', ' ', text)
    # Strip sequential spaces into a single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Clean PDF-style hanging page numbers or massive footers (e.g., "\n  12  \n")
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Return compressed text
    return text.strip()


async def summarize_page(raw_text: str, subject: str, url: str, log_func=None) -> str:
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
    
    async def log(msg):
        if log_func: 
            await log_func(msg)
        else:
            print(msg)
    
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
        
        await log(f"[Summarizer] Analyzing {len(words):,} words from {url}...")
        summary = await llm.generate_text_with_budget(
            system_prompt, user_prompt, max_input_words=MAX_INPUT_WORDS, temperature=0.3
        )
        return summary if summary else raw_text[:2000]  # Fallback to truncated raw if LLM fails
    
    # Long page: split into sections, summarize each, then consolidate
    sections = []
    for i in range(0, len(words), MAX_WORDS_PER_CALL):
        section = " ".join(words[i:i + MAX_WORDS_PER_CALL])
        sections.append(section)
    
    await log(f"[Summarizer] Long page ({len(words):,} words) — splitting into {len(sections)} sections...")
    
    partial_summaries = []
    for idx, section in enumerate(sections):
        user_prompt = (
            f"RESEARCH TOPIC: {subject}\n"
            f"SOURCE URL: {url}\n"
            f"SECTION {idx + 1} of {len(sections)}\n\n"
            f"TEXT TO ANALYZE:\n{section}\n\n"
            f"Extract all relevant information about '{subject}' from this section."
        )
        
        await log(f"[Summarizer] Processing section {idx + 1}/{len(sections)}...")
        partial = await llm.generate_text_with_budget(
            system_prompt, user_prompt, max_input_words=MAX_INPUT_WORDS, temperature=0.3
        )
        if partial:
            partial_summaries.append(partial)
    
    if not partial_summaries:
        await log("[Summarizer] All section summaries failed. Returning truncated raw text.")
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
    
    await log(f"[Summarizer] Consolidating {len(partial_summaries)} section summaries...")
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
    
    # Truncate if too long for a single call (now uses dynamic MAX_WORDS_PER_CALL)
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
