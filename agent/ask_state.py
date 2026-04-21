import os
import re
from difflib import SequenceMatcher


GAP_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "me", "of", "on", "or",
    "please", "show", "tell", "than", "that", "the", "their", "them", "these",
    "this", "those", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
}
GAP_FRESHNESS_TOKENS = {"current", "latest", "live", "new", "now", "recent", "today"}
GAP_ROUTE_PRIORITY = {
    "needs_web": 5,
    "partial_local": 4,
    "defer_local": 3,
    "blocked_offline": 2,
    "resolved_local": 1,
    "new": 0,
}
GAP_RESOLUTION_PRIORITY = {
    "web_exhausted": 5,
    "web_enriched": 4,
    "needs_web": 3,
    "partial_local": 2,
    "resolved_local": 1,
    "unseen": 0,
}

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_GAPS_PER_CYCLE = _env_int("ASK_MAX_GAPS_PER_CYCLE", 3)
LOCAL_RESOLUTION_THRESHOLD = _env_float("ASK_LOCAL_RESOLUTION_THRESHOLD", 0.72)
WEB_TRIGGER_THRESHOLD = _env_float("ASK_WEB_TRIGGER_THRESHOLD", 0.38)
PARTIAL_CONTEXT_THRESHOLD = _env_float("ASK_PARTIAL_CONTEXT_THRESHOLD", 0.28)
LOCAL_RETRY_LIMIT = _env_int("ASK_LOCAL_RETRY_LIMIT", 2)
WEB_BACKOFF_LOOPS = _env_int("ASK_WEB_BACKOFF_LOOPS", 2)
FRESHNESS_MAX_AGE_DAYS = _env_int("ASK_FRESHNESS_MAX_AGE_DAYS", 180)
FRESHNESS_PENALTY = _env_float("ASK_FRESHNESS_PENALTY", 0.18)
FRESHNESS_HINT_PATTERN = re.compile(r"\b(current|latest|recent|today|new|status|202[4-9]|live)\b", re.IGNORECASE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_gap_query(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s:/.-]", " ", query.lower())).strip()


def _stem_gap_token(token: str) -> str:
    if re.fullmatch(r"202[4-9]", token):
        return "current-year"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 5:
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _semantic_gap_tokens(query: str, *, drop_freshness: bool = False) -> tuple[str, ...]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", normalize_gap_query(query)):
        token = _stem_gap_token(token)
        if len(token) <= 1 or token in GAP_QUERY_STOPWORDS:
            continue
        if drop_freshness and token in GAP_FRESHNESS_TOKENS:
            continue
        tokens.append(token)
    return tuple(tokens)


def _has_token_subsequence(tokens: tuple[str, ...], other_tokens: tuple[str, ...]) -> bool:
    if len(other_tokens) < 2 or len(other_tokens) > len(tokens):
        return False
    window = len(other_tokens)
    return any(tokens[idx:idx + window] == other_tokens for idx in range(len(tokens) - window + 1))


def _gap_similarity_score(query_a: str, query_b: str) -> float:
    normalized_a = normalize_gap_query(query_a)
    normalized_b = normalize_gap_query(query_b)
    if not normalized_a or not normalized_b:
        return 0.0
    if normalized_a == normalized_b:
        return 1.0

    scores = []
    for tokens_a, tokens_b in (
        (_semantic_gap_tokens(query_a, drop_freshness=True), _semantic_gap_tokens(query_b, drop_freshness=True)),
        (_semantic_gap_tokens(query_a), _semantic_gap_tokens(query_b)),
    ):
        if len(tokens_a) < 2 or len(tokens_b) < 2:
            continue
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        shared = len(set_a & set_b)
        if shared < 2:
            continue
        coverage = shared / max(1, min(len(set_a), len(set_b)))
        jaccard = shared / max(1, len(set_a | set_b))
        ratio = SequenceMatcher(None, " ".join(tokens_a), " ".join(tokens_b)).ratio()
        score = (coverage * 0.55) + (jaccard * 0.25) + (ratio * 0.2)
        if set_a == set_b:
            score = max(score, 0.97)
        if _has_token_subsequence(tokens_a, tokens_b) or _has_token_subsequence(tokens_b, tokens_a):
            score = max(score, 0.94)
        scores.append(score)
    return max(scores, default=0.0)


def _select_representative_query(current_query: str | None, candidate_query: str | None) -> str:
    current_query = current_query or ""
    candidate_query = candidate_query or ""
    if not current_query:
        return candidate_query
    if not candidate_query:
        return current_query
    current_score = (
        1 if is_freshness_gap(current_query) else 0,
        len(set(_semantic_gap_tokens(current_query, drop_freshness=True))),
        len(set(_semantic_gap_tokens(current_query))),
        len(normalize_gap_query(current_query)),
    )
    candidate_score = (
        1 if is_freshness_gap(candidate_query) else 0,
        len(set(_semantic_gap_tokens(candidate_query, drop_freshness=True))),
        len(set(_semantic_gap_tokens(candidate_query))),
        len(normalize_gap_query(candidate_query)),
    )
    return candidate_query if candidate_score > current_score else current_query


def _resolve_gap_key(gap_state: dict, gap_query: str) -> str:
    normalized = normalize_gap_query(gap_query)
    if not normalized:
        return ""
    if normalized in gap_state.get("pending", {}) or normalized in gap_state.get("repeat_counts", {}) or normalized in gap_state.get("details", {}):
        return normalized

    order_index = {gap_key: idx for idx, gap_key in enumerate(gap_state.get("order", []))}
    best_key = normalized
    best_rank = None
    candidate_keys = gap_state.get("order", []) + list(gap_state.get("pending", {}).keys()) + list(gap_state.get("repeat_counts", {}).keys()) + list(gap_state.get("details", {}).keys())
    seen = set()
    for gap_key in candidate_keys:
        if gap_key in seen:
            continue
        seen.add(gap_key)
        representative_query = (
            gap_state.get("pending", {}).get(gap_key)
            or gap_state.get("details", {}).get(gap_key, {}).get("query")
            or gap_key
        )
        similarity = _gap_similarity_score(gap_query, representative_query)
        if similarity < 0.86:
            continue
        rank = (
            similarity,
            gap_state.get("repeat_counts", {}).get(gap_key, 0),
            -order_index.get(gap_key, len(order_index)),
        )
        if best_rank is None or rank > best_rank:
            best_key = gap_key
            best_rank = rank
    return best_key


def _prefer_gap_state_value(current_value: str | None, candidate_value: str | None, priorities: dict[str, int]) -> str | None:
    current_value = current_value or ""
    candidate_value = candidate_value or ""
    current_rank = priorities.get(current_value, -1)
    candidate_rank = priorities.get(candidate_value, -1)
    if candidate_rank > current_rank:
        return candidate_value
    return current_value or candidate_value


def ensure_gap_state(gap_state: dict | None) -> dict:
    if gap_state is None:
        return {"pending": {}, "order": [], "repeat_counts": {}, "details": {}, "loop_index": 0}
    gap_state.setdefault("pending", {})
    gap_state.setdefault("order", [])
    gap_state.setdefault("repeat_counts", {})
    gap_state.setdefault("details", {})
    gap_state.setdefault("loop_index", 0)
    return gap_state


def advance_gap_cycle(gap_state: dict, current_loop: int) -> dict:
    gap_state["loop_index"] = max(gap_state.get("loop_index", 0), current_loop)
    return gap_state


def ensure_gap_meta(gap_state: dict, normalized: str, gap_query: str) -> dict:
    details = gap_state["details"]
    gap_meta = details.get(normalized)
    if gap_meta is None:
        gap_meta = {
            "query": gap_query,
            "repeat_count": 0,
            "local_attempts": 0,
            "web_attempts": 0,
            "last_confidence": 0.0,
            "last_llm_confidence": 0.0,
            "last_route": "new",
            "last_resolution": "unseen",
            "last_source_count": 0,
            "last_raw_hits": 0,
            "last_summary_hits": 0,
            "local_evidence": "",
            "cooldown_until_loop": 0
        }
        details[normalized] = gap_meta

    gap_meta["query"] = _select_representative_query(gap_meta.get("query"), gap_query)

    gap_meta["repeat_count"] = gap_state["repeat_counts"].get(normalized, 0)
    return gap_meta


def queue_gap_queries(gap_state: dict, gap_queries: list[str]) -> dict:
    gap_state = ensure_gap_state(gap_state)
    pending = gap_state["pending"]
    order = gap_state["order"]
    repeat_counts = gap_state["repeat_counts"]

    for gap_query in gap_queries:
        gap_key = _resolve_gap_key(gap_state, gap_query)
        if not gap_key:
            continue

        repeat_counts[gap_key] = repeat_counts.get(gap_key, 0) + 1
        gap_meta = ensure_gap_meta(gap_state, gap_key, gap_query)
        gap_meta["repeat_count"] = repeat_counts[gap_key]
        pending[gap_key] = _select_representative_query(pending.get(gap_key), gap_query)
        if gap_key not in order:
            order.append(gap_key)

    return gap_state


def dequeue_gap_batch(gap_state: dict, limit: int = MAX_GAPS_PER_CYCLE) -> tuple[list[str], int]:
    order = gap_state["order"]
    if not order:
        return [], 0

    pending = gap_state["pending"]
    repeat_counts = gap_state["repeat_counts"]
    details = gap_state["details"]
    loop_index = gap_state.get("loop_index", 0)
    order_index = {gap_key: idx for idx, gap_key in enumerate(order)}
    ranked_keys = sorted(
        order,
        key=lambda gap_key: (
            1 if details.get(gap_key, {}).get("cooldown_until_loop", 0) > loop_index else 0,
            -1 if details.get(gap_key, {}).get("last_route") == "needs_web" else 0,
            -repeat_counts.get(gap_key, 0),
            -1 if details.get(gap_key, {}).get("last_route") == "partial_local" else 0,
            order_index[gap_key]
        )
    )

    selected_keys = ranked_keys[:limit]
    remaining_keys = ranked_keys[limit:]
    selected_queries = [pending[gap_key] for gap_key in selected_keys]

    gap_state["pending"] = {gap_key: pending[gap_key] for gap_key in remaining_keys}
    gap_state["order"] = remaining_keys
    return selected_queries, len(remaining_keys)


def restore_gap_batch(gap_state: dict, gap_queries: list[str]) -> dict:
    if not gap_queries:
        return gap_state

    gap_state = ensure_gap_state(gap_state)
    pending = gap_state["pending"]
    restored_keys = []

    for gap_query in gap_queries:
        gap_key = _resolve_gap_key(gap_state, gap_query)
        if not gap_key:
            continue
        ensure_gap_meta(gap_state, gap_key, gap_query)
        pending[gap_key] = _select_representative_query(pending.get(gap_key), gap_query)
        if gap_key not in restored_keys:
            restored_keys.append(gap_key)

    if not restored_keys:
        return gap_state

    gap_state["order"] = restored_keys + [gap_key for gap_key in gap_state["order"] if gap_key not in restored_keys]
    gap_state["pending"] = {gap_key: pending[gap_key] for gap_key in gap_state["order"] if gap_key in pending}
    return gap_state


def merge_gap_memory(runtime_state: dict | None, saved_memory: dict | None) -> dict:
    runtime_state = ensure_gap_state(runtime_state)
    if not saved_memory:
        return runtime_state

    saved_memory = ensure_gap_state(saved_memory)
    runtime_state["loop_index"] = max(runtime_state.get("loop_index", 0), saved_memory.get("loop_index", 0))

    repeat_counts = saved_memory.get("repeat_counts", {})
    saved_details = saved_memory.get("details", {})
    merged_keys = set()
    for saved_key in list(repeat_counts.keys()) + list(saved_details.keys()):
        if saved_key in merged_keys:
            continue
        merged_keys.add(saved_key)
        gap_meta = saved_details.get(saved_key, {})
        gap_query = gap_meta.get("query") or saved_memory.get("pending", {}).get(saved_key) or saved_key
        runtime_key = _resolve_gap_key(runtime_state, gap_query)
        runtime_state["repeat_counts"][runtime_key] = runtime_state["repeat_counts"].get(runtime_key, 0) + repeat_counts.get(saved_key, 0)
        merged_meta = ensure_gap_meta(runtime_state, runtime_key, gap_query)
        merged_meta["query"] = _select_representative_query(merged_meta.get("query"), gap_query)
        merged_meta["repeat_count"] = runtime_state["repeat_counts"][runtime_key]
        merged_meta["local_attempts"] = max(merged_meta.get("local_attempts", 0), gap_meta.get("local_attempts", 0))
        merged_meta["web_attempts"] = max(merged_meta.get("web_attempts", 0), gap_meta.get("web_attempts", 0))
        merged_meta["last_confidence"] = max(safe_float(merged_meta.get("last_confidence", 0.0)), safe_float(gap_meta.get("last_confidence", 0.0)))
        merged_meta["last_llm_confidence"] = max(safe_float(merged_meta.get("last_llm_confidence", 0.0)), safe_float(gap_meta.get("last_llm_confidence", 0.0)))
        merged_meta["last_source_count"] = max(merged_meta.get("last_source_count", 0), gap_meta.get("last_source_count", 0))
        merged_meta["last_raw_hits"] = max(merged_meta.get("last_raw_hits", 0), gap_meta.get("last_raw_hits", 0))
        merged_meta["last_summary_hits"] = max(merged_meta.get("last_summary_hits", 0), gap_meta.get("last_summary_hits", 0))
        merged_meta["cooldown_until_loop"] = max(merged_meta.get("cooldown_until_loop", 0), gap_meta.get("cooldown_until_loop", 0))
        merged_meta["last_seen_loop"] = max(merged_meta.get("last_seen_loop", 0), gap_meta.get("last_seen_loop", 0))
        merged_meta["last_web_sources_added"] = max(merged_meta.get("last_web_sources_added", 0), gap_meta.get("last_web_sources_added", 0))
        if len(gap_meta.get("local_evidence", "")) > len(merged_meta.get("local_evidence", "")):
            merged_meta["local_evidence"] = gap_meta["local_evidence"]
        merged_meta["last_route"] = _prefer_gap_state_value(
            merged_meta.get("last_route"),
            gap_meta.get("last_route"),
            GAP_ROUTE_PRIORITY,
        )
        merged_meta["last_resolution"] = _prefer_gap_state_value(
            merged_meta.get("last_resolution"),
            gap_meta.get("last_resolution"),
            GAP_RESOLUTION_PRIORITY,
        )

    ordered_queries = []
    seen_queries = set()
    for saved_key in saved_memory.get("order", []) + list(saved_memory.get("pending", {}).keys()):
        gap_query = _select_representative_query(
            saved_memory.get("pending", {}).get(saved_key),
            saved_details.get(saved_key, {}).get("query") or saved_key,
        )
        normalized = normalize_gap_query(gap_query)
        if not normalized or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        ordered_queries.append(gap_query)
    restore_gap_batch(runtime_state, ordered_queries)
    return runtime_state


def build_gap_memory_snapshot(gap_state: dict) -> dict:
    gap_state = ensure_gap_state(gap_state)
    return {
        "repeat_counts": dict(gap_state.get("repeat_counts", {})),
        "details": dict(gap_state.get("details", {}))
    }


def quality_from_meta(meta: dict) -> float:
    explicit_score = safe_float(meta.get("source_quality_score"), -1.0)
    if explicit_score >= 0:
        base_score = clamp(explicit_score)
    else:
        has_raw = safe_float(meta.get("source_has_raw"), 1.0 if meta.get("chunk_type") == "raw" else 0.0)
        has_summary = safe_float(meta.get("source_has_summary"), 1.0 if meta.get("chunk_type") == "summary" else 0.0)
        total_chunks = safe_float(meta.get("source_total_chunks"), meta.get("total_chunks", 1))
        coverage_score = min(total_chunks, 8.0) / 8.0
        base_score = clamp(0.2 + (0.25 * has_raw) + (0.25 * has_summary) + (0.3 * coverage_score))

    age_days = safe_float(meta.get("source_age_days"), 0.0)
    if age_days > FRESHNESS_MAX_AGE_DAYS:
        base_score -= FRESHNESS_PENALTY * min(1.0, age_days / max(FRESHNESS_MAX_AGE_DAYS, 1))
    return clamp(base_score)


def is_freshness_gap(gap_query: str) -> bool:
    return bool(FRESHNESS_HINT_PATTERN.search(gap_query))


def record_gap_probe(gap_state: dict, gap_query: str, probe: dict, current_loop: int) -> dict:
    gap_key = _resolve_gap_key(gap_state, gap_query)
    gap_meta = ensure_gap_meta(gap_state, gap_key, gap_query)
    gap_meta["local_attempts"] += 1
    gap_meta["last_confidence"] = round(probe["local_score"], 3)
    gap_meta["last_llm_confidence"] = round(probe["llm_confidence"], 3)
    gap_meta["last_source_count"] = probe["source_count"]
    gap_meta["last_raw_hits"] = probe["raw_hits"]
    gap_meta["last_summary_hits"] = probe["summary_hits"]
    gap_meta["local_evidence"] = probe["answer"][:2000] if probe["answer"] else ""
    gap_meta["last_seen_loop"] = current_loop
    return gap_meta


def set_gap_route(gap_meta: dict, route: str, current_loop: int, cooldown_loops: int = 0):
    gap_meta["last_route"] = route
    gap_meta["last_resolution"] = route
    gap_meta["cooldown_until_loop"] = current_loop + cooldown_loops


def select_gap_route(gap_meta: dict, probe: dict, no_web: bool) -> str:
    if probe["resolved"] and probe["local_score"] >= LOCAL_RESOLUTION_THRESHOLD:
        return "resolved_local"
    if no_web:
        return "blocked_offline"
    if is_freshness_gap(gap_meta["query"]) and probe["local_score"] < LOCAL_RESOLUTION_THRESHOLD:
        return "needs_web"
    if probe["total_hits"] == 0 or probe["local_score"] < WEB_TRIGGER_THRESHOLD:
        return "needs_web"
    if gap_meta["local_attempts"] >= LOCAL_RETRY_LIMIT and probe["local_score"] < LOCAL_RESOLUTION_THRESHOLD:
        return "needs_web"
    if probe["has_partial_answer"] or probe["local_score"] >= PARTIAL_CONTEXT_THRESHOLD:
        return "partial_local"
    return "defer_local"


def record_gap_web_outcome(gap_meta: dict, current_loop: int, sources_added: int, failed: bool = False):
    gap_meta["web_attempts"] += 1
    gap_meta["last_web_sources_added"] = sources_added
    if failed or sources_added <= 0:
        set_gap_route(gap_meta, "needs_web", current_loop, WEB_BACKOFF_LOOPS)
        gap_meta["last_resolution"] = "web_exhausted"
        return
    set_gap_route(gap_meta, "needs_web", current_loop, 1)
    gap_meta["last_resolution"] = "web_enriched"


def build_gap_context(gap_query: str, probe: dict) -> str:
    if not probe.get("answer"):
        return ""
    return f"LOCAL EVIDENCE FOR '{gap_query}':\n{probe['answer']}"


def explain_gap_route(gap_meta: dict, probe: dict, route: str, no_web: bool) -> str:
    score = probe.get("local_score", 0.0)
    llm_conf = probe.get("llm_confidence", 0.0)
    sources = probe.get("source_count", 0)
    attempts = gap_meta.get("local_attempts", 0)

    if route == "resolved_local":
        return f"score={score:.2f}, llm={llm_conf:.2f}, sources={sources}"
    if route == "blocked_offline":
        return f"offline-only mode, score={score:.2f}, sources={sources}"
    if route == "needs_web":
        if no_web:
            return f"offline-only mode blocked web escalation, score={score:.2f}"
        if is_freshness_gap(gap_meta["query"]):
            return f"freshness-sensitive gap, score={score:.2f}, sources={sources}"
        if probe.get("total_hits", 0) == 0:
            return "no local hits"
        if score < WEB_TRIGGER_THRESHOLD:
            return f"score below web threshold ({score:.2f} < {WEB_TRIGGER_THRESHOLD:.2f})"
        return f"local retries exhausted ({attempts}/{LOCAL_RETRY_LIMIT}), score={score:.2f}"
    if route == "partial_local":
        return f"partial answer kept, score={score:.2f}, sources={sources}"
    if route == "defer_local":
        return f"weak local signal kept for retry, score={score:.2f}, sources={sources}"
    return f"score={score:.2f}, sources={sources}"
