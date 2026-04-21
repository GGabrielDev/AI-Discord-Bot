import re
import os

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

    if len(gap_query) > len(gap_meta.get("query", "")):
        gap_meta["query"] = gap_query

    gap_meta["repeat_count"] = gap_state["repeat_counts"].get(normalized, 0)
    return gap_meta


def queue_gap_queries(gap_state: dict, gap_queries: list[str]) -> dict:
    pending = gap_state["pending"]
    order = gap_state["order"]
    repeat_counts = gap_state["repeat_counts"]

    for gap_query in gap_queries:
        normalized = normalize_gap_query(gap_query)
        if not normalized:
            continue

        repeat_counts[normalized] = repeat_counts.get(normalized, 0) + 1
        gap_meta = ensure_gap_meta(gap_state, normalized, gap_query)
        gap_meta["repeat_count"] = repeat_counts[normalized]
        best_query = pending.get(normalized)
        if best_query is None or len(gap_query) > len(best_query):
            pending[normalized] = gap_query
        if normalized not in order:
            order.append(normalized)

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
        normalized = normalize_gap_query(gap_query)
        if not normalized:
            continue
        ensure_gap_meta(gap_state, normalized, gap_query)
        best_query = pending.get(normalized)
        if best_query is None or len(gap_query) > len(best_query):
            pending[normalized] = gap_query
        if normalized not in restored_keys:
            restored_keys.append(normalized)

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
    runtime_state["repeat_counts"].update(saved_memory.get("repeat_counts", {}))
    for normalized, gap_meta in saved_memory.get("details", {}).items():
        existing = runtime_state["details"].get(normalized, {})
        merged = dict(gap_meta)
        merged.update({k: v for k, v in existing.items() if v not in ("", None)})
        runtime_state["details"][normalized] = merged
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
    normalized = normalize_gap_query(gap_query)
    gap_meta = ensure_gap_meta(gap_state, normalized, gap_query)
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
