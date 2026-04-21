import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse

from config.settings import (
    RUNTIME_TELEMETRY_ENABLED,
    RUNTIME_TELEMETRY_PRINT_SUMMARY,
    RUNTIME_TELEMETRY_TOP_SOURCES,
)

_ACTIVE_SESSION = ContextVar("runtime_telemetry_session", default=None)


def _safe_label_fragment(value: str, limit: int = 48) -> str:
    if not value:
        return ""
    compact = " ".join(str(value).split())
    return compact[:limit]


class TelemetrySession:
    def __init__(self, label: str, metadata: dict | None = None):
        self.label = _safe_label_fragment(label, limit=64) or "session"
        self.metadata = dict(metadata or {})
        self.started_at = time.monotonic()
        self.counters = Counter()
        self.totals = Counter()
        self.maxima = {}
        self.sources = {}

    def bump(self, name: str, value: int = 1):
        if value:
            self.counters[name] += value

    def add(self, name: str, value: float):
        if value:
            self.totals[name] += value

    def set_max(self, name: str, value: float | int):
        previous = self.maxima.get(name)
        if previous is None or value > previous:
            self.maxima[name] = value

    def note_source(self, source_url: str, **stats):
        if not source_url:
            return
        entry = self.sources.setdefault(
            source_url,
            {
                "host": urlparse(source_url).netloc or source_url,
                "summary_chunks": 0,
                "raw_chunks": 0,
                "raw_chunks_kept": 0,
                "summary_words": 0,
                "raw_words": 0,
            },
        )
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                entry[key] = entry.get(key, 0) + value
            elif value not in (None, ""):
                entry[key] = value

    def snapshot(self) -> dict:
        return {
            "label": self.label,
            "metadata": dict(self.metadata),
            "counters": dict(self.counters),
            "totals": dict(self.totals),
            "maxima": dict(self.maxima),
            "sources": {key: dict(value) for key, value in self.sources.items()},
            "duration_seconds": round(time.monotonic() - self.started_at, 3),
        }

    def format_summary(self) -> str:
        duration = time.monotonic() - self.started_at
        parts = [f"[Telemetry] {self.label} {duration:.1f}s"]

        llm_calls = self.counters.get("llm.calls", 0)
        if llm_calls:
            prompt_tokens = int(self.totals.get("llm.prompt_tokens", 0))
            completion_tokens = int(self.totals.get("llm.completion_tokens", 0))
            llm_seconds = self.totals.get("llm.seconds", 0.0)
            parts.append(
                f"llm={llm_calls} ({prompt_tokens:,} in/{completion_tokens:,} out, {llm_seconds:.1f}s)"
            )

        cache_hits = self.counters.get("cache.hits", 0)
        cache_misses = self.counters.get("cache.misses", 0)
        if cache_hits or cache_misses:
            parts.append(f"cache={cache_hits}h/{cache_misses}m")

        search_requests = self.counters.get("search.requests", 0)
        if search_requests:
            kept = int(self.totals.get("search.results_kept", 0))
            rejected = int(self.totals.get("search.results_rejected", 0))
            parts.append(f"search={search_requests} ({kept} kept/{rejected} rejected)")

        skipped = self.counters.get("url.skipped", 0)
        if skipped:
            parts.append(f"skipped={skipped}")

        route_counts = {
            key.removeprefix("gap.route."): value
            for key, value in self.counters.items()
            if key.startswith("gap.route.") and value
        }
        if route_counts:
            route_summary = ", ".join(f"{route}={count}" for route, count in sorted(route_counts.items()))
            parts.append(f"routes[{route_summary}]")

        if RUNTIME_TELEMETRY_TOP_SOURCES > 0 and self.sources:
            ranked_sources = sorted(
                self.sources.values(),
                key=lambda source: (
                    source.get("summary_chunks", 0) + source.get("raw_chunks_kept", 0),
                    source.get("raw_chunks", 0),
                ),
                reverse=True,
            )[:RUNTIME_TELEMETRY_TOP_SOURCES]
            source_summary = ", ".join(
                f"{src['host']} s{int(src.get('summary_chunks', 0))}/r{int(src.get('raw_chunks_kept', 0))}"
                f"of{int(src.get('raw_chunks', 0))}"
                for src in ranked_sources
            )
            if source_summary:
                parts.append(f"sources[{source_summary}]")

        return " | ".join(parts)


@contextmanager
def telemetry_session(label: str, metadata: dict | None = None):
    if not RUNTIME_TELEMETRY_ENABLED:
        yield None
        return

    existing = _ACTIVE_SESSION.get()
    if existing is not None:
        if metadata:
            for key, value in metadata.items():
                existing.metadata.setdefault(key, value)
        yield existing
        return

    session = TelemetrySession(label, metadata=metadata)
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        if RUNTIME_TELEMETRY_PRINT_SUMMARY:
            print(session.format_summary())
        _ACTIVE_SESSION.reset(token)


def current_session() -> TelemetrySession | None:
    return _ACTIVE_SESSION.get()


def bump(name: str, value: int = 1):
    session = current_session()
    if session is not None:
        session.bump(name, value)


def add(name: str, value: float):
    session = current_session()
    if session is not None:
        session.add(name, value)


def set_max(name: str, value: float | int):
    session = current_session()
    if session is not None:
        session.set_max(name, value)


def note_source(source_url: str, **stats):
    session = current_session()
    if session is not None:
        session.note_source(source_url, **stats)


def snapshot() -> dict | None:
    session = current_session()
    return session.snapshot() if session is not None else None
