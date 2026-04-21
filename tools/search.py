import httpx
from config.settings import (
    SEARXNG_URL,
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_KEEPALIVE_EXPIRY,
)
from runtime_cache import TTLCache
from runtime_telemetry import add as telemetry_add, bump as telemetry_bump

_search_client = None
_search_response_cache = TTLCache(ttl_seconds=120, max_entries=128)


def _search_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=HTTP_KEEPALIVE_EXPIRY,
    )


def _get_search_client() -> httpx.AsyncClient:
    global _search_client
    if _search_client is None:
        _search_client = httpx.AsyncClient(timeout=15, limits=_search_limits())
    return _search_client


async def close_search_client():
    global _search_client
    if _search_client is not None:
        await _search_client.aclose()
        _search_client = None


async def get_search_results(query: str, max_results: int = 5) -> list[dict]:
    """Uses local SearXNG to find URLs for a given search query.
    
    This function is fully async to prevent blocking the Discord event loop
    during the 5-15 second SearXNG aggregation window (it queries 10+ search
    engines simultaneously in the background).
    """
    normalized_query = query.strip()
    cache_key = (SEARXNG_URL, normalized_query)
    cached_results = _search_response_cache.get(cache_key)
    if cached_results is not None:
        telemetry_bump("cache.hits")
        telemetry_bump("cache.hits.search")
        print(f"[Search] Cache hit for '{normalized_query}'")
        return cached_results[:max_results]
    telemetry_bump("cache.misses")
    telemetry_bump("cache.misses.search")

    print(f"[Search] Querying SearXNG: '{normalized_query}'")
    try:
        async def fetch_results() -> list[dict]:
            params = {
                "q": normalized_query,
                "format": "json"
            }

            client = _get_search_client()
            response = await client.get(f"{SEARXNG_URL}/search", params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            cleaned_results = []
            for r in results:
                cleaned_results.append({
                    "title": r.get("title", "No Title"),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            return cleaned_results

        cleaned_results = await _search_response_cache.get_or_set(cache_key, fetch_results)
        sliced_results = cleaned_results[:max_results]
        telemetry_bump("search.requests")
        telemetry_add("search.results_returned", len(sliced_results))
        print(f"[Search] Found {len(sliced_results)} URLs via SearXNG.")
        return sliced_results

    except Exception as e:
        telemetry_bump("search.errors")
        print(f"[Search] Error fetching results from SearXNG: {e}")
        return []

# Quick manual test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        test_query = "Linux CachyOS vs Bazzite OS"
        results = await get_search_results(test_query, max_results=3)
        
        print("\n--- SearXNG Results ---")
        for res in results:
            print(f"- {res['title']}\n  {res['url']}")
    
    asyncio.run(test())
