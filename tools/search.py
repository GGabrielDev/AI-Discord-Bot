import httpx
from config.settings import (
    SEARXNG_URL,
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_KEEPALIVE_EXPIRY,
)

_search_client = None


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
    print(f"[Search] Querying SearXNG: '{query}'")
    try:
        params = {
            "q": query,
            "format": "json"
        }
        
        client = _get_search_client()
        response = await client.get(f"{SEARXNG_URL}/search", params=params)
        response.raise_for_status()
            
        data = response.json()
        results = data.get("results", [])
        
        cleaned_results = []
        for r in results[:max_results]:
            cleaned_results.append({
                "title": r.get("title", "No Title"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })
            
        print(f"[Search] Found {len(cleaned_results)} URLs via SearXNG.")
        return cleaned_results
        
    except Exception as e:
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
