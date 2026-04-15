import httpx
from config.settings import SEARXNG_URL

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
        
        # Generous 15-second timeout since SearXNG aggregates across multiple engines
        async with httpx.AsyncClient(timeout=15) as client:
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
