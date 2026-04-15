import httpx
from config.settings import SEARXNG_URL

def get_search_results(query: str, max_results: int = 5) -> list[dict]:
    """Uses local SearXNG to find URLs for a given query."""
    print(f"[Search] Querying SearXNG: '{query}'")
    try:
        # We explicitly ask SearXNG to return JSON data
        params = {
            "q": query,
            "format": "json"
        }
        
        # We give it a generous 15-second timeout since it is querying 
        # 10+ different search engines simultaneously in the background.
        with httpx.Client(timeout=15) as client:
            response = client.get(f"{SEARXNG_URL}/search", params=params)
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
    test_query = "Linux CachyOS vs Bazzite OS"
    results = get_search_results(test_query, max_results=3)
    
    print("\n--- SearXNG Results ---")
    for res in results:
        print(f"- {res['title']}\n  {res['url']}")
