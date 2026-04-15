import httpx
from bs4 import BeautifulSoup

# We must look like a standard web browser, or sites will block us instantly
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_text_from_url(url: str, timeout: int = 10) -> str:
    """Visits a URL, strips out HTML/ads, and returns pure readable text."""
    print(f"[Scraper] Visiting: {url}")
    try:
        # follow_redirects=True is crucial for modern web routing
        with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status() # Throw an error if we get a 404 or 500

        # Parse the raw HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # The Purge: Destroy tags that contain useless junk
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            junk.decompose()

        # Extract whatever text is left
        text = soup.get_text(separator="\n")
        
        # Clean up massive empty gaps and whitespace
        lines = (line.strip() for line in text.splitlines())
        clean_text = "\n".join(line for line in lines if line)

        print(f"[Scraper] Successfully extracted {len(clean_text)} characters.")
        return clean_text

    except httpx.TimeoutException:
        print(f"[Scraper] Warning: Timed out waiting for {url}")
        return ""
    except Exception as e:
        print(f"[Scraper] Failed to scrape {url}: {e}")
        return ""

# Quick manual test to ensure both tools work together
if __name__ == "__main__":
    from tools.search import get_search_results
    
    # 1. Search for a topic
    test_query = "Linux kernel Bazzite OS"
    results = get_search_results(test_query, max_results=1)
    
    # 2. Scrape the first result
    if results and results[0]["url"]:
        test_url = results[0]["url"]
        extracted_text = scrape_text_from_url(test_url)
        
        print("\n--- Scrape Preview (First 500 characters) ---")
        print(extracted_text[:500] + "...\n")
