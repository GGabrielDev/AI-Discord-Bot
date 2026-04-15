import httpx
from bs4 import BeautifulSoup
import tempfile
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 10MB limit for HTML, 150MB limit for PDFs
MAX_HTML_SIZE = 10 * 1024 * 1024
MAX_PDF_SIZE = 150 * 1024 * 1024

async def scrape_text_from_url(url: str, timeout: int = 15) -> str:
    """Safely streams a URL, checks content type, and extracts pure readable text or PDF markdown."""
    print(f"[Scraper] Connecting to: {url}")
    
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=timeout, follow_redirects=True, http2=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type", "").lower()
                
                if "application/pdf" in content_type:
                    print(f"[Scraper] 📄 PDF detected. Initiating secure stream...")
                    return await _stream_and_process_pdf(response)
                
                elif "text/html" in content_type or "text/plain" in content_type:
                    print(f"[Scraper] 🌐 HTML text detected. Initiating secure stream...")
                    return await _stream_and_process_html(response)
                    
                else:
                    print(f"[Scraper] ⚠️ Unsupported content type skipped: {content_type}")
                    return ""
                    
    except httpx.TimeoutException:
        print(f"[Scraper] Warning: Timed out waiting for {url}")
        return ""
    except Exception as e:
        print(f"[Scraper] Failed to scrape {url}: {e}")
        return ""

async def _stream_and_process_html(response: httpx.Response) -> str:
    """Streams up to limits, truncates if over, parses HTML gracefully."""
    downloaded_bytes = bytearray()
    
    async for chunk in response.aiter_bytes():
        downloaded_bytes.extend(chunk)
        if len(downloaded_bytes) > MAX_HTML_SIZE:
            print(f"[Scraper] ⚠️ HTML size exceeded {MAX_HTML_SIZE//1024//1024}MB. Truncating.")
            break
            
    html_content = downloaded_bytes.decode('utf-8', errors='ignore')
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Advanced purging 
    for junk in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]):
        junk.decompose()
        
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    clean_text = "\n".join(line for line in lines if line)
    
    # Very basic validation (e.g. Cloudflare captcha block check)
    if "Just a moment..." in clean_text and "Cloudflare" in html_content:
        print("[Scraper] ⚠️ Blocked by Cloudflare.")
        return ""
        
    print(f"[Scraper] Successfully extracted {len(clean_text)} characters from HTML.")
    return clean_text

async def _stream_and_process_pdf(response: httpx.Response) -> str:
    """Streams large PDFs directly to a temporary file, then parses them with marker."""
    downloaded_size = 0
    
    # We will implement the marker PDF parsing in Phase 2.
    # For now, we will stream it to disk safely and return an empty string
    # so the loop doesn't break while we're building the tool.
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_path = temp_pdf.name
        
        async for chunk in response.aiter_bytes():
            temp_pdf.write(chunk)
            downloaded_size += len(chunk)
            
            if downloaded_size > MAX_PDF_SIZE:
                print(f"[Scraper] ⚠️ PDF size exceeded {MAX_PDF_SIZE/1024/1024:.0f}MB. Truncating stream.")
                break
                
    size_mb = downloaded_size / (1024 * 1024)
    print(f"[Scraper] Successfully downloaded {size_mb:.2f}MB PDF to {temp_path}.")
    
    # --- Marker processing will be called here ---
    from tools.pdf_parser import extract_markdown_from_pdf
    import asyncio
    
    try:
        # Offload massive synchronous PyTorch loading to a background thread
        markdown_text = await asyncio.to_thread(extract_markdown_from_pdf, temp_path)
        return markdown_text
    except Exception as e:
        print(f"[Scraper] Failed to parse PDF with Marker: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Quick manual test to ensure both tools work together
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing HTML...")
        html_text = await scrape_text_from_url("https://en.wikipedia.org/wiki/Solid-state_battery")
        print(f"Got {len(html_text)} bytes of HTML text.")
        
    asyncio.run(test())
