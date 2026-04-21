import httpx
from bs4 import BeautifulSoup
import tempfile
import os
from config.settings import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_KEEPALIVE_EXPIRY,
    SCRAPER_MAX_HTML_SIZE,
    SCRAPER_MAX_PDF_SIZE,
    RESOURCE_PROFILE,
)
from runtime_telemetry import bump as telemetry_bump

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 10MB limit for HTML, 150MB limit for PDFs
MAX_HTML_SIZE = SCRAPER_MAX_HTML_SIZE
MAX_PDF_SIZE = SCRAPER_MAX_PDF_SIZE
_scraper_client = None


def _scraper_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=HTTP_KEEPALIVE_EXPIRY,
    )


def _get_scraper_client() -> httpx.AsyncClient:
    global _scraper_client
    if _scraper_client is None:
        _scraper_client = httpx.AsyncClient(
            headers=HEADERS,
            timeout=20,
            follow_redirects=True,
            http2=True,
            limits=_scraper_limits(),
        )
    return _scraper_client


async def close_scraper_client():
    global _scraper_client
    if _scraper_client is not None:
        await _scraper_client.aclose()
        _scraper_client = None


def _parse_content_length(headers) -> int | None:
    raw_value = headers.get("Content-Length")
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _should_skip_pdf_download(content_length: int | None) -> bool:
    return content_length is not None and content_length > MAX_PDF_SIZE

async def scrape_text_from_url(url: str, timeout: int = 15, log_func=None) -> str:
    """Safely streams a URL, checks content type, and extracts pure readable text or PDF markdown."""
    result = await _scrape_core(url, timeout, log_func, want_links=False)
    return result

async def scrape_with_links(url: str, timeout: int = 20, log_func=None) -> tuple[str, list[str]]:
    """Same as scrape_text_from_url but also returns all absolute URLs found on the page."""
    return await _scrape_core(url, timeout, log_func, want_links=True)

async def _scrape_core(url: str, timeout: int, log_func, want_links: bool) -> any:
    async def log(msg):
        if log_func: await log_func(msg)
        else: print(msg)

    await log(f"[Scraper] Connecting to: {url}")
    telemetry_bump("url.attempted")
    
    try:
        client = _get_scraper_client()
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "").lower()
            
            if "application/pdf" in content_type:
                content_length = _parse_content_length(response.headers)
                if _should_skip_pdf_download(content_length):
                    telemetry_bump("url.skipped")
                    telemetry_bump("url.skipped.pdf_too_large")
                    await log(
                        f"[Scraper] ⚠️ PDF advertised size {content_length / 1024 / 1024:.2f}MB "
                        f"exceeds profile cap {MAX_PDF_SIZE / 1024 / 1024:.0f}MB. Skipping download."
                    )
                    return ("", []) if want_links else ""
                await log(f"[Scraper] 📄 PDF detected. Initiating secure stream...")
                text = await _stream_and_process_pdf(response, log)
                return (text, []) if want_links else text
            
            elif "text/html" in content_type or "text/plain" in content_type:
                await log(f"[Scraper] 🌐 HTML text detected. Initiating secure stream...")
                return await _stream_and_process_html(response, log, want_links, base_url=str(response.url))
                
            else:
                telemetry_bump("url.skipped")
                telemetry_bump("url.skipped.unsupported_content_type")
                await log(f"[Scraper] ⚠️ Unsupported content type skipped: {content_type}")
                return ("", []) if want_links else ""
                    
    except httpx.TimeoutException:
        telemetry_bump("url.skipped")
        telemetry_bump("url.skipped.timeout")
        await log(f"[Scraper] Warning: Timed out waiting for {url}")
        return ("", []) if want_links else ""
    except Exception as e:
        telemetry_bump("url.skipped")
        telemetry_bump("url.skipped.error")
        await log(f"[Scraper] Failed to scrape {url}: {e}")
        return ("", []) if want_links else ""

from urllib.parse import urljoin

async def _stream_and_process_html(response: httpx.Response, log_func, want_links: bool = False, base_url: str = None) -> any:
    """Streams up to limits, truncates if over, parses HTML gracefully."""
    downloaded_bytes = bytearray()
    
    async for chunk in response.aiter_bytes():
        downloaded_bytes.extend(chunk)
        if len(downloaded_bytes) > MAX_HTML_SIZE:
            telemetry_bump("html.truncated")
            await log_func(f"[Scraper] ⚠️ HTML size exceeded {MAX_HTML_SIZE//1024//1024}MB. Truncating.")
            break
            
    html_content = downloaded_bytes.decode('utf-8', errors='ignore')
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    links = []
    if want_links and base_url:
        for a in soup.find_all("a", href=True):
            absolute_url = urljoin(base_url, a["href"])
            # Remove fragments to avoid duplicate processing of the same page
            absolute_url = absolute_url.split("#")[0]
            if absolute_url.startswith("http"):
                links.append(absolute_url)
        # Deduplicate links
        links = list(dict.fromkeys(links))

    # Advanced purging 
    for junk in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]):
        junk.decompose()
        
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    clean_text = "\n".join(line for line in lines if line)
    
    # Very basic validation (e.g. Cloudflare captcha block check)
    if "Just a moment..." in clean_text and "Cloudflare" in html_content:
        telemetry_bump("url.skipped")
        telemetry_bump("url.skipped.cloudflare")
        await log_func("[Scraper] ⚠️ Blocked by Cloudflare.")
        return ("", []) if want_links else ""
        
    await log_func(f"[Scraper] Successfully extracted {len(clean_text):,} characters from HTML.")
    return (clean_text, links) if want_links else clean_text

async def _stream_and_process_pdf(response: httpx.Response, log_func) -> str:
    """Streams a PDF to disk, then runs lightweight-first extraction with triage."""
    downloaded_size = 0

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_path = temp_pdf.name
        
        async for chunk in response.aiter_bytes():
            temp_pdf.write(chunk)
            downloaded_size += len(chunk)
            
            if downloaded_size > MAX_PDF_SIZE:
                telemetry_bump("pdf.truncated")
                await log_func(f"[Scraper] ⚠️ PDF size exceeded {MAX_PDF_SIZE/1024/1024:.0f}MB. Truncating stream.")
                break
                
    size_mb = downloaded_size / (1024 * 1024)
    await log_func(f"[Scraper] Successfully downloaded {size_mb:.2f}MB PDF. Parsing text...")
    
    from tools.pdf_parser import extract_markdown_from_pdf
    import asyncio
    
    try:
        # Offload massive synchronous PyTorch loading to a background thread
        markdown_text = await asyncio.to_thread(
            extract_markdown_from_pdf,
            temp_path,
            downloaded_size,
            RESOURCE_PROFILE,
        )
        await log_func(f"[PDF Parser] Handled PDF via backend engines. Extracted {len(markdown_text):,} characters.")
        return markdown_text
    except Exception as e:
        await log_func(f"[Scraper] Failed to parse PDF extraction: {e}")
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
