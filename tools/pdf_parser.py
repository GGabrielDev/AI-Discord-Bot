"""PDF Text Extraction — Two-Tier Strategy for Constrained Hardware.

Tier 1 (Default): PyMuPDF — Pure C library, ~5MB RAM, instant extraction.
    Handles 95% of PDFs perfectly. No deep learning, no GPU, no OOM risk.

Tier 2 (Optional): Marker — Deep learning vision models (Surya OCR).
    Superior for complex layouts (multi-column, tables, equations).
    Costs ~1.3GB RAM for model weights + additional for inference.
    On memory-constrained systems (e.g., BC-250 with llama-server running),
    this WILL cause OOM kills. Disabled by default.

The environment variable ENABLE_MARKER_PDF=1 enables Tier 2.
When disabled (default), only PyMuPDF is used.
"""

import os

# --- Configuration ---
ENABLE_MARKER = os.environ.get("ENABLE_MARKER_PDF", "0") == "1"

# Force CPU-only mode for Marker (if enabled) to avoid GPU memory conflicts
if ENABLE_MARKER:
    os.environ["TORCH_DEVICE"] = "cpu"
    os.environ["INFERENCE_RAM"] = "4"
    os.environ["VRAM_PER_MODEL"] = "0"

# --- Lazy Singleton for Marker Models ---
_cached_model_dict = None


def _extract_with_pymupdf(filepath: str) -> str:
    """Fast, lightweight PDF text extraction using PyMuPDF.
    
    Uses ~5MB of RAM regardless of PDF size. Handles text-based PDFs
    with near-perfect accuracy. Falls short on scanned/image-only PDFs
    and complex table layouts — but won't crash your system.
    """
    try:
        import pymupdf  # PyMuPDF package
    except ImportError:
        print("[PDF Parser] PyMuPDF not installed. Run: pip install pymupdf")
        return ""
    
    print("[PDF Parser] Extracting text with PyMuPDF (lightweight mode)...")
    
    try:
        doc = pymupdf.open(filepath)
        pages = []
        total_images = 0
        
        for page_num, page in enumerate(doc):
            # 1. Standard Extraction (with sort=True for better reconstruction)
            text = page.get_text("text", sort=True)
            
            # 2. Heuristic Scavenger Fallback
            if not text.strip():
                # If standard text is empty, try "blocks" – sometimes catches misaligned text objects
                blocks = page.get_text("blocks", sort=True)
                if blocks:
                    print(f"[PDF Parser] 🔍 Page {page_num + 1} empty via standard extraction. Attempting block scavenge...")
                    extracted_blocks = [b[4].strip() for b in blocks if b[4].strip()]
                    if extracted_blocks:
                        text = "\n".join(extracted_blocks)
            
            # 3. Final Fallback: "words" – extremely aggressive extraction
            if not text.strip():
                words = page.get_text("words", sort=True)
                if words:
                    print(f"[PDF Parser] 🔍 Page {page_num + 1} still empty. Attempting raw word scavenge...")
                    text = " ".join([w[4] for w in words])

            # Check for images if page is STILL empty (to suggest OCR)
            if not text.strip():
                img_list = page.get_images(full=True)
                total_images += len(img_list)

            if text.strip():
                pages.append(f"## Page {page_num + 1}\n\n{text.strip()}")
        
        doc.close()
        
        full_text = "\n\n".join(pages)
        
        if full_text:
            print(f"[PDF Parser] ✅ PyMuPDF extracted {len(full_text)} chars from {len(pages)} pages.")
        else:
            if total_images > 0:
                print(f"[PDF Parser] ⚠️ Found {total_images} images but 0 characters. This is likely a SCANNED PDF.")
                print("[PDF Parser] TIP: Enable Marker OCR with 'export ENABLE_MARKER_PDF=1' if you have 2GB+ RAM.")
            else:
                print("[PDF Parser] ⚠️ PyMuPDF found no extractable text or images.")
        
        return full_text
        
    except Exception as e:
        print(f"[PDF Parser] ❌ PyMuPDF extraction failed: {e}")
        return ""


def _extract_with_marker(filepath: str) -> str:
    """High-accuracy PDF extraction using Marker's deep learning pipeline.
    
    WARNING: Requires ~1.3GB RAM for model weights + additional for inference.
    On memory-constrained systems, this can trigger OOM kills.
    """
    global _cached_model_dict
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:
        print("[PDF Parser] Marker not installed. Falling back to PyMuPDF.")
        return ""
    
    # Load models once, cache for session
    if _cached_model_dict is None:
        print("[PDF Parser] Loading Surya deep learning models (first time only)...")
        _cached_model_dict = create_model_dict()
        print("[PDF Parser] ✅ Models cached for session.")
    else:
        print("[PDF Parser] Using cached Surya models.")
    
    converter = PdfConverter(artifact_dict=_cached_model_dict)
    
    print("[PDF Parser] Running deep learning inference on PDF pages...")
    rendered = converter(filepath)
    text, _, _ = text_from_rendered(rendered)
    
    if text:
        print(f"[PDF Parser] ✅ Marker extracted {len(text)} chars of precision Markdown.")
    else:
        print("[PDF Parser] ⚠️ Marker extraction yielded empty result.")
    
    return text if text else ""


def extract_markdown_from_pdf(filepath: str) -> str:
    """Extracts text from a PDF using the best available strategy.
    
    Default: PyMuPDF (fast, safe, low memory).
    Optional: Marker (deep learning, high accuracy, high memory cost).
    
    Set ENABLE_MARKER_PDF=1 in environment to use Marker.
    If Marker fails, automatically falls back to PyMuPDF.
    """
    print(f"[PDF Parser] Processing: {filepath}")
    
    if ENABLE_MARKER:
        print("[PDF Parser] 🔬 Marker mode enabled — attempting deep learning extraction...")
        try:
            result = _extract_with_marker(filepath)
            if result:
                return result
            # If Marker returns empty, fall through to PyMuPDF
            print("[PDF Parser] Marker returned empty. Falling back to PyMuPDF...")
        except Exception as e:
            print(f"[PDF Parser] ⚠️ Marker failed ({e}). Falling back to PyMuPDF...")
    
    # Default path: lightweight extraction
    return _extract_with_pymupdf(filepath)
