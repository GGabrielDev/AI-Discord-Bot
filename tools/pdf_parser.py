"""PDF text extraction with lightweight-first triage for constrained hardware."""

import os
from typing import Dict, Optional, Tuple

# --- Configuration ---
ENABLE_MARKER = os.environ.get("ENABLE_MARKER_PDF", "0") == "1"

# Force CPU-only mode for Marker (if enabled) to avoid GPU memory conflicts
if ENABLE_MARKER:
    os.environ["TORCH_DEVICE"] = "cpu"
    os.environ["INFERENCE_RAM"] = "4"
    os.environ["VRAM_PER_MODEL"] = "0"

# --- Lazy Singleton for Marker Models ---
_cached_model_dict = None
_PDF_TRIAGE_PROFILES = {
    "low-memory": {
        "marker_max_mb": 0,
        "enough_text_chars": 800,
        "enough_text_coverage": 0.35,
        "sparse_chars_per_page": 40,
        "max_sparse_text_coverage": 0.35,
        "min_image_only_pages": 1,
    },
    "balanced": {
        "marker_max_mb": 12,
        "enough_text_chars": 1200,
        "enough_text_coverage": 0.45,
        "sparse_chars_per_page": 90,
        "max_sparse_text_coverage": 0.55,
        "min_image_only_pages": 1,
    },
    "max-recall": {
        "marker_max_mb": 32,
        "enough_text_chars": 1600,
        "enough_text_coverage": 0.6,
        "sparse_chars_per_page": 160,
        "max_sparse_text_coverage": 0.8,
        "min_image_only_pages": 1,
    },
}


def _normalize_resource_profile(resource_profile: Optional[str]) -> str:
    profile = (resource_profile or os.environ.get("RESOURCE_PROFILE", "low-memory")).strip().lower()
    if profile not in _PDF_TRIAGE_PROFILES:
        return "low-memory"
    return profile


def _extract_with_pymupdf(filepath: str) -> Tuple[str, Dict[str, float]]:
    """Fast, lightweight PDF text extraction using PyMuPDF.
    
    Uses ~5MB of RAM regardless of PDF size. Handles text-based PDFs
    with near-perfect accuracy. Falls short on scanned/image-only PDFs
    and complex table layouts — but won't crash your system.
    """
    stats: Dict[str, float] = {
        "pages_seen": 0,
        "pages_with_text": 0,
        "text_chars": 0,
        "avg_chars_per_page": 0.0,
        "text_coverage": 0.0,
        "total_images": 0,
        "image_only_pages": 0,
    }
    try:
        import pymupdf  # PyMuPDF package
    except ImportError:
        print("[PDF Parser] PyMuPDF not installed. Run: pip install pymupdf")
        return "", stats
    
    print("[PDF Parser] Extracting text with PyMuPDF (lightweight mode)...")
    
    doc = None
    try:
        doc = pymupdf.open(filepath)
        pages = []
        
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

            img_list = page.get_images(full=True)
            stats["total_images"] += len(img_list)
            stats["pages_seen"] += 1

            if text.strip():
                stats["pages_with_text"] += 1
                pages.append(f"## Page {page_num + 1}\n\n{text.strip()}")
            elif img_list:
                stats["image_only_pages"] += 1
        
        full_text = "\n\n".join(pages)
        stats["text_chars"] = len(full_text)
        if stats["pages_seen"]:
            stats["avg_chars_per_page"] = stats["text_chars"] / stats["pages_seen"]
            stats["text_coverage"] = stats["pages_with_text"] / stats["pages_seen"]
        
        if full_text:
            print(f"[PDF Parser] ✅ PyMuPDF extracted {len(full_text)} chars from {len(pages)} pages.")
        else:
            if stats["total_images"] > 0:
                print(f"[PDF Parser] ⚠️ Found {int(stats['total_images'])} images but 0 characters. This is likely a SCANNED PDF.")
                print("[PDF Parser] TIP: Enable Marker OCR with 'export ENABLE_MARKER_PDF=1' if you have 2GB+ RAM.")
            else:
                print("[PDF Parser] ⚠️ PyMuPDF found no extractable text or images.")
        
        return full_text, stats
        
    except Exception as e:
        print(f"[PDF Parser] ❌ PyMuPDF extraction failed: {e}")
        return "", stats
    finally:
        if doc is not None:
            doc.close()


def _should_escalate_to_marker(
    stats: Dict[str, float],
    filepath: str,
    source_bytes: Optional[int] = None,
    resource_profile: Optional[str] = None,
) -> Tuple[bool, str]:
    profile = _normalize_resource_profile(resource_profile)
    profile_budget = _PDF_TRIAGE_PROFILES[profile]

    if not ENABLE_MARKER:
        return False, "marker-disabled"

    marker_max_mb = profile_budget["marker_max_mb"]
    if marker_max_mb <= 0:
        return False, f"{profile}-disables-marker"

    pdf_size = source_bytes if source_bytes is not None else os.path.getsize(filepath)
    if pdf_size > marker_max_mb * 1024 * 1024:
        return False, f"pdf-too-large-for-{profile}"

    pages_seen = int(stats.get("pages_seen", 0) or 0)
    if pages_seen == 0:
        return False, "no-pages-read"

    text_chars = int(stats.get("text_chars", 0) or 0)
    text_coverage = float(stats.get("text_coverage", 0.0) or 0.0)
    avg_chars_per_page = float(stats.get("avg_chars_per_page", 0.0) or 0.0)
    image_only_pages = int(stats.get("image_only_pages", 0) or 0)
    total_images = int(stats.get("total_images", 0) or 0)

    if (
        text_chars >= profile_budget["enough_text_chars"]
        and text_coverage >= profile_budget["enough_text_coverage"]
    ):
        return False, "lightweight-text-sufficient"

    if image_only_pages >= profile_budget["min_image_only_pages"]:
        return True, "image-only-pages-detected"

    if text_chars == 0 and total_images > 0:
        return True, "image-heavy-empty-text"

    if (
        avg_chars_per_page <= profile_budget["sparse_chars_per_page"]
        and text_coverage <= profile_budget["max_sparse_text_coverage"]
    ):
        return True, "sparse-text-density"

    return False, "lightweight-output-accepted"


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


def extract_markdown_from_pdf(
    filepath: str,
    source_bytes: Optional[int] = None,
    resource_profile: Optional[str] = None,
) -> str:
    """Extracts text from a PDF using the best available strategy.
    
    Default: PyMuPDF (fast, safe, low memory).
    Optional: Marker (deep learning, high accuracy, high memory cost) only
    after lightweight triage indicates the extra cost is justified.
    
    Set ENABLE_MARKER_PDF=1 in environment to use Marker.
    If Marker fails, automatically falls back to PyMuPDF.
    """
    print(f"[PDF Parser] Processing: {filepath}")

    light_text, light_stats = _extract_with_pymupdf(filepath)
    should_escalate, reason = _should_escalate_to_marker(
        light_stats,
        filepath,
        source_bytes=source_bytes,
        resource_profile=resource_profile,
    )

    if not should_escalate:
        print(f"[PDF Parser] Triage kept lightweight path ({reason}).")
        return light_text

    print(f"[PDF Parser] 🔬 Triage escalating to Marker ({reason}).")
    try:
        result = _extract_with_marker(filepath)
        if result:
            return result
        print("[PDF Parser] Marker returned empty. Falling back to PyMuPDF result...")
    except Exception as e:
        print(f"[PDF Parser] ⚠️ Marker failed ({e}). Falling back to PyMuPDF result...")

    return light_text
