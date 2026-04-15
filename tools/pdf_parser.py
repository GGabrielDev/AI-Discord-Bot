import os

# IMPORTANT: Force PyTorch/Marker to use the CPU exclusively.
# The BC-250 board has ~14.75GB of unified memory, most of which is claimed by llama-server.
# If Marker attempts to load its models (up to 5GB) into the unified/GPU context,
# it will OOM and freeze the OS. CPU mode is slower but dramatically safer.
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["INFERENCE_RAM"] = "4"     # Tell Marker to be mindful of RAM budget
os.environ["VRAM_PER_MODEL"] = "0"    # Disable VRAM completely

# --- Lazy Singleton Model Cache ---
# Marker's layout detection models (Surya OCR, LayoutLM, etc.) total ~1.3GB of weights.
# Loading them takes 20-40 seconds on CPU. Without caching, every single PDF processed
# during a research session would trigger a full reload — devastating on constrained hardware.
# This module-level variable holds the loaded models for the entire bot lifetime.
_cached_model_dict = None

def _get_or_load_models():
    """Loads Marker's deep learning models once and caches them for the bot session.
    
    First call: Downloads (if needed) and initializes all Surya/LayoutLM weights (~20-40s).
    Subsequent calls: Returns the cached dictionary instantly.
    """
    global _cached_model_dict
    
    if _cached_model_dict is not None:
        print("[PDF Parser] Using cached layout models (instant).")
        return _cached_model_dict
    
    print("[PDF Parser] First-time model initialization — loading Surya weights into CPU RAM...")
    print("[PDF Parser] (This only happens once per bot session. Subsequent PDFs will be instant.)")
    
    try:
        from marker.models import create_model_dict
        _cached_model_dict = create_model_dict()
        print("[PDF Parser] ✅ Models cached successfully. All future PDFs will skip this step.")
        return _cached_model_dict
    except Exception as e:
        print(f"[PDF Parser] ❌ Failed to load models: {e}")
        raise


def extract_markdown_from_pdf(filepath: str) -> str:
    """Uses datalab-to/marker to convert a PDF into fully structured Markdown.
    
    This function leverages deep learning vision models (Surya) via the CPU
    to parse formatting, tables, and equations with extremely high accuracy.
    The models are loaded once and cached for the entire bot session.
    """
    print(f"[PDF Parser] Processing: {filepath}")
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
    except ImportError:
        print("[PDF Parser] Marker not installed. Run: pip install marker-pdf[full]")
        return ""

    # Use the cached models — instant on 2nd+ PDF
    artifact_dict = _get_or_load_models()
    
    converter = PdfConverter(artifact_dict=artifact_dict)
    
    print("[PDF Parser] Converting PDF pages to Markdown...")
    rendered = converter(filepath)
    
    # We only care about the markdown text (discard metadata and image objects)
    text, _, _ = text_from_rendered(rendered)
    
    if text:
        print(f"[PDF Parser] ✅ Extracted {len(text)} characters of precision Markdown.")
    else:
        print("[PDF Parser] ⚠️ Extraction yielded an empty result.")
        
    return text if text else ""
