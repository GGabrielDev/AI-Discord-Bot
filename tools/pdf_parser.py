import os

# IMPORTANT: Force PyTorch/Marker to use the CPU exclusively.
# The user only has ~2.75GB of unified memory remaining after llama-server.
# If Marker attempts to load its models (up to 5GB) into the unified/GPU context,
# it will OOM and freeze the OS. The CPU mode is slower but dramatically safer.
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["INFERENCE_RAM"] = "4" # Optional: tell it to be mindful of RAM
os.environ["VRAM_PER_MODEL"] = "0" # Disable VRAM completely

def extract_markdown_from_pdf(filepath: str) -> str:
    """Uses datalab-to/marker to convert a PDF into fully structured Markdown.
    
    This function leverages deep learning vision models (Surya) via the CPU
    to parse formatting, tables, and equations with extremely high accuracy.
    """
    print(f"[PDF Parser] Initializing Marker (CPU Mode) for {filepath}")
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:
        print("[PDF Parser] Marker not installed. Run: pip install marker-pdf[full]")
        return ""

    print("[PDF Parser] Loading Layout Models (this may take a moment)...")
    
    # create_model_dict loads the Surya/LayoutLM weights into cpu RAM
    artifact_dict = create_model_dict()
    
    converter = PdfConverter(artifact_dict=artifact_dict)
    
    print("[PDF Parser] Converting PDF pages to Markdown...")
    rendered = converter(filepath)
    
    # We only care about the markdown text right now (discard metadata and image objects)
    text, _, _ = text_from_rendered(rendered)
    
    if text:
        print(f"[PDF Parser] Successfully extracted {len(text)} characters of precision Markdown.")
    else:
        print("[PDF Parser] Extraction yielded an empty result.")
        
    return text if text else ""
