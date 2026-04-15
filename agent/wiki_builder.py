import os
import re
from datetime import datetime

# The root directory for the Markdown Wikipedia
WIKI_ROOT = "knowledge_base"

def sanitize_filename(name: str) -> str:
    """Removes weird characters to make a safe directory/file name."""
    safe = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name)
    return safe.strip().replace(" ", "_").lower()

def generate_index_page():
    """Generates the master index.md based on what's currently in the knowledge base."""
    # Ensure root exists
    os.makedirs(WIKI_ROOT, exist_ok=True)
    
    index_path = os.path.join(WIKI_ROOT, "index.md")
    
    lines = [
        "# Research Knowledge Base",
        "\n*Generated autonomously by your AI Research Agent.*\n",
        "## Topics Master List\n"
    ]
    
    # Scan the WIKI_ROOT for topic directories
    topics = [d for d in os.listdir(WIKI_ROOT) if os.path.isdir(os.path.join(WIKI_ROOT, d))]
    topics.sort()
    
    if not topics:
        lines.append("No research topics stored yet.")
    else:
        for topic in topics:
            human_topic = topic.replace("_", " ").title()
            lines.append(f"### {human_topic}")
            
            topic_dir = os.path.join(WIKI_ROOT, topic)
            articles = [f for f in os.listdir(topic_dir) if f.endswith(".md")]
            articles.sort()
            
            if not articles:
                lines.append("- *(No specific source articles found)*\n")
            else:
                for article in articles:
                    rel_path = f"{topic}/{article}"
                    clean_title = article.replace(".md", "").replace("_", " ").title()
                    # Fallback to general link if the title gets mangled
                    lines.append(f"- [{clean_title}]({rel_path})")
            lines.append("\n") # Blank line between topics

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"[WikiBuilder] Regenerated master index at {index_path}")

def store_article(subject: str, url: str, summary_text: str):
    """Saves a beautiful Markdown file for the specific source in the topic folder."""
    # Create the nested directory
    safe_topic = sanitize_filename(subject)
    topic_dir = os.path.join(WIKI_ROOT, safe_topic)
    os.makedirs(topic_dir, exist_ok=True)
    
    # Attempt to derive a safe, somewhat meaningful filename from the URL, or use a timestamp
    # E.g. https://domain.com/path/to/article -> path_to_article.md
    url_parts = [p for p in url.split("/") if p]
    if url_parts:
        filename_base = sanitize_filename(url_parts[-1])
        if len(filename_base) < 3: 
            # If it was just a domain like 'domain.com/'
            filename_base = sanitize_filename(url_parts[1]) if len(url_parts) > 1 else "source"
    else:
        filename_base = "source"
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # To prevent overwriting, append timestamp
    filename = f"{filename_base}_{timestamp}.md"
    
    filepath = os.path.join(topic_dir, filename)
    
    content = [
        f"# Analyzed Data: {subject.title()}",
        "",
        f"**Source URL:** [{url}]({url})",
        f"**Data Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## AI Summary & Extraction",
        "",
        summary_text,
        "",
        "---",
        "*Stored dynamically during Autonomous Research Pipeline.*"
    ]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
        
    print(f"[WikiBuilder] Wrote markdown showcase to {filepath}")
    
    # Every time a new article is stored, regenerate the master index
    generate_index_page()
