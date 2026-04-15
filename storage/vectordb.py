import chromadb
import re
import time
from config.settings import CHROMA_DB_PATH

class VectorDB:
    def __init__(self, collection_name="research_data"):
        # CLEAN THE NAME: 
        # 1. Replace anything that isn't a letter, number, dot, or dash with an underscore
        # 2. Ensure it doesn't start or end with a symbol
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name).strip('_')
        
        # Ensure name is at least 3 chars
        if len(safe_name) < 3:
            safe_name = f"topic_{safe_name}" if len(safe_name) > 0 else "default_collection"

        print(f"[VectorDB] Connecting to storage at {CHROMA_DB_PATH} using safe name: {safe_name}")
        
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[str], source_url: str, chunk_type: str = "summary", source_title: str = ""):
        """Saves a list of text chunks into the database with rich metadata.
        
        Args:
            chunks: List of text strings to store
            source_url: URL where the content originated
            chunk_type: 'summary' for LLM-analyzed content, 'raw' for unprocessed text
            source_title: Title of the source page (if available)
        """
        if not chunks:
            return

        timestamp = str(int(time.time()))
        
        # Chroma requires a unique ID for every single chunk
        ids = [f"{source_url}_chunk_{i}" for i in range(len(chunks))]
        
        # Rich metadata for filtering and provenance tracking
        metadatas = [{
            "source": source_url,
            "chunk_type": chunk_type,
            "source_title": source_title,
            "timestamp": timestamp,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[VectorDB] Stored {len(chunks)} {chunk_type} chunks from {source_url}")

    def search(self, query: str, n_results: int = 5, chunk_type: str = None):
        """Searches the database for the most relevant text chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            chunk_type: Filter by type ('summary' or 'raw'). None = all types.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        if chunk_type:
            kwargs["where"] = {"chunk_type": chunk_type}
        
        results = self.collection.query(**kwargs)
        
        # ChromaDB returns a complex dictionary; we just want the text
        if results and results['documents']:
            return results['documents'][0]
        
        return []

    def search_with_metadata(self, query: str, n_results: int = 5):
        """Search and return both documents and their metadata."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        
        if results and results['documents'] and results['metadatas']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            return list(zip(docs, metas))
        
        return []

    def get_sample(self, n_samples: int = 10) -> list[str]:
        """Returns a diverse sample of stored knowledge for evaluation.
        
        Used by the re-planner to understand what the agent has learned so far.
        """
        total = self.collection.count()
        if total == 0:
            return []
        
        # Get up to n_samples documents, spread across the collection
        n = min(n_samples, total)
        results = self.collection.get(
            limit=n,
            include=["documents"]
        )
        
        if results and results['documents']:
            return results['documents']
        return []

    def get_collection_stats(self) -> dict:
        """Returns stats about the current collection for the re-planner."""
        total = self.collection.count()
        
        # Get unique source URLs from metadata
        sources = set()
        chunk_types = {}
        if total > 0:
            all_meta = self.collection.get(include=["metadatas"])
            if all_meta and all_meta['metadatas']:
                for meta in all_meta['metadatas']:
                    if meta:
                        if "source" in meta:
                            sources.add(meta["source"])
                        ct = meta.get("chunk_type", "unknown")
                        chunk_types[ct] = chunk_types.get(ct, 0) + 1
        
        return {
            "total_chunks": total,
            "unique_sources": len(sources),
            "source_urls": list(sources),
            "chunk_types": chunk_types
        }

    def delete_topic(self, collection_name: str):
        """Removes a specific collection from the database."""
        try:
            self.client.delete_collection(name=collection_name)
            print(f"[VectorDB] Collection '{collection_name}' deleted.")
        except Exception as e:
            print(f"[VectorDB] Error deleting collection: {e}")

# Quick manual test if you run this file directly
if __name__ == "__main__":
    db = VectorDB()
    db.add_chunks(
        ["The Venezuelan Central Bank recently updated its reserve policies."], 
        "http://test-url.com",
        chunk_type="summary",
        source_title="Test Article"
    )
    
    print("\n--- Testing Search ---")
    results = db.search("What is the central bank doing?")
    for res in results:
        print(f"Found: {res}")
    
    print("\n--- Testing Search with Metadata ---")
    results_meta = db.search_with_metadata("central bank")
    for doc, meta in results_meta:
        print(f"Doc: {doc[:80]}... | Source: {meta['source']} | Type: {meta['chunk_type']}")
    
    print("\n--- Testing Stats ---")
    stats = db.get_collection_stats()
    print(f"Stats: {stats}")
