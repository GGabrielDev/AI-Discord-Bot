import chromadb
import re
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

    def add_chunks(self, chunks: list[str], source_url: str):
        """Saves a list of text chunks into the database."""
        if not chunks:
            return

        # Chroma requires a unique ID for every single chunk
        ids = [f"{source_url}_chunk_{i}" for i in range(len(chunks))]
        
        # Metadata lets us track where the AI got the information
        metadatas = [{"source": source_url} for _ in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[VectorDB] Successfully stored {len(chunks)} chunks from {source_url}")

    def search(self, query: str, n_results: int = 3):
        """Searches the database for the most relevant text chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # ChromaDB returns a complex dictionary; we just want the text
        if results and results['documents']:
            return results['documents'][0]
        
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
        if total > 0:
            all_meta = self.collection.get(include=["metadatas"])
            if all_meta and all_meta['metadatas']:
                for meta in all_meta['metadatas']:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
        
        return {
            "total_chunks": total,
            "unique_sources": len(sources),
            "source_urls": list(sources)
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
    # Let's save a fake chunk of data
    db.add_chunks(["The Venezuelan Central Bank recently updated its reserve policies."], "http://test-url.com")
    
    # Now let's try to search for it
    print("\n--- Testing Search ---")
    results = db.search("What is the central bank doing?")
    for res in results:
        print(f"Found: {res}")
    
    # Test new methods
    print("\n--- Testing Sample ---")
    sample = db.get_sample(5)
    print(f"Got {len(sample)} sample chunks")
    
    print("\n--- Testing Stats ---")
    stats = db.get_collection_stats()
    print(f"Stats: {stats}")
