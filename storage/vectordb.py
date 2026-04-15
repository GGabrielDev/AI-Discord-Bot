import chromadb
from config.settings import CHROMA_DB_PATH

class VectorDB:
    def __init__(self, collection_name="research_data"):
        print(f"[VectorDB] Booting up storage at {CHROMA_DB_PATH}...")
        
        # Initialize the client with persistent storage so it survives reboots
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create a collection (think of it like a table in SQL)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Standard math for comparing text similarity
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
