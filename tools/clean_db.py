import os
import sys
import argparse
from storage.vectordb import VectorDB
import chromadb
import asyncio

async def clean_collection(collection_name: str, topic_name: str, dry_run: bool = True):
    """Identifies and removes duplicate URL entries in a ChromaDB collection."""
    print(f"\n--- Cleaning Collection: {collection_name} ---")
    db = VectorDB(collection_name=topic_name)
    
    # Get all metadata to identify duplicates (offload to thread)
    all_data = await asyncio.to_thread(db.collection.get, include=["metadatas"])
    if not all_data or not all_data['metadatas']:
        print("Collection is empty.")
        return

    # Map: URL -> List of (chunk_id, timestamp)
    url_map = {}
    for i, meta in enumerate(all_data['metadatas']):
        if not meta: continue
        url = meta.get("source")
        if not url: continue
        
        chunk_id = all_data['ids'][i]
        timestamp = int(meta.get("timestamp", 0))
        
        if url not in url_map:
            url_map[url] = []
        url_map[url].append((chunk_id, timestamp))

    to_delete = []
    total_sources = len(url_map)
    duplicates_found = 0

    for url, entries in url_map.items():
        # Check if this URL has chunks from different processing sessions
        # (Chroma chunk IDs are formatted as 'URL_chunk_index')
        # We group by the 'base' index if possible, but the best way is to 
        # check if multiple 'chunk 0's exist, or if timestamps differ significantly.
        
        # Simpler approach: Keep the chunks associated with the LATEST timestamp for each URL.
        entries.sort(key=lambda x: x[1], reverse=True)
        latest_ts = entries[0][1]
        
        # Mark everything that doesn't match the latest timestamp for deletion
        for chunk_id, ts in entries:
            if ts < latest_ts:
                to_delete.append(chunk_id)
        
        if len(set(e[1] for e in entries)) > 1:
            duplicates_found += 1

    print(f"Total Unique Sources: {total_sources}")
    print(f"Sources with duplicates: {duplicates_found}")
    print(f"Redundant chunks to remove: {len(to_delete)}")

    if not dry_run and to_delete:
        print(f"PROCEEDING WITH DELETION of {len(to_delete)} chunks...")
        # Chroma handles deletion in batches (offload to thread)
        batch_size = 100
        for i in range(0, len(to_delete), batch_size):
            await asyncio.to_thread(db.collection.delete, ids=to_delete[i:i + batch_size])
        print("Deletion complete.")
    elif to_delete:
        print("DRY RUN: No files were harmed. Use --execute to apply changes.")

async def main_async():
    parser = argparse.ArgumentParser(description="ChromaDB Maintenance: Prune duplicate URL entries.")
    parser.add_argument("--topic", type=str, help="Specific topic (collection) to clean. If omitted, cleans all.")
    parser.add_argument("--execute", action="store_true", help="Apply deletions (default is dry-run).")
    args = parser.parse_args()

    # List all collections
    client = chromadb.PersistentClient(path="./chroma_data")
    collections = client.list_collections()

    if args.topic:
        target = [c for c in collections if c.name == args.topic or c.name == args.topic.replace(" ", "_")]
        if not target:
            print(f"Error: Collection '{args.topic}' not found.")
            sys.exit(1)
        await clean_collection(target[0].name, args.topic, not args.execute)
    else:
        for col in collections:
            await clean_collection(col.name, col.name, not args.execute)

if __name__ == "__main__":
    asyncio.run(main_async())
