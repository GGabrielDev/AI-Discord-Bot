import chromadb
import re
import time
import asyncio
from urllib.parse import urlparse
from config.settings import (
    CHROMA_DB_PATH,
    RAW_CHUNK_SOFT_CAP,
    RAW_DEFAULT_MAX_CHUNKS_PER_SOURCE,
    RAW_HIGH_VALUE_MAX_CHUNKS_PER_SOURCE,
    RAW_SMALL_SOURCE_MAX_CHUNKS,
    RESOURCE_PROFILE,
)

_RAW_WORTHY_HINTS = (
    "/api",
    "/appendix",
    "/annex",
    "/compliance",
    "/code",
    "/docs",
    "/documentation",
    "/guide",
    "/law",
    "/manual",
    "/policy",
    "/protocol",
    "/reference",
    "/regulation",
    "/rfc",
    "/schema",
    "/spec",
    "/standard",
)

def _derive_source_domain(source_url: str) -> str:
    try:
        return urlparse(source_url).netloc.lower()
    except Exception:
        return ""

def _derive_path_depth(source_url: str) -> int:
    try:
        path = urlparse(source_url).path.strip("/")
        return len([segment for segment in path.split("/") if segment])
    except Exception:
        return 0

def _is_raw_worthy_source(source_url: str) -> bool:
    try:
        parsed = urlparse(source_url)
        normalized = f"{parsed.netloc}{parsed.path}".lower()
    except Exception:
        normalized = source_url.lower()
    if normalized.endswith(".pdf"):
        return True
    return any(hint in normalized for hint in _RAW_WORTHY_HINTS)

def _select_representative_chunks(chunks: list[str], max_chunks: int) -> list[str]:
    if max_chunks <= 0 or not chunks:
        return []
    if len(chunks) <= max_chunks:
        return list(chunks)
    if max_chunks == 1:
        return [chunks[0]]

    total = len(chunks)
    selected_indexes = []
    seen = set()
    for i in range(max_chunks):
        idx = round(i * (total - 1) / (max_chunks - 1))
        if idx in seen:
            right = idx
            while right < total and right in seen:
                right += 1
            if right < total:
                idx = right
            else:
                left = idx
                while left >= 0 and left in seen:
                    left -= 1
                idx = max(0, left)
        seen.add(idx)
        selected_indexes.append(idx)
    return [chunks[i] for i in sorted(selected_indexes)]

def _plan_raw_chunk_retention(source_url: str, raw_chunk_count: int, existing_raw_chunks: int) -> dict:
    pressure_ratio = (existing_raw_chunks / RAW_CHUNK_SOFT_CAP) if RAW_CHUNK_SOFT_CAP > 0 else 0.0
    is_small_source = raw_chunk_count <= RAW_SMALL_SOURCE_MAX_CHUNKS
    is_raw_worthy = _is_raw_worthy_source(source_url)

    if is_small_source:
        kept = raw_chunk_count
        tier = "full"
        reason = "small-source"
    else:
        base_cap = RAW_HIGH_VALUE_MAX_CHUNKS_PER_SOURCE if is_raw_worthy else RAW_DEFAULT_MAX_CHUNKS_PER_SOURCE
        if pressure_ratio >= 1.0 and not is_raw_worthy:
            kept = 0
            tier = "summary-only"
            reason = "soft-cap-pressure"
        else:
            pressure_factor = 1.0
            if pressure_ratio >= 1.0:
                pressure_factor = 0.5
            elif pressure_ratio >= 0.75:
                pressure_factor = 0.75 if is_raw_worthy else 0.5
            kept = min(raw_chunk_count, max(1, int(round(base_cap * pressure_factor))))
            if kept >= raw_chunk_count:
                tier = "full"
                reason = "high-value" if is_raw_worthy else "within-cap"
            else:
                tier = "capped"
                if pressure_ratio >= 0.75:
                    reason = "pressure-capped"
                elif is_raw_worthy:
                    reason = "high-value-capped"
                else:
                    reason = "default-cap"

    return {
        "raw_storage_profile": RESOURCE_PROFILE,
        "raw_storage_tier": tier,
        "raw_storage_reason": reason,
        "raw_storage_soft_cap": RAW_CHUNK_SOFT_CAP,
        "raw_storage_pressure": round(pressure_ratio, 3),
        "source_raw_chunks_planned": raw_chunk_count,
        "source_raw_chunks_stored": kept,
        "source_has_raw": 1 if kept > 0 else 0,
        "source_raw_worthy": 1 if is_raw_worthy else 0,
    }

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
        self._raw_chunk_count_cache = None

    async def _get_raw_chunk_count(self) -> int:
        if self._raw_chunk_count_cache is None:
            results = await asyncio.to_thread(
                self.collection.get,
                where={"chunk_type": "raw"},
                include=[]
            )
            self._raw_chunk_count_cache = len(results.get("ids", []))
        return self._raw_chunk_count_cache

    async def plan_raw_chunks(self, chunks: list[str], source_url: str) -> tuple[list[str], dict]:
        if not chunks:
            return [], _plan_raw_chunk_retention(source_url, 0, 0)

        existing_raw_chunks = await self._get_raw_chunk_count()
        policy = _plan_raw_chunk_retention(source_url, len(chunks), existing_raw_chunks)
        kept_chunks = _select_representative_chunks(chunks, policy["source_raw_chunks_stored"])
        return kept_chunks, policy

    async def add_chunks(self, chunks: list[str], source_url: str, chunk_type: str = "summary", source_title: str = "", extra_metadata: dict | None = None):
        """Saves a list of text chunks into the database with rich metadata."""
        if not chunks:
            return

        timestamp = str(int(time.time()))
        ids = [f"{source_url}_{chunk_type}_chunk_{i}" for i in range(len(chunks))]
        source_domain = _derive_source_domain(source_url)
        source_path_depth = _derive_path_depth(source_url)
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source_url,
                "chunk_type": chunk_type,
                "source_title": source_title,
                "timestamp": timestamp,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_word_count": len(chunk.split()),
                "chunk_char_count": len(chunk),
                "source_domain": source_domain,
                "source_path_depth": source_path_depth
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            metadatas.append(metadata)

        # Thread offload for blocking local embedding generation
        await asyncio.to_thread(
            self.collection.add,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        if chunk_type == "raw" and self._raw_chunk_count_cache is not None:
            self._raw_chunk_count_cache += len(chunks)
        print(f"[VectorDB] Stored {len(chunks)} {chunk_type} chunks from {source_url}")

    async def has_source(self, source_url: str) -> bool:
        """Checks if any information from this URL already exists in the collection."""
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                where={"source": source_url},
                limit=1,
                include=[]
            )
            return len(results['ids']) > 0
        except Exception:
            return False

    async def get_chunks_by_source(self, source_url: str) -> list[tuple[str, dict]]:
        """Retrieves all chunks and metadata associated with a specific URL."""
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                where={"source": source_url},
                include=["documents", "metadatas"]
            )
            if results and results['documents']:
                return list(zip(results['documents'], results['metadatas']))
            return []
        except Exception as e:
            print(f"[VectorDB] Error retrieving source {source_url}: {e}")
            return []

    async def search(self, query: str, n_results: int = 5, chunk_type: str = None):
        """Searches the database for the most relevant text chunks."""
        kwargs = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        if chunk_type:
            kwargs["where"] = {"chunk_type": chunk_type}
        
        results = await asyncio.to_thread(self.collection.query, **kwargs)
        
        if results and results['documents']:
            return results['documents'][0]
        
        return []

    async def search_with_metadata(self, query: str, n_results: int = 5, where: dict = None, include_distances: bool = False):
        """Search and return both documents and their metadata, with optional filtering."""
        include_fields = ["documents", "metadatas"]
        if include_distances:
            include_fields.append("distances")

        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": include_fields
        }
        
        if where:
            kwargs["where"] = where
            
        results = await asyncio.to_thread(self.collection.query, **kwargs)
        
        if results and results['documents'] and results['metadatas']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            if include_distances:
                raw_distances = results.get("distances", [[]])
                distances = raw_distances[0] if raw_distances else []
                if len(distances) < len(docs):
                    distances = list(distances) + [None] * (len(docs) - len(distances))
                return list(zip(docs, metas, distances))
            return list(zip(docs, metas))
        
        return []

    async def get_all_chunks(self) -> dict:
        """Returns all chunk ids, documents, and metadata for maintenance tooling."""
        return await asyncio.to_thread(
            self.collection.get,
            include=["documents", "metadatas"]
        )

    async def update_chunk_metadata(self, ids: list[str], metadatas: list[dict]):
        """Updates metadata for existing chunks."""
        if not ids:
            return
        await asyncio.to_thread(
            self.collection.update,
            ids=ids,
            metadatas=metadatas
        )

    async def get_sample(self, n_samples: int = 10) -> list[str]:
        """Returns a diverse sample of stored knowledge for evaluation."""
        total = await asyncio.to_thread(self.collection.count)
        if total == 0:
            return []
        
        n = min(n_samples, total)
        results = await asyncio.to_thread(
            self.collection.get,
            limit=n,
            include=["documents"]
        )
        
        if results and results['documents']:
            return results['documents']
        return []

    async def get_collection_stats(self) -> dict:
        """Returns stats about the current collection for the re-planner."""
        total = await asyncio.to_thread(self.collection.count)
        
        sources = set()
        chunk_types = {}
        if total > 0:
            all_meta = await asyncio.to_thread(self.collection.get, include=["metadatas"])
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
    async def test():
        db = VectorDB(collection_name="test_collection")
        
        print("--- Testing Add ---")
        await db.add_chunks(["The central bank raised interest rates."], "https://test.com", "summary")
        
        print("\n--- Testing Search ---")
        results = await db.search("interest rates")
        print(f"Results: {results}")
        
        print("\n--- Testing Search with Metadata ---")
        results_meta = await db.search_with_metadata("central bank")
        for doc, meta in results_meta:
            print(f"Doc: {doc[:80]}... | Source: {meta['source']} | Type: {meta['chunk_type']}")
        
        print("\n--- Testing Stats ---")
        stats = await db.get_collection_stats()
        print(f"Stats: {stats}")

    asyncio.run(test())
